import torch
from torch import nn, Tensor
import torch.nn.functional as F
from model.logic import Logic

from util import match_shapes, recursive_binop, shuffle


def cosine_similarity(a, b, alpha=3):
    return torch.sigmoid(
        alpha
        * (a * b).sum(dim=-1)
        / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1) + 1e-10)
    )


class EmbedLogic(Logic):
    def __init__(self, embed_dims: int, calculate_reg=True):
        super().__init__()
        self.embed_dims = embed_dims

        self._not = EmbedNOT(self)
        self._and = EmbedAND(self)
        self._or = EmbedOR(self)

        self.calculate_reg = calculate_reg
        self._reg = torch.tensor(0.0)
        self.zero_reg()

    @property
    def device(self):
        return self._reg.device

    def T(self, device='cpu'):
        t = torch.zeros(self.embed_dims)
        t[0] = 1.0
        return t.to(device)

    def F(self, device='cpu'):
        return self.neg_no_reg(self.T(device=device))

    def encode(self, input: Tensor):
        return torch.where(
            input.bool().unsqueeze(-1), 
            self.T(device=input.device), 
            self.F(device=input.device)
        )

    def decode(self, output: Tensor):
        ws = torch.cat((
            cosine_similarity(output, self.T(device=output.device)).unsqueeze(-1), 
            cosine_similarity(output, self.F(device=output.device)).unsqueeze(-1)
        ), dim=-1)
        ws = torch.softmax(ws, dim=-1)
        return ws[..., 0]

    def _matrix_shape(self, input: Tensor):
        matrix_shape = [1 for _ in range(len(input.shape) + 1)]
        matrix_shape[-1] = self.embed_dims
        matrix_shape[-2] = self.embed_dims
        return tuple(matrix_shape)

    def _bias_shape(self, input):
        bias_shape = [1 for _ in range(len(input.shape))]
        bias_shape[-1] = self.embed_dims
        return tuple(bias_shape)

    def _binop_to_axis(self, input: Tensor, binop, dim: int) -> Tensor:
        last_dim = len(input.shape) - 1
        if dim == last_dim:
            raise IndexError("Cannot apply binary operation over embedding dimension.")
        #shuffled = shuffle(input, dim, fix_dim=(last_dim,))
        shuffled = input[(*((slice(None),)*dim), torch.randperm(input.size(dim)))]
        return recursive_binop(shuffled, binop, dim)

    def conjoin(self, xs: Tensor, dim: int) -> Tensor:
        return self._binop_to_axis(xs, self.bin_conjoin, dim)

    def bin_conjoin(self, a: Tensor, b: Tensor) -> Tensor:
        return self._and(a, b)

    def disjoin(self, xs: Tensor, dim: int) -> Tensor:
        return self._binop_to_axis(xs, self.bin_disjoin, dim)

    def bin_disjoin(self, a: Tensor, b: Tensor) -> Tensor:
        return self._or(a, b)

    def neg(self, a: Tensor) -> Tensor:
        return self._not(a)

    def neg_no_reg(self, a: Tensor) -> Tensor:
        return self._not.f(a)

    def xor(self, xs: Tensor, dim: int) -> Tensor:
        return self._binop_to_axis(xs, self.bin_xor, dim)

    def logic_reg(self):
        return self._reg

    def add_reg(self, reg):
        self._reg = self._reg.to(reg.device)
        self._reg += reg

    def zero_reg(self):
        with torch.no_grad():
            self._reg = torch.tensor(0.0).to(self._reg.device)


class EmbedOp(nn.Module):
    def __init__(self, logic: EmbedLogic):
        super().__init__()
        self._logic = [logic]

    @property
    def logic(self):
        return self._logic[0]

    def logic_reg(self, a_0=None):
        return torch.tensor(0)


class EmbedUnaryOp(EmbedOp):
    def __init__(self, logic: EmbedLogic):
        super().__init__(logic)
        embed_dims = logic.embed_dims

        self.model = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.Linear(embed_dims, embed_dims, bias=False)
        )

    @property
    def device(self):
        return self.w_1.device

    def f(self, a_0):
        if a_0.shape[-1] != self.logic.embed_dims:
            raise Exception(
                "Final dimension must represent embedding of size %s."
                % self.logic.embed_dims
            )

        return self.model(a_0)

    def forward(self, a_0):
        output = self.f(a_0)
        if self.logic.calculate_reg:
            self.logic.add_reg(self.logic_reg(a_0))
        return output


class EmbedNOT(EmbedUnaryOp):
    def logic_reg(self, a_0):
        T = self.logic.T(a_0.device)
        r_1 = cosine_similarity(a_0, self.f(a_0))
        r_1_ = cosine_similarity(T, self.f(T))
        r_2 = 1 - cosine_similarity(a_0, self.f(self.f(a_0)))
        return (r_1 + r_2).sum() + r_1_


class EmbedBinaryOp(EmbedOp):
    def __init__(self, logic: EmbedLogic):
        super().__init__(logic)
        embed_dims = logic.embed_dims

        self.l1_1 = nn.Linear(embed_dims, embed_dims)
        self.l1_2 = nn.Linear(embed_dims, embed_dims)
        self.l2 = nn.Linear(embed_dims, embed_dims, bias=False)


    def f(self, a_0_1, a_0_2):
        if (
            a_0_1.shape[-1] != self.logic.embed_dims
            or a_0_2.shape[-1] != self.logic.embed_dims
        ):
            raise Exception(
                "Final dimension must represent embedding of size %s."
                % self.logic.embed_dims
            )

        z_1 = self.l1_1(a_0_1) + self.l1_2(a_0_2)
        z_2 = self.l2(z_1)
        return z_2#F.normalize(z_2, dim=-1) 

    def forward(self, a_0_1, a_0_2):
        output = self.f(a_0_1, a_0_2)
        if self.logic.calculate_reg:
            self.logic.add_reg(self.logic_reg(a_0_1))
            self.logic.add_reg(self.logic_reg(a_0_2))
        return output


class EmbedAND(EmbedBinaryOp):
    def logic_reg(self, a_0):
        T = self.logic.T(a_0.device)
        F = self.logic.F(a_0.device)
        r_3 = 1 - cosine_similarity(self.f(a_0, T), a_0)
        r_4 = 1 - cosine_similarity(self.f(a_0, F), F)
        r_5 = 1 - cosine_similarity(self.f(a_0, a_0), a_0)
        r_6 = 1 - cosine_similarity(self.f(a_0, self.logic.neg_no_reg(a_0)), F)
        return (r_3 + r_4 + r_5 + r_6).sum()


class EmbedOR(EmbedBinaryOp):
    def logic_reg(self, a_0):
        T = self.logic.T(a_0.device)
        F = self.logic.F(a_0.device)
        r_7 = 1 - cosine_similarity(self.f(a_0, T), T)
        r_8 = 1 - cosine_similarity(self.f(a_0, F), a_0)
        r_9 = 1 - cosine_similarity(self.f(a_0, a_0), a_0)
        r_10 = 1 - cosine_similarity(self.f(a_0, self.logic.neg_no_reg(a_0)), T)
        return (r_7 + r_8 + r_9 + r_10).sum()

