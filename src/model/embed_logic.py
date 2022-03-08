import torch
from torch import nn, Tensor
from src.model.logic import Logic

from src.util import match_shapes, recursive_binop, shuffle


def cosine_similarity(a, b, alpha=10, dim=-1):
    return torch.sigmoid(
        alpha
        * torch.dot(a, b, dim=-1)
        / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1))
    )


class EmbedLogic(Logic):
    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims

        self._not = EmbedNOT(self)
        self._and = EmbedAND(self)
        self._or = EmbedOR(self)

    def T(self):
        t = torch.zeros(self.embed_dims)
        t[0] = 1.0
        return t

    def _matrix_shape(self, input):
        matrix_shape = tuple(1 for _ in range(len(input.shape) + 1))
        matrix_shape[-1] = self.embed_dims
        matrix_shape[-2] = self.embed_dims
        return matrix_shape

    def _bias_shape(self, input):
        bias_shape = tuple(1 for _ in range(len(input.shape)))
        bias_shape[-1] = self.embed_dims
        return bias_shape

    def _binop_to_axis(self, input: Tensor, binop, dim: int) -> Tensor:
        last_dim = len(input.shape) - 1
        if dim == last_dim:
            raise IndexError("Cannot apply binary operation over embedding dimension.")
        shuffled = shuffle(input, dim, fix_dim=(last_dim,))
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

    def xor(self, xs: Tensor, dim: int) -> Tensor:
        return self._binop_to_axis(xs, self.bin_xor, dim)


class EmbedOp(nn.Module):
    def logic_reg(self):
        return torch.tensor(0)


class EmbedUnaryOp(EmbedOp):
    def __init__(self, logic: EmbedLogic):
        self.logic = logic
        embed_dims = self.logic.embed_dims

        self.w_1 = nn.Parameter(
            torch.rand((embed_dims, embed_dims)), requires_grad=True
        )
        self.w_2 = nn.Parameter(
            torch.rand((embed_dims, embed_dims)), requires_grad=True
        )
        self.b_1 = nn.Parameter(torch.rand((embed_dims)), requires_grad=True)

    def forward(self, a_0):
        if a_0.shape[-1] != self.logic.embed_dims:
            raise Exception(
                "Final dimension must represent embedding of size %s."
                % self.logic.embed_dims
            )

        matrix_shape = self.logic._matrix_shape(a_0)
        w_1 = self.w_1.view(*matrix_shape)
        w_2 = self.w_2.view(*matrix_shape)

        bias_shape = self.logic._bias_shape(a_0)
        b_1 = self.b_1.view(*bias_shape)

        a_0 = a_0.unsqueeze(-1)
        z_1 = (a_0 * w_1).sum(-2) + b_1
        a_1 = torch.relu(z_1)
        a_1 = a_1.unsqueeze(-1)
        z_2 = (a_1 * w_2).sum(-2)
        return z_2


class EmbedNOT(EmbedUnaryOp):
    def logic_reg(self, a_0):
        T = self.logic.T()
        r_1 = cosine_similarity(a_0, self.forward(a_0))
        r_1_ = cosine_similarity(T, self.forward(T))
        r_2 = 1 - cosine_similarity(a_0, self.forward(self.forward(a_0)))
        return (r_1 + r_2).sum() + r_1_


class EmbedBinaryOp(EmbedOp):
    def __init__(self, logic: EmbedLogic):
        self.logic = logic
        embed_dims = self.logic.embed_dims

        self.w_1_1 = nn.Parameter(
            torch.rand((embed_dims, embed_dims)), requires_grad=True
        )
        self.w_1_2 = nn.Parameter(
            torch.rand((embed_dims, embed_dims)), requires_grad=True
        )
        self.w_2 = nn.Parameter(
            torch.rand((embed_dims, embed_dims)), requires_grad=True
        )
        self.b_1 = nn.Parameter(torch.rand((embed_dims)), requires_grad=True)

    def forward(self, a_0_1, a_0_2):
        if (
            a_0_1.shape[-1] != self.logic.embed_dims
            or a_0_2.shape[-1] != self.logic.embed_dims
        ):
            raise Exception(
                "Final dimension must represent embedding of size %s."
                % self.logic.embed_dims
            )

        if len(a_0_1.shape) == 1:
            a_0_1 = a_0_1.view(*[1 for _ in range(len(a_0_2.shape) - 1)], -1)
        elif len(a_0_2.shape) == 1:
            a_0_2 = a_0_2.view(*[1 for _ in range(len(a_0_1.shape) - 1)], -1)

        a_0_1, a_0_2 = match_shapes(a_0_1, a_0_2)

        matrix_shape = self.logic._matrix_shape(a_0_1)
        w_1_1 = self.w_1_1.view(*matrix_shape)
        w_1_2 = self.w_1_2.view(*matrix_shape)
        w_2 = self.w_2.view(*matrix_shape)

        bias_shape = self.logic._bias_shape(a_0_1)
        b_1 = self.b_1.view(*bias_shape)

        a_0_1 = a_0_1.unsqueeze(-1)
        a_0_2 = a_0_2.unsqueeze(-1)
        z_1 = (a_0_1 * w_1_1).sum(-2) + (a_0_2 * w_1_2).sum(-2) + b_1
        a_1 = torch.relu(z_1)
        a_1 = a_1.unsqueeze(-1)
        z_2 = (a_1 * w_2).sum(-2)
        return z_2


class EmbedAND(EmbedBinaryOp):
    def logic_reg(self, a_0):
        T = self.logic.T()
        F = self.neg(T)
        r_3 = 1 - cosine_similarity(self.forward(a_0, T), a_0)
        r_4 = 1 - cosine_similarity(self.forward(a_0, F), F)
        r_5 = 1 - cosine_similarity(self.forward(a_0, a_0), a_0)
        r_6 = 1 - cosine_similarity(self.forward(a_0, self.neg(a_0)), F)
        return (r_3 + r_4 + r_5 + r_6).sum()


class EmbedOR(EmbedBinaryOp):
    def logic_reg(self, a_0):
        T = self.logic.T()
        F = self.neg(T)
        r_7 = 1 - cosine_similarity(self.forward(a_0, T), T)
        r_8 = 1 - cosine_similarity(self.forward(a_0, F), a_0)
        r_9 = 1 - cosine_similarity(self.forward(a_0, a_0), a_0)
        r_10 = 1 - cosine_similarity(self.forward(a_0, self.neg(a_0)), T)
        return (r_7 + r_8 + r_9 + r_10).sum()

