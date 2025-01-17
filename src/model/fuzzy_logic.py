import torch
from torch import Tensor
from model.logic import Logic
from util import dim_size, match_shapes, recursive_binop


# Implementations of various Fuzzy Logic systems. Much of the derivations of sequence operations
# are found using the methods given in the following article.
#
# https://en.wikipedia.org/wiki/Construction_of_t-norms


class FuzzyLogic(Logic):
    """A base class for different instances of fuzzy logics."""

    def neg(self, xs: Tensor) -> Tensor:
        return 1 - xs

    def encode(self, xs: Tensor) -> Tensor:
        return xs.float()

    def decode(self, xs: Tensor) -> Tensor:
        return xs


# Implementations ----------------------------------------------------------------------


class ProductLogic(FuzzyLogic):
    """`a ⊗ b = ab`"""

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return xs.prod(dim=dim)


class MinimumLogic(FuzzyLogic):
    """`a ⊗ b = min{a, b}`"""

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        if dim == None:
            return xs.min()
        else:
            return xs.min(dim=dim).values

    def disjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        if dim == None:
            return xs.max()
        else:
            return xs.max(dim=dim).values


class LukasiewiczLogic(FuzzyLogic):
    """`a ⊗ b = max{a + b - 1, 0}`"""

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return (xs.sum(dim=dim) - dim_size(xs, dim=dim) + 1).clamp(0, 1)

    def disjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return (xs.sum(dim=dim)).clamp(0, 1)



class DrasticLogic(FuzzyLogic):
    """
    ```
    a ⊗ b = 
    a if b == 1, 
    b if a == 1, 
    0 otherwise
    ```
    """

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        if dim == None:
            return torch.where(
                torch.count_nonzero(xs != 1.0) <= 1, xs.prod(), torch.zeros(1)
            )
        else:
            return torch.where(
                torch.count_nonzero(xs != 1.0, dim=dim) <= 1,
                xs.prod(dim=dim),
                torch.zeros(xs.shape[:dim] + xs.shape[dim + 1 :]),
            )

    def disjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        if dim == None:
            return torch.where(torch.count_nonzero(xs) <= 1, xs.sum(), torch.ones(1))
        else:
            return torch.where(
                torch.count_nonzero(xs, dim=dim) <= 1,
                xs.sum(dim=dim),
                torch.ones(xs.shape[:dim] + xs.shape[dim + 1 :]),
            )

class NegSSAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xs, p, dim):
        min_x = xs.min(dim=dim).values
        offset = 1 - dim_size(xs, dim=dim)
        output = torch.where(
            min_x == 0,
            torch.tensor(0.0, device=min_x.device),
            min_x * (
                ((xs / min_x.unsqueeze(dim)) ** p).sum(dim=dim) 
                + offset * (min_x ** -p)
            ) ** (1 / p)
        )
        ctx.dim = dim
        ctx.save_for_backward(xs, output, p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        xs, output, p = ctx.saved_tensors
        return grad_output.unsqueeze(dim=ctx.dim) * torch.where(
            xs == 0.0,
            torch.tensor(1.0, device=grad_output.device),
            (xs / output.unsqueeze(dim=ctx.dim)) ** (p - 1)
        ), None, None

class SchweizerSklarLogic(FuzzyLogic):
    """`a ⊗ b = (a**p + b**p - 1) ** (1/p)`"""

    def __init__(self, p: Tensor):
        super().__init__()
        self.p = p

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        if self.p == 0:
            return xs.prod(dim=dim)
        elif self.p > 0:
            return (
                (xs ** self.p).sum(dim=dim) - dim_size(xs, dim=dim) + 1
            ).clamp(0, 1) ** (1 / self.p)
        else:
            return NegSSAutograd.apply(xs, self.p, dim)


class HamacherLogic(FuzzyLogic):
    """`a ⊗ b = ab / (p + (1 - p)(a + b - ab))`"""

    def __init__(self, p: Tensor):
        super().__init__()
        self.p = p

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return torch.where(
            self.p == 0,
            ((xs ** -1).sum(dim=dim) - dim_size(xs, dim=dim) + 1) ** -1,
            self.p
            * (
                (self.p - (self.p - 1) * xs).prod(dim=dim) / xs.prod(dim=dim)
                + self.p
                - 1
            )
            ** -1,
        )


class WNLLogic(FuzzyLogic):
    """
    `a ⊗ b = max{a + b - 2 + beta, 0}`
    Equivalent to Lukasiewicz Logic for `beta == 1`.
    Does not necessarily satisfy the t-norm axioms.
    See https://arxiv.org/pdf/2006.13155.pdf for more information.
    """

    def __init__(self, beta: Tensor):
        super().__init__()
        self.beta = beta

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return (xs.sum(dim=dim) - dim_size(xs, dim=dim) + self.beta).clamp(0, 1)

    def disjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return (xs.sum(dim=dim) + 1 - self.beta).clamp(0, 1)


