import torch
from torch import Tensor
from src.model.logic import Logic
from src.util import match_shapes, recursive_binop


# Implementations of various Fuzzy Logic systems. Much of the derivations of sequence operations
# are found using the methods given in the following article.
#
# https://en.wikipedia.org/wiki/Construction_of_t-norms


class FuzzyLogic(Logic):
    """A base class for different instances of fuzzy logics."""


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
        return (xs.sum(dim=dim) - self.dim_size(xs, dim=dim) + 1).clamp(0, 1)

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


class SchweizerSklarLogic(FuzzyLogic):
    """`a ⊗ b = (a**p + b**p - 1) ** (1/p)`"""

    def __init__(self, p: Tensor):
        super().__init__()
        self.p = p

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return torch.where(
            self.p == 0,
            xs.prod(dim=dim),
            (
                ((xs ** self.p).sum(dim=dim) - self.dim_size(xs, dim=dim) + 1)
                ** (1 / self.p)
            ).clamp(0, 1),
        )


class HamacherLogic(FuzzyLogic):
    """`a ⊗ b = ab / (p + (1 - p)(a + b - ab))`"""

    def __init__(self, p: Tensor):
        super().__init__()
        self.p = p

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return torch.where(
            self.p == 0,
            ((xs ** -1).sum(dim=dim) - self.dim_size(xs, dim=dim) + 1) ** -1,
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
        return (xs.sum(dim=dim) - self.dim_size(xs, dim=dim) + self.beta).clamp(0, 1)

    def disjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return (xs.sum(dim=dim) + 1 - self.beta).clamp(0, 1)

