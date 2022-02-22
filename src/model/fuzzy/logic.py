import torch
from torch import Tensor


# Implementations of various Fuzzy Logic systems. Much of the derivations of sequence operations
# are derived from the methods given in the following article.
#
# https://en.wikipedia.org/wiki/Construction_of_t-norms


class FuzzyLogic:
    """A base class for different instances of fuzzy logics."""

    # Default is drastic t-norm logic
    def conjoin(self, xs: Tensor, dim: int = None):
        """
        Take the conjunction over all values in a Tensor. A single dimension may also be specified to take the conjunction over only one axis.
        """
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

    def bin_conjoin(self, a: Tensor, b: Tensor):
        """
        Take the conjunction of two values. 
        This is referred to as the t-norm in fuzzy logic.
        """
        return self.conjoin(torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0), dim=0)

    def neg(self, xs: Tensor):
        """Fuzzy negation. This tends to just be the operation x -> 1 - x."""
        return 1 - xs

    def disjoin(self, xs: Tensor, dim=None):
        """
        Take the disjunction over all values in a Tensor. 
        A single dimension may also be specified to take the disjunction over only one axis.
        """
        return self.neg(self.conjoin(self.neg(xs), dim=dim))

    def bin_disjoin(self, a: Tensor, b: Tensor):
        """
        Take the disjunction of two values. 
        This is referred to as the t-conorm in fuzzy logic.
        """
        return self.neg(self.bin_conjoin(self.neg(a), self.neg(b)))

    def residuum(self, a: Tensor, b: Tensor):
        """
        The residuum in fuzzy logic is analogous to the impliciation operator `=>` in classical logic.
        """
        return self.neg(self.bin_conjoin(a, self.neg(b)))

    def xor(self, a: Tensor, b: Tensor):
        """
        Take the XOR over all values in a Tensor.
        A single dimension may also be specified to take the XOR over only one axis.
        """
        raise Exception("Axis-wise XOR not yet implemented.")

    def bin_xor(self, a: Tensor, b: Tensor):
        """
        Take the XOR of two values.
        """
        return self.bin_conjoin(
            self.bin_disjoin(a, b), self.neg(self.bin_conjoin(a, b))
        )

    def bin_xnor(self, a: Tensor, b: Tensor):
        """
        Take the XNOR of two values.
        """
        return self.bin_disjoin(
            self.neg(self.bin_disjoin(a, b)), self.bin_conjoin(a, b)
        )

    @staticmethod
    def _align_weight_shapes(xs: Tensor, ws: Tensor):

        B, N_1 = xs.shape
        N_2, W = ws.shape
        if N_1 != N_2:
            raise Exception(
                "Error - Weight shapes must align with input. xs: [%s, %s] ws: [%s, %s]"
                % (B, N_1, N_2, W)
            )

        xs = xs.reshape(B, N_1, 1)
        ws = ws.reshape(1, N_1, W)

        return xs, ws

    def weighted_conjoins(self, xs: Tensor, ws: Tensor):
        """
        For learning a single weighted conjunction, without signs.
        - `xs` has shape `(B, N)`, for `B` batch size, `N` number of input features
        - `ws` has shape `(N, W)`, for `N`, number of input features, `W` number of output features.
        - Returns value of shape `(B, W)`
        """
        xs, ws = self._align_weight_shapes(xs, ws)
        return self.conjoin(self.residuum(ws, xs), dim=1)

    def weighted_disjoins(self, xs: Tensor, ws: Tensor):
        """
        For learning a single weighted disjunction, without signs.
        - `xs` has shape `(B, N)`, for `B` batch size, `N` number of input features
        - `ws` has shape `(N, W)`, for `N`, number of input features, `W` number of output features.
        - Returns value of shape `(B, W)`
        """
        xs, ws = self._align_weight_shapes(xs, ws)
        return self.disjoin(self.bin_conjoin(ws, xs), dim=1)

    @staticmethod
    def _align_signed_weight_shapes(xs: Tensor, ws: Tensor, ss: Tensor):

        B, N_1 = xs.shape
        N_2, W_1 = ws.shape
        N_3, W_2 = ss.shape
        if N_1 != N_2 or N_2 != N_3 or W_1 != W_2:
            raise Exception(
                "Error - Signed weight shapes must align with input. xs: [%s, %s] ws: [%s, %s] ss: [%s %s]"
                % (B, N_1, N_2, W_1, N_3, W_2)
            )

        xs = xs.reshape(B, N_1, 1)
        ws = ws.reshape(1, N_1, W_1)
        ss = ws.reshape(1, N_1, W_1)

        return xs, ws, ss

    def signed_weighted_conjoins(self, xs: Tensor, ws: Tensor, ss: Tensor):
        """
        For learning a single weighted conjunction, with signs.
        - `xs` has shape `(B, N)`, for `B` batch size, `N` number of input features
        - `ws` has shape `(N, W)`, for `N`, number of input features, `W` number of output features.
        - `ss` has shape `(N, W)`, for `N`, number of input features, `W` number of output features.
        - Returns value of shape `(B, W)`
        """
        xs, ws, ss = self._align_signed_weight_shapes(xs, ws, ss)
        return self.weighted_conjoins(self.bin_xnor(xs, ss), ws)

    def signed_weighted_disjoins(self, xs: Tensor, ws: Tensor, ss: Tensor):
        """
        For learning a single weighted disjunction, with signs.
        - `xs` has shape `(B, N)`, for `B` batch size, `N` number of input features
        - `ws` has shape `(N, W)`, for `N`, number of input features, `W` number of output features.
        - `ss` has shape `(N, W)`, for `N`, number of input features, `W` number of output features.
        - Returns value of shape `(B, W)`
        """
        xs, ws, ss = self._align_signed_weight_shapes(xs, ws, ss)
        return self.weighted_disjoins(self.bin_xnor(xs, ss), ws)

    @staticmethod
    def dim_size(xs: Tensor, dim: int = None) -> int:
        if dim == None:
            return xs.numel()
        else:
            return xs.size(dim=dim)


# Implementations ----------------------------------------------------------------------


class ProductLogic(FuzzyLogic):
    """`a ⊗ b = ab`"""

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return xs.prod(dim=dim)


class MinimumLogic(FuzzyLogic):
    """`a ⊗ b = min{a, b}`"""

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return xs.min(dim=dim)

    def disjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return xs.max(dim=dim)


class LukasiewiczLogic(FuzzyLogic):
    """`a ⊗ b = max{a + b - 1, 0}"""

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
    See https://arxiv.org/pdf/2006.13155.pdf for more information.
    """

    def __init__(self, beta: Tensor):
        self.beta = beta

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return (xs.sum(dim=dim) - self.dim_size(xs, dim=dim) + self.beta).clamp(0, 1)

    def disjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return (xs.sum(dim=dim) + 1 - self.beta).clamp(0, 1)

