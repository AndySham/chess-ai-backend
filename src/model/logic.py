import torch
from torch import Tensor, nn

from src.util import match_shapes, recursive_binop


class Logic(nn.Module):
    """A base class for different definitions of logics."""

    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        """
        Take the conjunction over all values in a Tensor. A single dimension may also be specified to take the conjunction over only one axis.
        """
        raise NotImplementedError

    def bin_conjoin(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Take the conjunction of two values. 
        This is referred to as the t-norm in fuzzy logic.
        """
        a, b = match_shapes(a, b)
        return self.conjoin(torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0), dim=0)

    def neg(self, xs: Tensor) -> Tensor:
        """Fuzzy negation. This tends to just be the operation x -> 1 - x."""
        raise NotImplementedError

    def disjoin(self, xs: Tensor, dim=None) -> Tensor:
        """
        Take the disjunction over all values in a Tensor. 
        A single dimension may also be specified to take the disjunction over only one axis.
        """
        return self.neg(self.conjoin(self.neg(xs), dim=dim))

    def bin_disjoin(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Take the disjunction of two values. 
        This is referred to as the t-conorm in fuzzy logic.
        """
        return self.neg(self.bin_conjoin(self.neg(a), self.neg(b)))

    def implies(self, a: Tensor, b: Tensor) -> Tensor:
        """
        The impliciation operator `=>`.
        """
        # This definition is not the residuum. The residuum should
        # be constructed as the adjoint of conjunction, but
        # the following is suitable for neurosymbolic learning
        # purposes (for now?)
        return self.neg(self.bin_conjoin(a, self.neg(b)))

    def xor(self, xs: Tensor, dim: int) -> Tensor:
        """
        Take the XOR over all values in a Tensor.
        A single dimension may also be specified to take the XOR over only one axis.
        """
        return recursive_binop(xs, self.bin_xor, dim)

    def bin_xor(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Take the XOR of two values.
        """
        return self.bin_conjoin(
            self.bin_disjoin(a, b), self.neg(self.bin_conjoin(a, b))
        )

    def xnor(self, xs: Tensor, dim: int) -> Tensor:
        """
        Take the XNOR over all values in a Tensor.
        A single dimension may also be specified to take the XNOR over only one axis.
        """
        return recursive_binop(xs, self.bin_xnor, dim)

    def bin_xnor(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Take the XNOR of two values.
        """
        return self.bin_disjoin(
            self.neg(self.bin_disjoin(a, b)), self.bin_conjoin(a, b)
        )
