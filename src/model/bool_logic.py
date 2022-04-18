import torch
from torch import Tensor

from model.logic import Logic
from util import dim_size

class BoolLogic(Logic):
    def conjoin(self, xs: Tensor, dim: int = None) -> Tensor:
        return xs.sum(dim) == dim_size(xs, dim)

    def neg(self, xs: Tensor) -> Tensor:
        return xs == False

    def encode(self, xs: Tensor) -> Tensor:
        return xs

    def decode(self, xs: Tensor) -> Tensor:
        return xs.float()