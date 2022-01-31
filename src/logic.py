import torch

def reduce_or(xs: torch.Tensor, dim: int):
    return (xs.float().sum(dim=dim) >= 1).float()

def reduce_and(xs: torch.Tensor, dim: int):
    return 1 - reduce_or(1 - xs.float(), dim)