import torch
from torch import Tensor


def _align_shapes(xs: Tensor, ws: Tensor):

    B, N_1 = xs.shape()
    N_2, W = ws.shape()
    if N_1 != N_2:
        raise Exception("Error - Shapes must align.")

    xs = xs.reshape(1, N_1, W)
    ws = ws.reshape(B, N_1, 1)

    return xs, ws


def bf_clamp(xs: Tensor) -> Tensor:
    return torch.clamp(xs, min=0.0, max=1.0)


# Product T-Norm : a, b -> ab


def prod_tnorm(xs: Tensor, dim: int) -> Tensor:
    return xs.prod(dim=dim)


def prod_tconorm(xs: Tensor, dim: int) -> Tensor:
    return 1 - (1 - xs).prod(dim=dim)


def prod_conjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return prod_tnorm((1 - ws) + xs * ws, dim=1)


def prod_disjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return prod_tconorm(xs * ws, dim=1)


# Minimum T-Norm : a, b -> min{a, b}


def min_tnorm(xs: Tensor, dim: int) -> Tensor:
    return xs.min(dim=dim)


def min_tconorm(xs: Tensor, dim: int) -> Tensor:
    return xs.max(dim=dim)


def min_conjunction(xs, ws) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return min_tnorm((1 - ws) + xs * ws, dim=1)


def min_disjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return min_tconorm(xs * ws, dim=1)


# Lukasiewicz T-Norm : a, b -> max(a + b - 1, 0)


def luk_tnorm(xs: Tensor, dim: int) -> Tensor:
    return bf_clamp(xs.sum(dim=dim) - xs.size(dim=dim) + 1)


def luk_tconorm(xs: Tensor, dim: int) -> Tensor:
    return bf_clamp(xs.sum(dim=dim))


def luk_conjunction(xs, ws) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return luk_tnorm((1 - ws) + xs * ws, dim=1)


def luk_disjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return luk_tconorm(xs * ws, dim=1)


# Weighted Non-Linear Logic : x, y, beta -> max(a + b - 2 + beta)
# Equivalent to Lukasiewicz logic for beta = 1


def wnl_tnorm(xs: Tensor, beta: float, dim: int) -> Tensor:
    return bf_clamp(xs.sum(dim=dim) - xs.size(dim=dim) + beta)


def wnl_tconorm(xs: Tensor, beta: float, dim: int) -> Tensor:
    return bf_clamp(xs.sum(dim=dim) + 1 - beta)


def wnl_conjunction(xs, ws) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return wnl_tnorm((1 - ws) + xs * ws, dim=1)


def wnl_disjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return wnl_tconorm(xs * ws, dim=1)


# Fuzzy Negation


def fnot(a: Tensor) -> Tensor:
    return 1 - a
