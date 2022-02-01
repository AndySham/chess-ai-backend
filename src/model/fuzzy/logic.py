import torch
from torch import Tensor

# https://en.wikipedia.org/wiki/Construction_of_t-norms

def _align_shapes(xs: Tensor, ws: Tensor):

    B, N_1 = xs.shape()
    N_2, W = ws.shape()
    if N_1 != N_2:
        raise Exception("Error - Shapes must align.")

    xs = xs.reshape(1, N_1, W)
    ws = ws.reshape(B, N_1, 1)

    return xs, ws


def _bf_clamp(xs: Tensor) -> Tensor:
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


def min_conjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return min_tnorm((1 - ws) + xs * ws, dim=1)


def min_disjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return min_tconorm(xs * ws, dim=1)


# Lukasiewicz T-Norm : a, b -> max(a + b - 1, 0)


def luk_tnorm(xs: Tensor, dim: int) -> Tensor:
    return _bf_clamp(xs.sum(dim=dim) - xs.size(dim=dim) + 1)


def luk_tconorm(xs: Tensor, dim: int) -> Tensor:
    return _bf_clamp(xs.sum(dim=dim))


def luk_conjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return luk_tnorm((1 - ws) + xs * ws, dim=1)


def luk_disjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return luk_tconorm(xs * ws, dim=1)


# Drastic T-Norm : a, b -> b if a = 1, a if b = 1, 0 otherwise


def dra_tnorm(xs: Tensor, dim: int) -> Tensor:
    return torch.where(
        torch.count_nonzero(xs != 1.0, dim=dim) <= 1, 
        xs.prod(dim=dim), 
        torch.zeros(xs.shape[:dim] + xs.shape[dim+1:])
    )


def dra_tconorm(xs: Tensor, dim: int) -> Tensor:
    return torch.where(
        torch.count_nonzero(xs, dim=dim) <= 1, 
        xs.sum(dim=dim), 
        torch.ones(xs.shape[:dim] + xs.shape[dim+1:])
    )


def dra_conjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return dra_tnorm((1 - ws) + xs * ws, dim=1)


def dra_disjunction(xs: Tensor, ws: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return dra_tconorm(xs * ws, dim=1)


# Schweizer-Sklar T-Norms : a, b, p -> clamped ( x^p + y^p - 1 )^(1/p)


def ss_tnorm(xs: Tensor, p: Tensor, dim: int) -> Tensor:
    return torch.where(
        p == 0,
        prod_tnorm(xs, dim),
        _bf_clamp(((xs ** p).sum(dim=dim) - xs.size(dim=dim) + 1) ** (1 / p))
    )


def ss_tconorm(xs: Tensor, p: Tensor, dim: int) -> Tensor:
    return 1 - ss_tnorm(1 - xs, p, dim)


def ss_conjunction(xs: Tensor, ws: Tensor, p: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return ss_tnorm((1 - ws) + xs * ws, p, dim=1)


def ss_disjunction(xs: Tensor, ws: Tensor, p: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return ss_tconorm(xs * ws, p, dim=1)


# Hamacher T-Norms : a, b, p -> xy / (p + (1 - p)(x + y - xy)) (with limits)


def hmc_tnorm(xs: Tensor, p: Tensor, dim: int) -> Tensor:
    return torch.where(
        p == 0,
        ((xs ** -1).sum(dim=dim) - xs.size(dim=dim) + 1) ** -1,
        p * ((p / xs - (p - 1)).prod(dim=dim) + p - 1) ** -1
    )


def hmc_tconorm(xs: Tensor, p: Tensor, dim: int) -> Tensor:
    return 1 - hmc_tnorm(1 - xs, p, dim)


def hmc_conjunction(xs: Tensor, ws: Tensor, p: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return hmc_tnorm((1 - ws) + xs * ws, p, dim=1)


def hmc_disjunction(xs: Tensor, ws: Tensor, p: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return hmc_tconorm(xs * ws, p, dim=1)



# Weighted Non-Linear Logic : x, y, beta -> max(a + b - 2 + beta)
# Equivalent to Lukasiewicz logic for beta = 1
# https://arxiv.org/pdf/2006.13155.pdf



def wnl_tnorm(xs: Tensor, beta: Tensor, dim: int) -> Tensor:
    return _bf_clamp(xs.sum(dim=dim) - xs.size(dim=dim) + beta)


def wnl_tconorm(xs: Tensor, beta: Tensor, dim: int) -> Tensor:
    return _bf_clamp(xs.sum(dim=dim) + 1 - beta)


def wnl_conjunction(xs: Tensor, ws: Tensor, beta: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return wnl_tnorm((1 - ws) + xs * ws, beta=beta, dim=1)


def wnl_disjunction(xs: Tensor, ws: Tensor, beta: Tensor) -> Tensor:
    xs, ws = _align_shapes(xs, ws)
    return wnl_tconorm(xs * ws, beta=beta, dim=1)


# Fuzzy Negation


def fnot(a: Tensor) -> Tensor:
    return 1 - a
