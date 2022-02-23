import torch
from math import pi, sqrt


def norm_pdf(xs, mu, sigma):
    min_mean = xs - mu
    return torch.exp(-min_mean * min_mean / (2 * sigma * sigma)) / sqrt(
        2 * pi * sigma * sigma
    )


def match_shapes(a, b):
    """
    For binary operations, we can choose to apply them element-wise for vectors, or repeating one value for scalars.
    This tiles Tensors to allow for both, given operations that only support the former.
    """
    if len(a.shape) != len(b.shape):
        raise Exception(
            "Must have same number of dimensions. (%s, %s)"
            % (len(a.shape), len(b.shape))
        )
    for i in range(len(a.shape)):
        if a.shape[i] != b.shape[i]:
            if a.shape[i] != 1 and b.shape[i] != 1:
                raise Exception(
                    "Where dimension sizes differ, one Tensor must have dimension size 1. (%s, %s)"
                    % (a.shape, b.shape)
                )

    a_shape = torch.tensor(a.shape)
    b_shape = torch.tensor(b.shape)
    preferred_shape = torch.max(a_shape, b_shape)
    a_tiling = torch.ones_like(preferred_shape)
    a_tiling[a_shape != preferred_shape] = preferred_shape[a_shape != preferred_shape]
    b_tiling = torch.ones_like(preferred_shape)
    b_tiling[b_shape != preferred_shape] = preferred_shape[b_shape != preferred_shape]

    return torch.tile(a, tuple(a_tiling)), torch.tile(b, tuple(b_tiling))
