import torch
from torch import Tensor
from math import pi, sqrt


def norm_pdf(xs, mu, sigma):
    min_mean = xs - mu
    return torch.exp(-min_mean * min_mean / (2 * sigma * sigma)) / sqrt(
        2 * pi * sigma * sigma
    )


def dim_size(xs: Tensor, dim: int = None) -> int:
    if dim == None:
        return xs.numel()
    else:
        return xs.size(dim=dim)


def match_shapes(*args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """
    For operations, we can choose to apply them element-wise for vectors, or repeating one value for scalars.
    This tiles Tensors to allow for both, given operations that only support the former.
    """
    shapes = [torch.tensor(t.shape) for t in args]
    max_len = max(len(s) for s in shapes)
    args = [arg[(None,) * (max_len - len(arg.shape))] for arg in args]

    shapes = [
        torch.cat((torch.ones(max_len - len(shape)), shape), dim=0) for shape in shapes
    ]
    shapes = torch.stack(shapes, dim=0).int()
    preferred_shape = shapes.max(dim=0).values
    shapes_fit = (
        torch.logical_or((shapes == preferred_shape), (shapes == 1)).prod().item() == 1
    )
    if not shapes_fit:
        raise Exception(
            "Where dimension sizes differ, one Tensor must have dimension size 1. %s"
            % [tuple(shape.tolist()) for shape in shapes]
        )

    tiling = torch.div(preferred_shape, shapes, rounding_mode="floor")
    return tuple(
        torch.tile(args[i], tuple(tiling[i].tolist())) for i in range(len(args))
    )


def batch_randperm(shape: torch.Size, fix_dims=()) -> Tensor:
    """
    Generates random permutations in batches.
    Specifying `fix_dims` allows the user to specify whether permutations should
    be equal along certain axes.
    """
    shape = torch.Tensor(list(shape)).long()
    fix_dims = torch.Tensor(list(fix_dims)).long()
    if (fix_dims == len(shape) - 1).any():
        raise Exception("Cannot fix final dimension.")
    perm_shape = shape.clone().detach()
    perm_shape[fix_dims] = 1
    tile_shape = torch.div(shape, perm_shape, rounding_mode="floor")
    return (
        torch.rand(torch.Size(perm_shape)).argsort(dim=-1).tile(torch.Size(tile_shape))
    )


def shuffle(xs: Tensor, dim: int, fix_dim=()) -> Tensor:
    """Permutes elements along a dimension."""
    final_dim = len(xs.shape) - 1
    fix_dim = torch.Tensor(list(fix_dim))
    is_dim = fix_dim == dim
    is_final = fix_dim == final_dim
    fix_dim[is_dim] = final_dim
    fix_dim[is_final] = dim

    xs_ = xs.transpose(dim, -1)
    perms = batch_randperm(xs_.shape, fix_dim)
    perm_size = xs_.shape[-1]

    perms = perms.view(-1, perm_size)
    offset = torch.arange(perms.shape[0]).unsqueeze(-1).tile(1, perm_size) * perm_size
    idxs = (perms + offset).view(-1)

    return (xs_.reshape(-1)[idxs]).view(xs_.shape).transpose(dim, -1)


def recursive_binop(xs: Tensor, binop, dim: int) -> Tensor:
    no_dims = len(xs.shape)
    slices = [slice(None, None) for _ in range(no_dims)]
    key_0 = tuple(slices + [0])
    key_1 = tuple(slices + [1])
    key_ltn1 = tuple(slices[:-1] + [slice(None, -1)])
    key_n1 = tuple(slices[:-1] + [-1])

    def rec(xs):
        size = xs.shape[-1]
        if size < 2:
            return xs.squeeze(-1)
        if size % 2 == 0:
            xs = xs.reshape(*xs.shape[:-1], size // 2, 2)
            return rec(
                binop(
                    xs[key_0].transpose(dim, -1), xs[key_1].transpose(dim, -1)
                ).transpose(dim, -1)
            )
        else:
            xs_ = xs[key_ltn1]
            xs_ = xs_.reshape(*xs.shape[:-1], size // 2, 2)
            return rec(
                torch.cat(
                    (
                        binop(
                            xs_[key_0].transpose(dim, -1), xs_[key_1].transpose(dim, -1)
                        ).transpose(dim, -1),
                        xs[key_n1].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )

    out = rec(xs.transpose(dim, -1))
    if dim < no_dims - 1:
        return out.transpose(dim, -1)
    else:
        return out
