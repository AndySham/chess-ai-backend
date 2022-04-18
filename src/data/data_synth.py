import torch
from torch.utils.data import Dataset
from logic import reduce_or, reduce_and
from util import match_shapes


class OpDataset(Dataset):
    def __init__(self, no_dims, no_samples, p_sample, p_set, f):
        self.no_dims = no_dims
        self.no_samples = no_samples
        self.p_sample = p_sample
        self.p_set = p_set
        self.op_set = torch.rand(no_dims) < p_set
        self.X = (torch.rand(no_samples, no_dims) < p_sample).float()
        self.y = f(self.X[:, self.op_set])

    def __len__(self):
        return self.no_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ORDataset(OpDataset):
    def __init__(self, no_dims, no_samples, p_sample, p_set):
        super(ORDataset, self).__init__(
            no_dims, no_samples, p_sample, p_set, lambda xs: reduce_or(xs, dim=1)
        )


class ANDDataset(OpDataset):
    def __init__(self, no_dims, no_samples, p_sample, p_set):
        super(ANDDataset, self).__init__(
            no_dims, no_samples, p_sample, p_set, lambda xs: reduce_and(xs, dim=1)
        )


def logical_xnor(a, b):
    return torch.logical_not(torch.logical_xor(a, b))


def logical_impl(a, b):
    return torch.logical_or(torch.logical_not(a), b)


def dnfs(X, conj_weights, conj_signs, disj_weights):
    B, N = X.shape
    N_2, H = conj_weights.shape
    N_3, H_2 = conj_signs.shape
    H_3, O = disj_weights.shape

    if N != N_2 or N_2 != N_3 or H != H_2 or H_2 != H_3:
        raise Exception("Misaligned shapes for DNF.")

    X, conj_weights = match_shapes(X.view(B, N, 1), conj_weights.view(1, N, H))
    _, conj_signs = match_shapes(X, conj_signs.view(1, N, H))

    conjs = reduce_and(logical_impl(conj_weights, logical_xnor(X, conj_signs)), dim=1)

    conjs, disj_weights = match_shapes(conjs.view(B, H, 1), disj_weights.view(1, H, O))

    return reduce_or(torch.logical_and(conjs, disj_weights), dim=1)


class DNFDataset(Dataset):
    def __init__(self, no_dims, no_samples, p_sample, no_conjs, p_conj_set):
        self.X = (torch.rand(no_samples, no_dims) < p_sample).float()
        self.conj_weights = torch.rand(no_dims, no_conjs) < p_conj_set
        self.conj_signs = torch.rand(no_dims, no_conjs) < 0.5
        self.disj_weights = torch.ones(no_conjs, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            torch.squeeze(
                dnfs(
                    self.X[idx : idx + 1, :],
                    self.conj_weights,
                    self.conj_signs,
                    self.disj_weights,
                )[0]
            ),
        )

