import torch
from torch.utils.data import Dataset
from logic import reduce_or, reduce_and

class OpDataset(Dataset):
    def __init__(self, no_dims, no_samples, p_sample, p_set, f):
        self.no_dims = no_dims
        self.no_samples = no_samples
        self.p_sample = p_sample
        self.p_set = p_set
        self.op_set = torch.rand(no_dims) < p_set
        self.X = (torch.rand(no_samples, no_dims) < p_sample).float()
        self.y = f(self.X[:,self.op_set])

    def __len__(self):
        return self.no_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ORDataset(OpDataset):
    def __init__(self, no_dims, no_samples, p_sample, p_set):
        super(ORDataset, self).__init__(
            no_dims, no_samples, p_sample, p_set, 
            lambda xs: reduce_or(xs, dim=1)
        )
    
class ANDDataset(OpDataset):
    def __init__(self, no_dims, no_samples, p_sample, p_set):
        super(ANDDataset, self).__init__(
            no_dims, no_samples, p_sample, p_set, 
            lambda xs: reduce_and(xs, dim=1)
        )

def dnfs(X, conj_sets, disj_set):
    with_nots = torch.cat([X, 1-X], dim=1)
    stacked = torch.unsqueeze(with_nots, dim=0).repeat((conj_sets.shape[0], 1, 1))
    conj_filter = conj_sets.repeat(1,1,X.shape[0]).reshape(conj_sets.shape[0], X.shape[0], X.shape[1] * 2)
    conjs = reduce_and(torch.logical_or(torch.logical_and(stacked, conj_filter), torch.logical_not(conj_filter)), dim=2).transpose(1,0)
    return reduce_or(torch.logical_and(conjs, disj_set.reshape(1, conj_sets.shape[0]).repeat((X.shape[0], 1))), dim=1)

class DNFDataset(Dataset):
    def __init__(self, no_dims, no_samples, p_sample, no_conjs, p_conj_set, p_disj_set):
        self.X = (torch.rand(no_samples, no_dims) < p_sample).float()
        self.conj_sets = torch.rand(no_conjs, 2 * no_dims) < p_conj_set
        self.disj_set = torch.rand(no_conjs) < p_disj_set

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], dnfs(self.X[idx:idx+1,:], self.conj_sets, self.disj_set)[0]