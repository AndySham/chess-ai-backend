import torch
from logic import reduce_or, reduce_and


class OpIdealModel:
    def __init__(self, dim):
        self.op_set = torch.ones(dim, dtype=bool)

    def update(self, X, y):
        return self.op_set

    def step(self, X, y):
        updated = self.update(X, y)
        self.op_set = torch.logical_and(self.op_set.to(updated.device), updated)


class ORIdealModel(OpIdealModel):
    def __init__(self, dim):
        super().__init__(dim)

    def update(self, X, y):
        return torch.logical_not(
            reduce_or(
                torch.logical_and(X, torch.logical_not(y).reshape(len(X), 1)), dim=0
            )
        )


class ANDIdealModel(OpIdealModel):
    def __init__(self, dim):
        super().__init__(dim)

    def update(self, X, y):
        return torch.logical_not(
            reduce_or(
                torch.logical_and(torch.logical_not(X), y.reshape(len(X), 1)), dim=0
            )
        )

