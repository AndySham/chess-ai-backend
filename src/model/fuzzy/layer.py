from model.fuzzy import logic
import torch
from torch import nn


class FuzzyOp(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(FuzzyOp, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features), requires_grad=True
        )

        self.beta = torch.nn.Parameter(torch.rand(out_features), requires_grad=True)

    def harden(self):
        self.weights.requires_grad = False
        self.weights[:] = self.weights.data.clamp(min=0, max=1)
        self.weights.requires_grad = True


class OR(FuzzyOp):
    def forward(self, xs: torch.Tensor):
        return logic.prod_disjunction(xs, self.weights)


class AND(FuzzyOp):
    def forward(self, xs: torch.Tensor):
        return logic.prod_conjunction(xs, self.weights)


class FuzzyDNF(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int = 10):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.conjunctions = AND(2 * in_features, hidden_features)
        self.disjunctions = OR(hidden_features, out_features)

    def forward(self, xs):
        return self.disjunctions(self.conjunctions(torch.cat(xs, logic.fnot(xs))))

    def harden(self):
        self.conjunctions.harden()
        self.disjunctions.harden()
