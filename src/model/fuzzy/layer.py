import torch
from torch import nn
import fuzzy

class NOT(nn.Module):
    def forward(self, input):
        return fuzzy.fnot(input)

class FuzzyOp(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(FuzzyOp, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = torch.nn.Parameter(torch.rand(in_features, out_features), requires_grad=True)

    def harden(self):
        self.weights.requires_grad = False
        self.weights[:] = self.weights.data.clamp(min = 0, max = 1)
        self.weights.requires_grad = True

class OR(FuzzyOp):
    def forward(self, input: torch.Tensor):
        return 1 - (1 - 
            self.weights.reshape((self.weights.shape[1], self.weights.shape[0], 1))  * 
            input.transpose(1, 0).transpose(1, 0).reshape(1, input.shape[1], input.shape[0])
        ).prod(dim=1).transpose(1,0)

class AND(FuzzyOp):
    def forward(self, input: torch.Tensor):
        return (1 - 
            self.weights.transpose(1,0).reshape((self.weights.shape[1], self.weights.shape[0], 1)) 
            * (1 - input).transpose(1, 0).reshape(1, input.shape[1], input.shape[0])
        ).prod(dim=1).transpose(1,0)

class FuzzyDNF(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int = 10):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.conjunctions = FuzzyAND(2 * in_features, hidden_features)
        self.disjunctions = FuzzyOR(hidden_features, out_features)

        self.net = nn.Sequential(
            NOTs(in_features),
            self.conjunctions,
            self.disjunctions
        )

    def forward(self, input):
        return self.net(input)

    def harden(self):
        self.conjunctions.harden()
        self.disjunctions.harden()