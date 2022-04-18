from model.fuzzy_logic import FuzzyLogic
import torch
from torch import nn, Tensor
from dnf import format_dnf


class FuzzySignedAxisOp(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, logic: FuzzyLogic,
    ):
        super().__init__()
        self._logic = logic
        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features) - 0.5, requires_grad=True
        )
        self.signs = torch.nn.Parameter(
            torch.rand(in_features, out_features) - 0.5, requires_grad=True
        )

    def fuzzy_params(self):
        return (
            torch.sigmoid(self.weights),
            torch.sigmoid(self.signs),
        )


class FuzzySignedConjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights, signs = self.fuzzy_params()
        return self._logic.conjoin(
            self._logic.implies(weights, self._logic.bin_xnor(input, signs)), dim=1,
        )


class FuzzySignedDisjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights, signs = self.fuzzy_params()
        return self._logic.disjoin(
            self._logic.bin_conjoin(weights, self._logic.bin_xnor(input, signs)), dim=1,
        )


class FuzzyUnsignedAxisOp(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, logic: FuzzyLogic,
    ):
        super().__init__()
        self._logic = logic
        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features) - 0.5, requires_grad=True
        )

    def fuzzy_params(self):
        return torch.sigmoid(self.weights)


class FuzzyUnsignedConjunction(FuzzyUnsignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.fuzzy_params()
        return self._logic.conjoin(self._logic.implies(weights, input), dim=1)


class FuzzyUnsignedDisjunction(FuzzyUnsignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.fuzzy_params()
        return self._logic.disjoin(self._logic.bin_conjoin(weights, input), dim=1)


class FuzzyDNF(nn.Module):
    def __init__(self, shape: tuple[int, int, int], logic: FuzzyLogic):
        super().__init__()
        in_f, hidden_f, out_f = shape
        self.layer = nn.Sequential(
            FuzzySignedConjunction(in_f, hidden_f, logic),
            FuzzyUnsignedDisjunction(hidden_f, out_f, logic),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.layer(input)

