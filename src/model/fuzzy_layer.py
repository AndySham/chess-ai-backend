from model.fuzzy_logic import FuzzyLogic
import torch
from torch import nn, Tensor
from dnf import format_dnf



class FuzzyParam(nn.Module):
    def __init__(self, shape: torch.Size):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.reinitialize()

    def initialize(self, shape: torch.Size):
        ws = torch.rand(shape)
        with torch.no_grad():
            self.param[:] = ws.log() - (1 - ws).log()

    def reinitialize(self):
        self.initialize(self.param.shape)

    def value(self) -> Tensor:
        return torch.sigmoid(self.param)

class FuzzySignedAxisOp(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, logic: FuzzyLogic,
    ):
        super().__init__()
        self._logic = logic
        self.weights = FuzzyParam((in_features, out_features))
        self.signs = FuzzyParam((in_features, out_features))

class FuzzySignedConjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
        signs = self.signs.value()
        return self._logic.conjoin(
            self._logic.implies(weights, self._logic.bin_xnor(input, signs)), dim=1,
        )


class FuzzySignedDisjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
        signs = self.signs.value()
        return self._logic.disjoin(
            self._logic.bin_conjoin(weights, self._logic.bin_xnor(input, signs)), dim=1,
        )


class FuzzyUnsignedAxisOp(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, logic: FuzzyLogic,
    ):
        super().__init__()
        self._logic = logic
        self.weights = FuzzyParam((in_features, out_features))

class FuzzyUnsignedConjunction(FuzzyUnsignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
        return self._logic.conjoin(self._logic.implies(weights, input), dim=1)


class FuzzyUnsignedDisjunction(FuzzyUnsignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
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
