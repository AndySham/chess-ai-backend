from src.model.fuzzy_logic import FuzzyLogic
import torch
from torch import nn, Tensor
from src.dnf import format_dnf


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


class FuzzyUnsignedConjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.fuzzy_params()
        return self._logic.conjoin(self._logic.implies(weights, input), dim=1)


class FuzzyUnsignedDisjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.fuzzy_params()
        return self._logic.disjoin(self._logic.bin_conjoin(weights, input), dim=1)


class FuzzyDNF(nn.Module):
    def __init__(self, shape: tuple[int, int, int], logic: FuzzyLogic):
        super().__init__()
        self._logic = logic
        in_f, hidden_f, out_f = shape
        self.conj = FuzzySignedConjunction(in_f, hidden_f, logic)
        self.disj = FuzzyUnsignedDisjunction(hidden_f, out_f, logic)

    def forward(self, input: Tensor) -> Tensor:
        return self.disj(self.conj(input))


"""
class FuzzyDNF(FuzzyModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 10,
        logic_system=None,
    ):
        super().__init__(logic_system)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.conj_weights = torch.nn.Parameter(
            torch.rand(in_features, hidden_features) - 0.5, requires_grad=True
        )
        self.conj_signs = torch.nn.Parameter(
            torch.rand(in_features, hidden_features) - 0.5, requires_grad=True
        )
        self.disj_weights = torch.nn.Parameter(
            torch.rand(hidden_features, out_features), requires_grad=True
        )

    def fuzzy_params(self):
        return (
            torch.sigmoid(self.conj_weights),
            torch.sigmoid(self.conj_signs),
            torch.sigmoid(self.disj_weights),
        )

    def forward(self, xs):
        conj_weights, conj_signs, disj_weights = self.fuzzy_params()
        return self.weighted_disjoins(
            self.signed_weighted_conjoins(xs, conj_weights, conj_signs), disj_weights,
        )

    def harden_params(self):
        conj_weights, conj_signs, disj_weights = self.fuzzy_params()
        return (conj_weights > 0.5, conj_signs > 0.5, disj_weights > 0.5)

    def params_to_str(self):
        conj_weights, conj_signs, disj_weights = self.harden_params()

        dnf_strs = []
        for dnf_idx in range(disj_weights.shape[1]):
            conj_weights_here = conj_weights[:, disj_weights[:, dnf_idx]]
            conj_signs_here = conj_signs[:, disj_weights[:, dnf_idx]]
            dnf_strs.append(
                "DNF Feature %s:\n%s"
                % (dnf_idx, format_dnf(conj_signs_here, conj_weights_here))
            )

        return "\n\n".join(dnf_strs)
"""
