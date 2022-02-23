from src.model.fuzzy import logic
import torch
from torch import nn


class FuzzyModule(nn.Module):
    def __init__(
        self, logic_system=None,
    ):
        super().__init__()

        self.logic_system = (
            logic.LukasiewiczLogic if logic_system == None else logic_system
        )
        self.logic: logic.FuzzyLogic = None

        if self.logic_system == logic.SchweizerSklarLogic:
            self.ss_param = torch.nn.Parameter(torch.rand(1), requires_grad=True)
            self.logic = self.logic_system(self.ss_param)
        elif self.logic_system == logic.HamacherLogic:
            self.hmc_param = torch.nn.Parameter(torch.rand(1), requires_grad=True)
            self.logic = self.logic_system(torch.exp(self.hmc_param))
        elif self.logic_system == logic.WNLLogic:
            self.wnl_param = torch.nn.Parameter(torch.rand(1), requires_grad=True)
            self.logic = self.logic_system(torch.exp(self.wnl_param))
        else:
            self.logic = self.logic_system()


class FuzzyDisjunction(FuzzyModule):
    def __init__(
        self, in_features: int, out_features: int, logic_system=None,
    ):
        super().__init__(logic_system)

        self.in_features = in_features
        self.out_features = out_features

        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features), requires_grad=True
        )

    def forward(self, xs):
        weights = torch.sigmoid(self.weights)
        return self.logic.weighted_disjoins(xs, weights)


class FuzzyConjunction(FuzzyModule):
    def __init__(
        self, in_features: int, out_features: int, logic_system=None,
    ):
        super().__init__(logic_system)

        self.in_features = in_features
        self.out_features = out_features

        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features), requires_grad=True
        )

    def forward(self, xs):
        weights = torch.sigmoid(self.weights)
        return self.logic.weighted_conjoins(xs, weights)


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
            torch.rand(in_features, hidden_features), requires_grad=True
        )
        self.conj_signs = torch.nn.Parameter(
            torch.rand(in_features, hidden_features), requires_grad=True
        )
        self.disj_weights = torch.nn.Parameter(
            torch.rand(hidden_features, out_features), requires_grad=True
        )

    def forward(self, xs):
        conj_weights = torch.sigmoid(self.conj_weights)
        conj_signs = torch.sigmoid(self.conj_signs)
        disj_weights = torch.sigmoid(self.disj_weights)
        return self.logic.weighted_disjoins(
            self.logic.signed_weighted_conjoins(xs, conj_weights, conj_signs),
            disj_weights,
        )

