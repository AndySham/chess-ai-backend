from src.model.fuzzy import logic
import torch
from torch import nn


class FuzzyNOT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: torch.Tensor):
        return logic.fnot(xs)


class FuzzyOperator(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, logic_type: str = logic.lukasiewicz
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features), requires_grad=True
        )

        self.logic_type = logic_type
        if logic_type == logic.schweizer_sklar:
            self.ss_param = torch.nn.Parameter(torch.rand(out_features))
        if logic_type == logic.schweizer_sklar:
            self.hmc_param = torch.nn.Parameter(torch.rand(out_features))
        if logic_type == logic.wnl:
            self.wnl_param = torch.nn.Parameter(torch.rand(out_features))

    def get_weights(self):
        """Weights must be in range [0,1] to be valid fuzzy booleans."""
        return torch.sigmoid(self.weights)


class FuzzyOR(FuzzyOperator):
    def forward(self, xs: torch.Tensor):
        weights = self.get_weights()
        if self.logic_type == logic.product:
            return logic.prod_disjunction(xs, weights)
        elif self.logic_type == logic.minimum:
            return logic.min_disjunction(xs, weights)
        elif self.logic_type == logic.lukasiewicz:
            return logic.luk_disjunction(xs, weights)
        elif self.logic_type == logic.drastic:
            return logic.dra_disjunction(xs, weights)
        elif self.logic_type == logic.schweizer_sklar:
            param = self.ss_param
            return logic.ss_disjunction(xs, weights, param)
        elif self.logic_type == logic.hamacher:
            param = torch.exp(self.hmc_param)  # must be >= 0
            return logic.hmc_disjunction(xs, weights, param)
        elif self.logic_type == logic.wnl:
            param = torch.exp(self.wnl_param)  # must be >= 0
            return logic.wnl_disjunction(xs, weights, param)
        else:
            raise Exception("No valid logic type chosen.")


class FuzzyAND(FuzzyOperator):
    def forward(self, xs: torch.Tensor):
        weights = self.get_weights()
        if self.logic_type == logic.product:
            return logic.prod_conjunction(xs, weights)
        elif self.logic_type == logic.minimum:
            return logic.min_conjunction(xs, weights)
        elif self.logic_type == logic.lukasiewicz:
            return logic.luk_conjunction(xs, weights)
        elif self.logic_type == logic.drastic:
            return logic.dra_conjunction(xs, weights)
        elif self.logic_type == logic.schweizer_sklar:
            param = self.ss_param
            return logic.ss_conjunction(xs, weights, param)
        elif self.logic_type == logic.hamacher:
            param = torch.exp(self.hmc_param)  # must be >= 0
            return logic.hmc_conjunction(xs, weights, param)
        elif self.logic_type == logic.wnl:
            param = torch.exp(self.wnl_param)  # must be >= 0
            return logic.wnl_conjunction(xs, weights, param)
        else:
            raise Exception("No valid logic type chosen.")


class FuzzyDNF(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 10,
        logic_type: str = logic.lukasiewicz,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.conjunctions = FuzzyAND(
            2 * in_features, hidden_features, logic_type=logic_type
        )
        self.disjunctions = FuzzyOR(
            hidden_features, out_features, logic_type=logic_type
        )

    def forward(self, xs):
        with_negs = torch.cat([xs, logic.fnot(xs)], dim=1)
        return self.disjunctions(self.conjunctions(with_negs))

