import torch
from torch import nn, Tensor

from model.embed_logic import EmbedLogic


class EmbedLayer(nn.Module):
    def __init__(self, logic: EmbedLogic):
        super().__init__()
        self._logic = [logic]

    @property
    def logic(self):
        return self._logic[0]


class EmbedSignedAxisOp(EmbedLayer):
    def __init__(self, in_features: int, out_features: int, logic: EmbedLogic):
        super().__init__(logic)
        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features, logic.embed_dims), requires_grad=True
        )
        self.signs = torch.nn.Parameter(
            torch.rand(in_features, out_features, logic.embed_dims), requires_grad=True
        )


class EmbedSignedConjunction(EmbedSignedAxisOp):
    def forward(self, input: Tensor) -> Tensor:
        input = input.unsqueeze(-2)
        return self.logic.conjoin(
            self.logic.implies(self.weights, self.logic.bin_xnor(input, self.signs)),
            dim=1,
        )


class EmbedSignedDisjunction(EmbedSignedAxisOp):
    def forward(self, input: Tensor) -> Tensor:
        input = input.unsqueeze(-2)
        return self.logic.disjoin(
            self.logic.bin_conjoin(
                self.weights, self.logic.bin_xnor(input, self.signs)
            ),
            dim=1,
        )


class EmbedUnsignedAxisOp(EmbedLayer):
    def __init__(self, in_features: int, out_features: int, logic: EmbedLogic):
        super().__init__(logic)
        self.weights = torch.nn.Parameter(
            torch.rand(in_features, out_features, logic.embed_dims), requires_grad=True
        )


class EmbedUnsignedConjunction(EmbedUnsignedAxisOp):
    def forward(self, input: Tensor) -> Tensor:
        input = input.unsqueeze(-2)
        return self.logic.conjoin(self.logic.implies(self.weights, input), dim=1)


class EmbedUnsignedDisjunction(EmbedUnsignedAxisOp):
    def forward(self, input: Tensor) -> Tensor:
        input = input.unsqueeze(-2)
        return self.logic.disjoin(self.logic.bin_conjoin(self.weights, input), dim=1)


class EmbedDNF(nn.Module):
    def __init__(self, shape: tuple[int, int, int], logic: EmbedLogic):
        super().__init__()
        in_f, hidden_f, out_f = shape
        self.layers = nn.Sequential(
            EmbedSignedConjunction(in_f, hidden_f, logic),
            EmbedUnsignedDisjunction(hidden_f, out_f, logic),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
