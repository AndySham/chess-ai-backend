from model.fuzzy_logic import FuzzyLogic
import torch
from torch import nn, Tensor
from util import shuffle


class FuzzyParam(nn.Module):
    def __init__(self, shape: torch.Size):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.reinitialize()

    def initialize(self, shape: torch.Size):
        ws = torch.rand(shape)
        with torch.no_grad():
            self.param[:] = ws.log() - (1 - ws).log()
        # with torch.no_grad():
        #    self.param[:] = torch.rand(shape)

    def reinitialize(self):
        self.initialize(self.param.shape)

    def value(self) -> Tensor:
        # with torch.no_grad():
        #    self.param[:] = self.param.clamp(0, 1)
        # return self.param
        return torch.sigmoid(self.param)


def fuzzy_dropin(xs, updates):
    ys = 0.5 * torch.ones_like(xs)
    ys[updates == False] = xs[updates == False]
    return ys


def fuzzy_dropout(xs, updates):
    ys = (xs > 0.5).float()
    ys[updates == False] = xs[updates == False]
    return ys


def fuzzy_dropup(xs, updates):
    ys = torch.ones_like(xs)
    ys[updates == False] = xs[updates == False]
    return ys


def fuzzy_dropdown(xs, updates):
    ys = torch.zeros_like(xs)
    ys[updates == False] = xs[updates == False]
    return ys


def keepidx(shape, n, dim=-1):
    idxs = torch.zeros(shape)
    idxs = idxs.transpose(dim, -1)
    idxs[..., :n] = 1.0
    idxs = idxs.transpose(dim, -1)
    return shuffle(idxs.bool(), dim=dim)


def take_rand_n(xs, n, dim=-1):
    xs = xs.transpose(dim, -1)
    idxs = keepidx(xs.shape, n=n, dim=-1)
    vals = xs[idxs]
    vals = vals.reshape(*xs.shape[:-1], -1)
    return vals.transpose(dim, -1)


class FuzzyProbDrop(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.dist = torch.distributions.bernoulli.Bernoulli(p)


class FuzzyProbDropin(FuzzyProbDrop):
    def forward(self, input):
        updates = self.dist.sample(input.shape).bool()
        return fuzzy_dropin(input, updates)


class FuzzyProbDropout(FuzzyProbDrop):
    def forward(self, input):
        updates = self.dist.sample(input.shape).bool()
        return fuzzy_dropout(input, updates)


class FuzzyProbDropup(FuzzyProbDrop):
    def forward(self, input):
        updates = self.dist.sample(input.shape).bool()
        return fuzzy_dropup(input, updates)


class FuzzyProbDropdown(FuzzyProbDrop):
    def forward(self, input):
        updates = self.dist.sample(input.shape).bool()
        return fuzzy_dropdown(input, updates)


class FuzzyNumKeep(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n


class FuzzyNumKeepin(FuzzyNumKeep):
    def forward(self, input):
        updates = keepidx(input.shape, self.n)
        return fuzzy_dropin(input, updates)


class FuzzyNumKeepout(FuzzyNumKeep):
    def forward(self, input):
        updates = keepidx(input.shape, self.n)
        return fuzzy_dropout(input, updates)


class FuzzyNumKeepup(FuzzyNumKeep):
    def forward(self, input):
        updates = keepidx(input.shape, self.n)
        return fuzzy_dropup(input, updates)


class FuzzyNumKeepdown(FuzzyNumKeep):
    def forward(self, input):
        updates = keepidx(input.shape, self.n)
        return fuzzy_dropdown(input, updates)


class FuzzySignedAxisOp(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, logic: FuzzyLogic, keepn: int = None
    ):
        super().__init__()
        self._logic = logic
        self.weights = FuzzyParam((in_features, out_features))
        self.signs = FuzzyParam((in_features, out_features))
        self._keepn = keepn

    def keepn(self):
        return self._keepn if self.training else None


class FuzzySignedConjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
        signs = self.signs.value()
        pre_drop = self._logic.implies(weights, self._logic.bin_xnor(input, signs))
        post_drop = (
            take_rand_n(pre_drop, self.keepn(), dim=-2)
            if self.keepn() is not None
            else pre_drop
        )
        return self._logic.conjoin(post_drop, dim=1)


class FuzzySignedDisjunction(FuzzySignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
        signs = self.signs.value()
        pre_drop = self._logic.bin_conjoin(weights, self._logic.bin_xnor(input, signs))
        post_drop = (
            take_rand_n(pre_drop, self.keepn(), dim=-2)
            if self.keepn() is not None
            else pre_drop
        )
        return self._logic.disjoin(post_drop, dim=1)


class FuzzyUnsignedAxisOp(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, logic: FuzzyLogic, keepn: int = None
    ):
        super().__init__()
        self._logic = logic
        self.weights = FuzzyParam((in_features, out_features))
        self._keepn = keepn

    def keepn(self):
        return self._keepn if self.training else None


class FuzzyUnsignedConjunction(FuzzyUnsignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
        pre_drop = self._logic.implies(weights, input)
        post_drop = (
            take_rand_n(pre_drop, self.keepn(), dim=-2)
            if self.keepn() is not None
            else pre_drop
        )
        return self._logic.conjoin(post_drop, dim=1)


class FuzzyUnsignedDisjunction(FuzzyUnsignedAxisOp):
    def forward(self, input):
        input = input.unsqueeze(-1)
        weights = self.weights.value()
        pre_drop = self._logic.bin_conjoin(weights, input)
        post_drop = (
            take_rand_n(pre_drop, self.keepn(), dim=-2)
            if self.keepn() is not None
            else pre_drop
        )
        return self._logic.disjoin(post_drop, dim=1)


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
