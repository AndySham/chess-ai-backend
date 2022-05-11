from cmath import exp
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

    def crisp_value(self) -> Tensor:
        return (self.value() > 0.5).float()


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


class FuzzyLoss(nn.Module):
    def __init__(self, logic: FuzzyLogic, exp=1, memory=0.99, fix_imbalance=False):
        super().__init__()
        self.logic = logic
        self.exp = exp
        self.count_1 = 0
        self.count_0 = 0
        self.memory = memory
        self.fix_imbalance = fix_imbalance

    def forward(self, y, y_hat):
        self.count_1 *= self.memory
        self.count_0 *= self.memory
        self.count_1 += (y == True).sum().item()
        self.count_0 += (y == False).sum().item()


        diffs = self.logic.bin_xnor(y, y_hat) ** self.exp
        if self.fix_imbalance:
            avg_count = (self.count_1 + self.count_0)
            #print(avg_count/self.count_1, avg_count/self.count_0)
            if self.count_1 != 0:
                diffs[y == True] *= avg_count / self.count_1
            if self.count_0 != 0:
                diffs[y == False] *= avg_count / self.count_0
        return diffs.sum()


def fuzzy_loss(y, y_hat, flogic, fix_imbalance=False, exp=1):
    diffs = flogic.bin_xnor(y, y_hat) ** exp
    if fix_imbalance:
        total_count = diffs.size(0)
        diffs[y == True] *= 1 / (2*(y == True).sum())
        diffs[y == False] *= 1 / (2*(y == False).sum())
    return diffs.mean()


class FuzzyMLP(nn.Module):
    def __init__(self, logic: FuzzyLogic, shape: tuple[int,...], clauses_per_output=10, signed_clauses=True):
        super().__init__()
        shapes = [(shape[i], shape[i+1]) for i in range(len(shape) - 1)]
        layers = []
        for idx, (in_f, out_f) in enumerate(shapes):
            layers.append(nn.Linear(in_f, out_f))
            if idx == len(shapes) - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.LeakyReLU())

        self.model = nn.Sequential(*layers)

        input_dim = shape[0]
        output_dim = shape[-1]

        self.clauses_per_output = clauses_per_output
        self.signed_clauses = signed_clauses

        self.logic = logic

        self.fuzzy_m = FuzzyParam((output_dim, clauses_per_output, input_dim))
        self.fuzzy_s = FuzzyParam((output_dim, clauses_per_output, input_dim))

        with torch.no_grad():
            self.fuzzy_m.param -= 1

    def forward(self, input):
        return self.model(input)

    def keepn(self):
        return 16 if self.training else None

    def literals(self, input):
        input = input.unsqueeze(1).unsqueeze(1)
        if self.signed_clauses:
            return self.logic.bin_xor(self.fuzzy_s.value(), input)
        else: 
            return input

    def memberships(self, input):
        literals = self.literals(input)
        if self.signed_clauses:
            return literals
        else: 
            return self.logic.implies(self.fuzzy_m.value(), literals)

    def preconditions(self, input):
        memberships = self.memberships(input)
        if self.training:
            memberships = take_rand_n(memberships, 16, dim=-1)
        return self.logic.conjoin(memberships, dim=-1)

    def satisfactions(self, input, output):
        output = output.unsqueeze(-1)
        preconditions = self.preconditions(input)
        return self.logic.implies(preconditions, output)

    def examples(self, input, output):
        output = output.unsqueeze(-1)
        preconditions = self.preconditions(input)
        return self.logic.bin_conjoin(preconditions, output)

    def logic_preds(self, input):
        preconditions = self.preconditions(input)
        return self.logic.disjoin(preconditions, dim=-1)


    def crispness(self):
        if self.signed_clauses:
            return torch.min(self.fuzzy_s.value(), 1 - self.fuzzy_s.value())
        else:
            return torch.min(self.fuzzy_m.value(), 1 - self.fuzzy_m.value())

    def crisp_satisfactions(self, input, output):
        input = input.unsqueeze(1).unsqueeze(1)
        output = output.unsqueeze(-1)
        if self.signed_clauses:
            literals = self.logic.bin_xor(self.fuzzy_s.crisp_value(), input)
        else: 
            literals = input
        memberships = self.logic.implies(self.fuzzy_m.crisp_value(), literals)
        preconditions = self.logic.conjoin(memberships, dim=-1)
        examples = self.logic.implies(preconditions, output)
        return examples


