from typing import Iterable, Iterator
import torch

class GradSampler:
    def __init__(self, f, params: Iterable[torch.Tensor], on_loop=None):

        self.f = f
        self.on_loop = on_loop
        self.params = tuple(params)

        self.inputs = None
        self.outputs = None
        self.param_values = []
        self.grads = []

        for param in self.params:
            self.param_values.append(torch.ones((0, *param.shape)))
            self.grads.append(torch.ones((0, *param.shape)))

    def _append_input(self, input: torch.Tensor):
        with torch.no_grad():
            if self.inputs == None:
                self.inputs = input.clone().detach()
            else:
                self.inputs = torch.cat([self.inputs, input], dim=0)

    def _append_output(self, output: torch.Tensor):
        with torch.no_grad():
            with_batch = output.unsqueeze(0)
            if self.outputs == None:
                self.outputs = with_batch
            else:
                self.outputs = torch.cat([self.outputs, with_batch], dim=0)

    def _append_params(self):
        with torch.no_grad():
            for idx, param in enumerate(self.params):
                self.param_values[idx] = torch.cat([
                    self.param_values[idx], 
                    param.unsqueeze(0)
                ], dim=0)
                self.grads[idx] = torch.cat([
                    self.grads[idx], 
                    param.grad.unsqueeze(0)
                ], dim=0)

    def _zero_grad(self):
        for param in self.params:
            param.grad = None

    def loop(self, xss: Iterable[torch.Tensor]):
        self._zero_grad()
        for xs in xss:
            self._append_input(xs)
            for idx in range(xs.size(0)):
                if self.on_loop != None:
                    self.on_loop()
                x = xs[idx:idx+1]
                output = self.f(x)
                self._append_output(output)
                output.backward()
                self._append_params()
                self._zero_grad()
