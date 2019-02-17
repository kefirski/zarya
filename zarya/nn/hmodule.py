import torch
from .hyperbolic_parameter import HParameter


class HModule(torch.nn.Module):
    def hparameters(self):
        res = []
        self.children()
        for val in self.__dict__.items():
            if isinstance(val[1], HParameter):
                res += [val[1]]

        for module in self.children():
            if isinstance(module, HModule):
                res += module.hparameters()

        return (val for val in res)
