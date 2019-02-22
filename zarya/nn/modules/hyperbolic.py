import torch.nn as nn


class Hyperbolic(nn.Module):
    def __init__(self, f, manifold):
        super(Hyperbolic, self).__init__()

        self.f = f
        self.mf = manifold

    def forward(self, input, dim=-1):
        tangent_input = self.mf.zero_log(input, dim)
        result = self.f(tangent_input)

        return self.mf.zero_exp(result, dim)
