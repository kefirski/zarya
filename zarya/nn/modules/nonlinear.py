import torch.nn as nn

from zarya import HTensor


class NonLinear(nn.Module):
    def __init__(self, f):
        super(NonLinear, self).__init__()

        self.f = f

    def forward(self, input: HTensor):
        tangent_input = input.zero_log()
        result = self.f(tangent_input)

        return HTensor.zero_exp(result, input.manifold, hdim=input.hdim)
