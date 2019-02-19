import math

import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, manifold, bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.mf = manifold

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.uniform_(self.bias, -1e-3, 1e-3)

    def forward(self, input):

        result = self.mf.linear(input, self.weight)
        return (
            self.mf.exp(result, self.mf.parallel_transport(self.bias, _to=result))
            if self.bias is not None
            else result
        )
