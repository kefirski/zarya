import math

import torch
import torch.nn as nn
import torch.nn.init as init

from zarya import HTensor


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

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

    def forward(self, input: HTensor):
        if input.is_transposed():
            raise ValueError(
                "input.hdim should be equal to input.dim() - 1, {} found".format(
                    input.hdim
                )
            )

        result = input.like(tensor=input.manifold.linear(input.tensor, self.weight))
        return (
            result.exp(result.manifold.parallel_transport(self.bias, _to=result.tensor))
            if self.bias is not None
            else result
        )
