import math

import torch
import torch.nn as nn
import torch.nn.init as init

from zarya import HTensor


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.m = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.m, a=math.sqrt(5))

    def forward(self, input: HTensor):
        if input.is_transposed():
            raise ValueError(
                "input.hdim should be equal to input.dim() - 1, {} found".format(
                    input.hdim
                )
            )

        return input.like(tensor=input.manifold.linear(input.tensor, self.m))
