import torch
import torch.nn as nn
import torch.nn.init as init

from zarya import HTensor
from zarya.nn import HParameter, HModule


class Hyperplane(HModule):
    def __init__(self, in_features, out_features, manifold):
        super(Hyperplane, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.manifold = manifold

        self.p = HParameter(
            nn.Parameter(torch.randn(out_features, in_features)),
            manifold=manifold,
            project=False,
        )
        self.a = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.p.tensor, -0.001, 0.001)
        self.p.proj_()

        init.uniform_(self.a, -0.001, 0.001)

    def forward(self, input: HTensor):
        assert (
            input.tensor.dim() == 2
        ), "input.dim() should be equal to 2, {} found".format(input.tensor.dim())

        assert (
            not input.is_transposed()
        ), "input.hdim should be equal to input.dim() - 1, {} found".format(input.hdim)

        b_s, _ = input.tensor.size()

        result = (
            input.tensor.unsqueeze(1)
            .repeat(1, self.out_features, 1)
            .view(-1, self.in_features)
        )

        a = input.manifold.parallel_transport(self.a, _to=self.p.tensor)

        p = self.p.tensor.unsqueeze(0).repeat(b_s, 1, 1).view(-1, self.in_features)
        a = a.unsqueeze(0).repeat(b_s, 1, 1).view(-1, self.in_features)

        result = input.manifold.hyperplane(result, p, a)
        return result.view(b_s, self.out_features)
