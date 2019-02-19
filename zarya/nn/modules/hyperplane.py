import torch
import torch.nn as nn
import torch.nn.init as init

from zarya.nn.parameter import Parameter


class Hyperplane(nn.Module):
    def __init__(self, in_features, out_features, manifold):
        super(Hyperplane, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.mf = manifold

        self.p = Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.p, -0.001, 0.001)
        self.mf.proj_(self.p)

        init.uniform_(self.a, -0.001, 0.001)

    def forward(self, input):
        b_s, _ = input.size()

        result = (
            input.unsqueeze(1)
            .repeat(1, self.out_features, 1)
            .view(-1, self.in_features)
        )

        a = self.mf.parallel_transport(self.a, _to=self.p)

        p = self.p.unsqueeze(0).repeat(b_s, 1, 1).view(-1, self.in_features)
        a = a.unsqueeze(0).repeat(b_s, 1, 1).view(-1, self.in_features)

        result = self.mf.hyperplane(result, p, a)
        return result.view(b_s, self.out_features)
