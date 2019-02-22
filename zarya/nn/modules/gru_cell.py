import math

import torch
import torch.nn as nn
import torch.nn.init as init

from ..modules.hyperbolic import Hyperbolic
from ..modules.linear import Linear


class GRUCell(nn.Module):
    def __init__(self, in_features, out_features, manifold):
        super(GRUCell, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.mf = manifold

        self.w_r = Linear(out_features, out_features, manifold, bias=False)
        self.u_r = Linear(in_features, out_features, manifold, bias=True)

        self.w_z = Linear(out_features, out_features, manifold, bias=False)
        self.u_z = Linear(in_features, out_features, manifold, bias=True)

        self.u = Linear(in_features, out_features, manifold, bias=True)

        self.w = nn.Parameter(torch.Tensor(out_features, out_features))

        self.tanh = Hyperbolic(nn.Tanh(), manifold)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x, hx):
        """
        :param x: Float Tensor with shape [batch_size, in_features]
        :param hx: Float Tensor with shape [batch_size, out_features]
        :return: Float Tensor with shape [batch_size, out_features]
        """

        batch_size = x.size(0)

        r = torch.sigmoid(self.mf.zero_log(self.mf.add(self.w_r(hx), self.u_r(x))))
        z = torch.sigmoid(self.mf.zero_log(self.mf.add(self.w_z(hx), self.u_z(x))))

        w = self.w.unsqueeze(0).repeat(batch_size, 1, 1) * r.unsqueeze(1).repeat(
            1, self.out_features, 1
        )

        intermediate_hx_next = self.tanh(
            self.mf.add(
                self.mf.zero_exp(
                    torch.bmm(w, self.mf.zero_log(hx).unsqueeze(-1)).squeeze(-1)
                ),
                self.u(x),
            )
        )

        return self.mf.add(
            hx,
            self.mf.zero_exp(
                z * self.mf.zero_log(self.mf.add(self.mf.neg(hx), intermediate_hx_next))
            ),
        )
