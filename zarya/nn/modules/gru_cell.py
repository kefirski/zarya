import math

import torch
import torch.nn as nn
import torch.nn.init as init

import zarya.manifolds as mf
from zarya import HTensor
from zarya.nn import HModule
from ..modules.linear import Linear
from ..modules.nonlinear import NonLinear


class GRUCell(HModule):
    def __init__(self, in_features, out_features, manifold: mf.Manifold):
        super(GRUCell, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.manifold = manifold

        self.w_r = Linear(out_features, out_features, bias=False)
        self.u_r = Linear(in_features, out_features, bias=True)

        self.w_z = Linear(out_features, out_features, bias=False)
        self.u_z = Linear(in_features, out_features, bias=True)

        self.u = Linear(in_features, out_features, bias=True)

        self.weight = nn.Parameter(torch.Tensor(out_features, out_features))

        self.tanh = NonLinear(nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: HTensor, hx: HTensor):
        """
        :param x: Float HTensor with shape [batch_size, in_features]
        :param hx: Float HTensor with shape [batch_size, out_features]
        :return: Float HTensor with shape [batch_size, out_features]
        """

        assert (
            not x.is_transposed() and not hx.is_transposed()
        ), "x and hidden_state are transposed"

        batch_size = x.size(0)

        r = torch.sigmoid((self.w_r(hx) + self.u_r(x)).zero_log())
        z = torch.sigmoid((self.w_z(hx) + self.u_z(x)).zero_log())

        w = self.weight.unsqueeze(0).repeat(batch_size, 1, 1) * r.unsqueeze(1).repeat(
            1, self.out_features, 1
        )

        intermediate_hx_next = self.tanh(
            HTensor.zero_exp(
                torch.bmm(w, hx.zero_log().unsqueeze(-1)).squeeze(-1), self.manifold
            )
            + self.u(x)
        )

        return hx + HTensor.zero_exp(
            z * (-hx + intermediate_hx_next).zero_log(), self.manifold
        )
