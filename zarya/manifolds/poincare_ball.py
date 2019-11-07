from math import sqrt

import torch

from zarya.manifolds import Manifold
from zarya.utils import atanh, asinh, acosh


class PoincareBall(Manifold):
    def __init__(self, c=1.0, eps=1e-5):

        if c == 0:
            raise ValueError(
                "c=0 corresponds to Euclidean geometry. Use torch.Tensor instead"
            )

        self.c = c
        self.sqrt_c = sqrt(c)
        self.eps = eps

    def proj_(self, x, dim=-1):
        with torch.no_grad():
            exp = self.zero_exp(x, dim=dim)
            x.copy_(exp)

    def renorm_(self, x, dim=-1):
        *_, d = x.shape
        x = x.view(-1, d)
        with torch.no_grad():
            x.renorm_(2, 0, 1 - self.eps)

    def conf_factor(self, x=None, dim=-1, keepdim=False):
        return (
            torch.clamp(
                2 / (1 - self.c * torch.sum(x * x, dim=dim, keepdim=keepdim)),
                min=self.eps,
            )
            if x is not None
            else 2.0
        )

    def add(self, x, y, dim=-1):

        c = self.c

        xy = torch.sum(x * y, dim=dim, keepdim=True)
        xx = torch.sum(x * x, dim=dim, keepdim=True)
        yy = torch.sum(y * y, dim=dim, keepdim=True)

        a = (1 + 2 * c * xy + c * yy) * x
        b = (1 - c * xx) * y
        c = 1 + 2 * c * xy + c * c * xx * yy

        self.clamp_inside_(c, -1e-12, 1e-12)

        return (a + b) / c

    def mul(self, x, r, dim=-1):
        x_norm = torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=self.eps)
        return (
            (1 / self.sqrt_c)
            * torch.tanh(torch.clamp(r * atanh(self.sqrt_c * x_norm), min=-15, max=15))
            * x
            / x_norm
        )

    def log(self, x, y, dim=-1):
        r"""
        Mapping of point y from Manifold to Tangent Space at point x
        """

        _sum = self.add(self.mul(x, -1, dim), y, dim)
        _sum_norm = torch.clamp(torch.norm(_sum, dim=dim, keepdim=True), min=self.eps)

        return (
            (2 / (self.sqrt_c * self.conf_factor(x, dim, keepdim=True)))
            * atanh(self.sqrt_c * _sum_norm)
            * _sum
            / _sum_norm
        )

    def exp(self, x, v, dim=-1):
        r"""
        Mapping of point v from Tangent space at point x back to Manifold
        """

        c_vv = self.sqrt_c * torch.clamp(
            torch.norm(v, dim=dim, keepdim=True), min=self.eps
        )

        return self.add(
            x,
            torch.tanh(self.conf_factor(x, dim, keepdim=True) * c_vv / 2) * v / c_vv,
            dim,
        )

    def zero_log(self, y, dim=-1):
        r"""
        Mapping of point y from Manifold to Tangent Space at point 0
        """

        y_norm = torch.clamp(torch.norm(y, dim=dim, keepdim=True), min=self.eps)
        return (1 / self.sqrt_c) * atanh(self.sqrt_c * y_norm) * y / y_norm

    def zero_exp(self, v, dim=-1):
        r"""
        Mapping of point v from Tangent space at point 0 back to Manifold
        """

        c_vv = self.sqrt_c * torch.clamp(
            torch.norm(v, dim=dim, keepdim=True), min=self.eps
        )
        return torch.tanh(torch.clamp(c_vv, min=-15, max=15)) * v / c_vv

    def parallel_transport(self, x, dim=-1, _from=None, _to=None):
        return (
            x
            * self.conf_factor(_from, dim=dim, keepdim=True)
            / self.conf_factor(_to, dim=dim, keepdim=True)
        )

    def linear(self, x, m):
        r"""
        Fused composition of zero_log mapping, linear mapping and zero_exp mapping
        """

        mx = x.matmul(m.t())
        mx_norm = torch.clamp(torch.norm(mx, dim=-1, keepdim=True), min=self.eps)
        x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=self.eps)

        return (
            (1 / self.sqrt_c)
            * torch.tanh(mx_norm * atanh(self.sqrt_c * x_norm) / x_norm)
            * mx
            / mx_norm
        )

    def hyperplane(self, x, p, a):
        _sum = self.add(self.mul(p, -1), x)
        _sum_norm_2 = torch.sum(_sum * _sum, dim=-1)
        a_norm = torch.norm(a, dim=-1)

        denominator = (1 - self.c * _sum_norm_2) * a_norm
        self.clamp_inside_(denominator, -self.eps, self.eps)

        return (self.conf_factor(p) * a_norm / self.sqrt_c) * asinh(
            (2 * self.sqrt_c * torch.sum(_sum * a, dim=-1)) / denominator
        )

    def distance(self, x, y, p="fro", dim=-1, keepdim=False):
        """Poincare Manifod Distance
        Args:
            x (torch.Tensor): input tensor.
            y (torch.Tensor): input tensor.
            p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of
                norm. Defaults to `fro`: frobenius form.
            dim (int, optional): dimension used. Defaults to -1.
            keepdim (bool, optional): whether to keep dims. Defaults to False.
        Returns:
            torch.Tensor: poincare distance.
        """
        alpha = 1 - torch.norm(x, p, dim, keepdim) ** 2
        beta = 1 - torch.norm(y, p, dim, keepdim) ** 2
        gamma = torch.norm(x - y, p, dim, keepdim) ** 2
        _val = 1 + 2 * gamma / (alpha * beta)
        _clamped = torch.clamp(_val, min=1)
        return acosh(_clamped)

    def clamp_inside_(self, value, _from, _to):
        indices = (value > _from) * (value < _to)
        if indices.any():
            value[indices] = self.eps * torch.sign(value[indices])

    def __repr__(self):
        return "Poincare Ball Manifold, c = {}".format(self.c)

    def __eq__(self, other):
        return self.c == other.c
