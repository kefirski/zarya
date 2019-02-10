from math import sqrt

import torch

from zarya.manifolds import Manifold
from zarya.utils import atanh, asinh


class PoincareBall(Manifold):
    def __init__(self, c=1.0, eps=1e-5):

        if c == 0:
            raise ValueError(
                "c=0 corresponds to Euclidean geometry. Use torch.Tensor instead"
            )

        self.c = c
        self.sqrt_c = sqrt(c)
        self.eps = eps

    def proj_(self, x):
        with torch.no_grad():
            norm = torch.norm(x, dim=-1)
            norm = torch.clamp(norm, min=self.eps)

            indices = self.c * norm >= 1

            if indices.any():
                x[indices] *= ((1 / self.sqrt_c) - self.eps) / norm[indices].unsqueeze(
                    1
                )

    def conf_factor(self, x, dim=-1, keepdim=False):
        return 2 / (1 - self.c * torch.sum(x * x, dim=dim, keepdim=keepdim))

    def zero_conf_factor(self):
        return 2.0

    def add(self, x, y, dim=-1):

        c = self.c

        xy = torch.sum(x * y, dim=dim, keepdim=True)
        xx = torch.sum(x * x, dim=dim, keepdim=True)
        yy = torch.sum(y * y, dim=dim, keepdim=True)

        a = (1 + 2 * c * xy + c * yy) * x
        b = (1 - c * xx) * y
        c = 1 + 2 * c * xy + (c ** 2) * xx * yy

        return (a + b) / c

    def mul(self, x, r, dim=-1):
        x_norm = torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=self.eps)
        return (
            (1 / self.sqrt_c) * torch.tanh(r * atanh(self.sqrt_c * x_norm)) * x / x_norm
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
        return torch.tanh(c_vv) * v / c_vv

    def parallel_transport(self, x, dim=-1, _from=None, _to=None):
        return (
            x
            * (
                self.conf_factor(_from, dim=dim, keepdim=True)
                if _from is not None
                else self.zero_conf_factor()
            )
            / (
                self.conf_factor(_to, dim=dim, keepdim=True)
                if _to is not None
                else self.zero_conf_factor()
            )
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

        return (self.conf_factor(p) * a_norm / self.sqrt_c) * asinh(
            (2 * self.sqrt_c * torch.sum(_sum * a, dim=-1))
            / ((1 - self.c * _sum_norm_2) * a_norm)
        )

    def __repr__(self):
        return "Poincare Ball Manifold, c = {}".format(self.c)

    def __eq__(self, other):
        return self.c == other.c
