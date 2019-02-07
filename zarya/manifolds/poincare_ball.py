from math import sqrt

import torch
from zarya.utils import atanh
from zarya.manifolds import Manifold


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
                x[indices] *= (1 / self.sqrt_c) / norm[indices].unsqueeze(1)

    def conf_factor(self, x, dim, keepdim=False):
        return 2 / (1 - self.c * torch.sum(x * x, dim=dim, keepdim=keepdim))

    def add(self, x, y, dim):

        c = self.c

        xy = torch.sum(x * y, dim=dim, keepdim=True)
        xx = torch.sum(x * x, dim=dim, keepdim=True)
        yy = torch.sum(y * y, dim=dim, keepdim=True)

        a = (1 + 2 * c * xy + c * yy) * x
        b = (1 - c * xx) * y
        c = 1 + 2 * c * xy + (c ** 2) * xx * yy

        return (a + b) / c

    def mul(self, x, r, dim):
        x_norm = torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=self.eps)
        return (
            (1 / self.sqrt_c) * torch.tanh(r * atanh(self.sqrt_c * x_norm)) * x / x_norm
        )

    def log(self, x, y, dim):
        r"""
        Mapping of point y from Manifold to Tangent Space at point x
        """

        x_neg_plus_y = self.add(self.mul(x, -1, dim), y, dim)
        x_neg_plus_y_norm = torch.norm(x_neg_plus_y, dim=dim, keepdim=True)

        return (
            (2 / (self.sqrt_c * self.conf_factor(x, dim, keepdim=True)))
            * atanh(self.sqrt_c * x_neg_plus_y_norm)
            * x_neg_plus_y
            / x_neg_plus_y_norm
        )

    def exp(self, x, v, dim):
        r"""
        Mapping of point v from Tangent space at point x back to Manifold
        """

        c_v_norm = self.sqrt_c * torch.clamp(
            torch.norm(v, dim=dim, keepdim=True), min=self.eps
        )
        return self.add(
            x,
            torch.tanh(self.conf_factor(x, dim, keepdim=True) * c_v_norm / 2)
            * v
            / c_v_norm,
            dim,
        )

    def __repr__(self):
        return "Poincare Ball Manifold, c = {}".format(self.c)

    def __eq__(self, other):
        return self.c == other.c
