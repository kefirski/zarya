from math import sqrt

import torch

from zarya.manifolds import Manifold


class PoincareBall(Manifold):
    def __init__(self, c=1.0, eps=1e-5):

        if c == 0:
            raise ValueError(
                "c=0 corresponds to Euclidean geometry. Use torch.Tensor instead"
            )

        self.c = c
        self.eps = eps

    def proj_(self, x):
        with torch.no_grad():
            norm = torch.norm(x, dim=-1)
            norm.masked_fill_(norm < self.eps, self.eps)

            indices = self.c * norm >= 1

            if indices.any():
                x[indices] *= (1 / sqrt(self.c) - self.eps) / norm[indices].unsqueeze(1)

    def sum(self, x, y, dim):

        c = self.c

        dot = torch.sum(x * y, dim=dim, keepdim=True)
        x_norm_2 = torch.sum(x * x, dim=dim, keepdim=True)
        y_norm_2 = torch.sum(y * y, dim=dim, keepdim=True)

        alpha = (1 + 2 * c * dot + c * y_norm_2) * x
        betta = (1 - c * x_norm_2) * y

        gamma = 1 + 2 * c * dot + (c ** 2) * x_norm_2 * y_norm_2

        return (alpha + betta) / gamma

    def __repr__(self):
        return "Poincare Ball Manifold, c = {}".format(self.c)

    def __eq__(self, other):
        return self.c == other.c
