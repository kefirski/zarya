from zarya.manifolds import Manifold
from zarya.utils import acosh

import torch


class LorentzManifold(Manifold):
    def __repr__(self):
        return "Lorentz Manifold"

    def proj_(self, x: torch.Tensor, dim=-1):
        with torch.no_grad():
            self.renorm_(x, dim=dim)

    def dot(
        self, x: torch.Tensor, y: torch.Tensor, dim=-1, keepdim=False
    ) -> torch.Tensor:
        """Dot product

        Args:
            x (torch.Tensor): input tensor.
            y (torch.Tensor): input tensor.
            dim (int): dimension used. Defaults to -1.
            keepdim (bool): whether to keep dims. Default to False.

        Returns:
            torch.Tensor: dot product value.
        """
        dim_size = x.size(dim)
        neg = -x.narrow(dim, 0, 1) * y.narrow(dim, 0, 1)
        mult = x.narrow(dim, 1, dim_size - 1) * y.narrow(dim, 1, dim_size - 1)
        pos = torch.sum(mult, dim, keepdim=True)
        dot = neg + pos
        return dot if keepdim else torch.squeeze(dot, dim)

    def norm(self, x: torch.Tensor, dim=-1, keepdim=False) -> torch.Tensor:
        """Lorentz vector norm.

        Args:
            x (torch.Tensor): input tensor.
            dim (int, optional): dimension used. Defaults to -1.
            keepdim (bool, optional): whether to keep dims. Defaults to False.

        Returns:
            torch.Tensor: vector norm.
        """
        dot = self.dot(x, x, dim, keepdim)
        clamped_ = torch.clamp(dot, min=0)
        return torch.sqrt(clamped_)

    def renorm_(self, x: torch.Tensor, dim=-1):
        """Move point back to manifold.

        Args:
            x (torch.Tensor): input tensor.
            dim (int, optional): dimension used. Defaults to -1.
        """
        dim_size = x.size(dim)
        with torch.no_grad():
            x_ = x.narrow(dim, 1, dim_size - 1)
            n_ = torch.sqrt(1 + x_.norm(dim=dim, keepdim=True) ** 2)
            x.narrow(dim, 0, 1).copy_(n_)

    def distance(
        self, x: torch.Tensor, y: torch.Tensor, dim=-1, keepdim=False
    ) -> torch.Tensor:
        """Lorentz Manifod Distance

        Args:
            x (torch.Tensor): input tensor.
            y (torch.Tensor): input tensor.
            dim (int, optional): dimension used. Defaults to -1.
            keepdim (bool, optional): whether to keep dims. Defaults to False.
        Returns:
            torch.Tensor: lorentz distance.
        """
        dot = self.dot(x, y, dim, keepdim)
        _clamped = torch.clamp(-dot, min=1)
        return acosh(_clamped)

    def exp(self, x: torch.Tensor, v: torch.Tensor, dim=-1) -> torch.Tensor:
        """Mapping of point v from Tangent space at point x back to Manifold.

        Args:
            x (torch.Tensor): input tensor on manifold.
            v (torch.Tensor): input tensor from tangent space.
            dim (int, optional): dimension used. Defaults to -1.

        Returns:
            torch.Tensor: output tensor.
        """
        v_norm = self.norm(v, dim, keepdim=True)
        return torch.cosh(v_norm) * x + torch.sinh(v_norm) * v / v_norm

    def grad_proj(self, p: torch.Tensor, p_grad: torch.Tensor, dim=-1) -> torch.Tensor:
        """Projection from the ambient Euclidean space onto the tangent space
            of the current parameter.

        Args:
            p (torch.Tensor, optional): parameter. Defaults to None.
            p_grad (torch.Tensor, optional): parameter gradient. Defaults to None.
            dim (int, optional): dimension used. Defaults to -1.

        Returns:
            torch.Tensor: Gradient parameter tangent space projection.
        """
        g = torch.ones_like(p)
        g.narrow(dim, 0, 1).mul_(-1)
        h = p_grad * g
        return h + self.dot(p, h, dim=dim, keepdim=True) * p
