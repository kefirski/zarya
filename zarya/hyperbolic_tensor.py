import torch

import zarya.manifolds as mf
from zarya.utils import _check_view


class HTensor:
    eps = 1e-5

    def __init__(
        self, tensor, manifold=mf.PoincareBall(eps=eps), hdim=-1, project=True
    ):
        r"""
        Arguments:
            tensor (torch.Tensor): source tensor.
            manifold (zarya.Manifold): Manifold instance. Default: PoincareBall(c=1.0)
            hdim (int): index of hyperbolic data. Default: -1
            project (bool): whether to project source tensor on manifold. Default: True
        """

        self.tensor = tensor
        self.hdim = hdim if hdim >= 0 else len(tensor.size()) + hdim
        self.manifold = manifold
        self.info = "{}, hdim = {}".format(self.manifold, self.hdim)

        if project:
            self.proj_()

    def proj_(self):
        with torch.no_grad():
            is_transposed = self.is_transposed()

            if is_transposed:
                self.tensor = self.tensor.transpose(self.hdim, -1)

            self.manifold.proj_(self.tensor)

            if is_transposed:
                self.tensor = self.tensor.transpose(self.hdim, -1)

    def transpose(self, dim0, dim1):
        return self.like(
            tensor=self.tensor.transpose(dim0, dim1),
            hdim=self._transposed_hdim(dim0, dim1),
        )

    def view(self, *new_size):

        tensor = self.tensor.view(*new_size)

        view_is_ok, hdim = _check_view(
            list(self.tensor.size()), list(tensor.size()), self.hdim
        )

        if not view_is_ok:
            raise ValueError("View over HTensor should't affect hdim")

        return self.like(tensor=tensor, hdim=hdim)

    def is_transposed(self):
        return not self.hdim == len(self.tensor.size()) - 1

    def _transposed_hdim(self, dim0, dim1):
        if self.hdim == dim0:
            return dim1
        if self.hdim == dim1:
            return dim0

        return self.hdim

    def is_transposed(self):
        return not self.hdim == self.tensor.dim() - 1

    def like(self, **kwargs):
        return HTensor(
            tensor=kwargs.get("tensor", self.tensor),
            manifold=kwargs.get("manifold", self.manifold),
            hdim=kwargs.get("hdim", self.hdim),
            project=kwargs.get("project", False),
        )

    def log(self, y):
        if self.manifold != y.manifold or self.hdim != y.hdim:
            raise ValueError("x: {} and y: {} found".format(self.info, y.info))

        return self.manifold.log(self.tensor, y.tensor, dim=self.hdim)

    def exp(self, v):
        return self.like(tensor=self.manifold.exp(self.tensor, v, dim=self.hdim))

    def zero_log(self):
        return self.manifold.zero_log(self.tensor, dim=self.hdim)

    @staticmethod
    def zero_exp(v, manifold=mf.PoincareBall(eps=eps), hdim=-1):
        return HTensor(manifold.zero_exp(v, hdim), manifold, hdim, project=False)

    def __add__(self, other):

        if self.manifold != other.manifold or self.hdim != other.hdim:
            raise ValueError("x: {} and y: {} found".format(self.info, other.info))

        return self.like(
            tensor=self.manifold.add(self.tensor, other.tensor, dim=self.hdim)
        )

    def __sub__(self, other):
        return self.__add__(-other)

    def __rmul__(self, other):
        return self.like(tensor=self.manifold.mul(self.tensor, other, self.hdim))

    def __mul__(self, other):
        return self.like(tensor=self.manifold.mul(self.tensor, other, self.hdim))

    def __neg__(self):
        return self.like(tensor=self.manifold.neg(self.tensor, self.hdim))

    def __repr__(self):
        return (
            "{} \n{}\n".format(self.tensor, self.info)
            .replace("tensor", "htensor\n")
            .replace("        ", "  ")
        )
