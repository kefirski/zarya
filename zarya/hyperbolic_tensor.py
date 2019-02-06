import torch

import zarya.manifolds as mf
from zarya.utils import check_view


class HTensor:
    eps = 1e-5

    def __init__(self, tensor, manifold=mf.PoincareBall(), hdim=-1, project=True):
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

    def __add__(self, other):

        if self.manifold != other.manifold or self.hdim != other.hdim:
            raise ValueError("x: {} and y: {} found".format(self.info, other.info))

        return self.like(
            tensor=self.manifold.add(self.tensor, other.tensor, dim=self.hdim)
        )

    def transpose(self, dim0, dim1):
        return self.like(
            tensor=self.tensor.transpose(dim0, dim1),
            hdim=self._transposed_hdim(dim0, dim1),
        )

    def view(self, *new_size):

        tensor = self.tensor.view(*new_size)

        view_is_ok, hdim = check_view(
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
        return not self.hdim == len(self.tensor.size()) - 1

    def like(self, **kwargs):
        return HTensor(
            tensor=kwargs.get("tensor", self.tensor),
            manifold=kwargs.get("manifold", self.manifold),
            hdim=kwargs.get("hdim", self.hdim),
            project=kwargs.get("project", False),
        )

    def __repr__(self):
        return (
            "{} \n{}\n".format(self.tensor, self.info)
            .replace("tensor", "htensor\n")
            .replace("        ", "  ")
        )
