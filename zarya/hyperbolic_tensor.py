import torch

import zarya.manifolds as mf


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

        return HTensor(
            self.manifold.sum(self.tensor, other.tensor, dim=self.hdim),
            self.manifold,
            self.hdim,
            project=False,
        )

    def transpose(self, dim0, dim1):
        return HTensor(
            self.tensor.transpose(dim0, dim1),
            self.manifold,
            hdim=self._transposed_hdim(dim0, dim1),
            project=False,
        )

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

    def __repr__(self):
        return (
            "{} \n{}\n".format(self.tensor, self.info)
            .replace("tensor", "htensor\n")
            .replace("        ", "  ")
        )
