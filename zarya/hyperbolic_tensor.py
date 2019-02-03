from math import sqrt

import torch


class HTensor(torch.Tensor):
    eps = 1e-5

    def __new__(
        cls, *args, c=1.0, requires_grad=False, projected=False, hdim=-1, **kwargs
    ):

        if c == 0:
            raise ValueError(
                "c=0 corresponds to Euclidean geometry. Use torch.Tensor instead"
            )

        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor.__new__(cls, *args, **kwargs)

        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))

        instance = torch.Tensor._make_subclass(cls, data, requires_grad)

        instance.kwargs = kwargs
        instance.c = c
        instance.hdim = hdim if hdim >= 0 else len(data.size()) + hdim
        instance.info = "c = {}, hdim = {}".format(c, instance.hdim)

        if not projected:
            instance.proj_()

        return instance

    def proj_(self):
        with torch.no_grad():
            transposed = self.is_transposed()

            if transposed:
                self.data = self._transpose_hdim()

            norm = torch.norm(self.data, dim=-1)
            norm.masked_fill_(norm < self.eps, self.eps)

            indices = self.c * norm >= 1

            if indices.any():
                self.data[indices] *= (1 / sqrt(self.c) - self.eps) / norm[
                    indices
                ].unsqueeze(1)

            if transposed:
                self.data = self._transpose_hdim()

    def htranspose(self, dim_a, dim_b):
        return HTensor(
            self.transpose(dim_a, dim_b),
            c=self.c,
            requires_grad=self.requires_grad,
            projected=True,
            hdim=self._hdim_after_transpose(self.hdim, dim_a, dim_b),
            kwargs=self.kwargs,
        )

    def is_transposed(self):
        return not self.hdim == len(self.size()) - 1

    def _transpose_hdim(self):
        return self.data.transpose(self.hdim, -1)

    @staticmethod
    def _hdim_after_transpose(hdim, dim_a, dim_b):
        if hdim == dim_a:
            hdim = dim_b
        elif hdim == dim_b:
            hdim = dim_a

        return hdim

    def __repr__(self):
        return super(HTensor, self).__repr__().replace(
            "tensor", "htensor"
        ) + ", {}".format(self.info)


if __name__ == "__main__":
    x = HTensor([[1, 2, 3], [4, 5, 6]])
    y = HTensor([[1, 4], [2, 5], [3, 6]], hdim=0)
    print(x)
    print(y)
    z = y.htranspose(0, 1)
    print(z)