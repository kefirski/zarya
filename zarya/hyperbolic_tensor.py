from math import sqrt

import torch


class HTensor(torch.Tensor):
    eps = 1e-5

    def __new__(cls, *args, c=1.0, requires_grad=False, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor.__new__(cls, *args, **kwargs)

        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))

        instance = torch.Tensor._make_subclass(cls, data, requires_grad)

        if c == 0:
            raise ValueError(
                "c=0 corresponds to Euclidean geometry. Use torch.Tensor instead"
            )

        instance.c = c
        instance.proj_()

        return instance

    def proj_(self):
        with torch.no_grad():
            norm = torch.norm(self.data, dim=-1)
            norm.masked_fill_(norm < self.eps, self.eps)

            indices = self.c * norm >= 1

            if indices.any():
                self.data[indices] *= (1 / sqrt(self.c) - self.eps) / norm[
                    indices
                ].unsqueeze(1)
