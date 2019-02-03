import torch


class HParameter(torch.nn.Parameter):
    r"""A kind of Parameter that is considered to be a member of Poincaré Ball Model.

    HParameters are :class:`~torch.nn.Parameter` subclasses
    that have to be separated from usual Parameters,
    since it's necessary to scale its grad with
    the inverse of Poincaré metric tensor to perform RSGD
    and to renorm its data after optimization step to make it fit the ball of radius c.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
        c (float): Poincaré Ball radius. Default: 1.0
    """
    eps = 1e-8

    def __new__(cls, data, requires_grad=True, *args, **kwargs):
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, *args, **kwargs):
        super(HParameter, self).__init__()

        self.c = kwargs.get("c", 1.0)
        self.renorm()

    def renorm(self):
        with torch.no_grad():
            norm = torch.norm(self.data, dim=-1)
            norm.masked_fill_(norm < self.eps, self.eps)

            indices = norm >= self.c

            if indices.any():
                self.data[indices] *= (self.c - self.eps) / norm[indices].unsqueeze(1)
