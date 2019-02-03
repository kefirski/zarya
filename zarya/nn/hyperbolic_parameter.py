import torch

from zarya import HTensor


class HParameter(HTensor, torch.nn.Parameter):
    r"""A kind of Parameter that is considered to be a member of Poincaré Ball Model.

    HParameters are :class:`~torch.nn.Parameter` subclasses
    that have to be separated from usual Parameters,
    since it's necessary to scale its grad with
    the inverse of Poincaré metric tensor to perform RSGD
    and to project its data after optimization step to make it fit the ball of radius 1/sqrt(c).

    Arguments:
        data (HTensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
        c (float): Poincaré Ball parameter. Default: 1.0
    """
    eps = 1e-5

    def __new__(cls, data, c=1.0, requires_grad=True):
        if not isinstance(data, HTensor):
            data = HTensor(data, c=c)

        instance = HTensor._make_subclass(cls, data, requires_grad)
        instance.c = data.c

        return instance
