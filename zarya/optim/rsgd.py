import torch
import torch.optim as optim
from torch.optim.optimizer import required

from zarya import HTensor


class RSGD(optim.SGD):
    r"""Implements Riemannian stochastic gradient descent.
    """

    def __init__(self, params, lr=required):
        super(RSGD, self).__init__(params, lr)

    def step(self):
        """Performs a single optimization step.
        """

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:

                    if p.tensor.grad is None:
                        continue

                    p.tensor.data.copy_(
                        p.manifold.exp(
                            p.tensor,
                            -group["lr"]
                            * p.tensor.grad.data
                            / torch.clamp(p.conf_factor(keepdim=True) ** 2, min=1e-12),
                            p.hdim,
                        )
                    )

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group["params"]
        if isinstance(params, HTensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, HTensor):
                raise TypeError(
                    "hyperbolic optimizer can only optimize HTensors, "
                    "but one of the params is " + torch.typename(param)
                )
            if not param.tensor.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name
                )
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
