import torch
import torch.optim as optim
from torch.optim.optimizer import required


class RSGD(optim.SGD):
    r"""Implements Riemannian stochastic gradient descent.
    """

    def __init__(self, params, manifold, lr=required):
        super(RSGD, self).__init__(params, lr)

        self.mf = manifold

    def step(self):
        """Performs a single optimization step.
        """

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:

                    if p.grad is None:
                        continue

                    p.data.copy_(
                        self.mf.exp(
                            p,
                            -group["lr"]
                            * p.grad.data
                            / self.mf.conf_factor(p, -1, keepdim=True) ** 2,
                        )
                    )
