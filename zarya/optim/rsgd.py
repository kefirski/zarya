import torch
import torch.optim as optim
from torch.optim.optimizer import required


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
                            / torch.clamp(p.conf_factor(keepdim=True), min=1e-12),
                            p.hdim,
                        )
                    )
