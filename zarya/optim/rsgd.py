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
                lr = group["lr"]
                for p in group["params"]:

                    if p.grad is None:
                        continue

                    lambda_square = self.mf.conf_factor(p, keepdim=True) ** 2
                    p.data.copy_(self.mf.exp(p, -lr * p.grad.data / lambda_square))
