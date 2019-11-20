import torch
import torch.optim as optim
from torch.optim.optimizer import required


class RSGD(optim.SGD):
    r"""Implements Riemannian stochastic gradient descent.
    """

    def __init__(self, params, manifold, lr=required, retraction=False):
        super(RSGD, self).__init__(params, lr)

        self.mf = manifold
        self.retraction = retraction

    def step(self):
        """Performs a single optimization step.
        """

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:

                    if p.grad is None:
                        continue

                    step = -lr * self.mf.grad_proj(p, p.grad.data)
                    update = p + step if self.retraction else self.mf.exp(p, step)
                    p.data.copy_(update)
