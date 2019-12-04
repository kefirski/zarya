from typing import Union

import torch
from torch import nn

from zarya import nn as znn
from zarya.manifolds import LorentzManifold, PoincareBall


class HierarchicalEmbeddings(nn.Module):
    """Hierarchical Embedding Model.

    Args:
        vocab_size (int): Vocabulary size.
        emb_size (int): Embedding size.
        manifold (zarya.manifolds.Manifold): Hyperbolic manifold.
    """

    def __init__(self, vocab_size, emb_size, manifold):
        super().__init__()

        self.mf = manifold
        self.embeddings = znn.Embedding(vocab_size, emb_size, manifold)
        self.objective = nn.CrossEntropyLoss()

    def forward(self, pred, succ, neg):
        """Calculate hierarchy embeddings loss value.

        Args:
            succ (torch.Tensor): Successor of hierarchy pair.
            pred (torch.Tensor): Predecessor of hierarchy pair.
            neg (torch.Tensor): Random negative samples.

        Returns:
            torch.Tensor: Calculated hierarchy loss.
        """
        target = torch.zeros_like(succ)
        succ = succ.view(-1, 1)
        pred = pred.view(-1, 1)

        union = torch.cat((pred, neg), dim=-1)
        succ_emb = self.embeddings(succ)
        union_emb = self.embeddings(union)
        distances = -self.mf.distance(succ_emb, union_emb)

        return self.objective(distances, target)

    def dump(self):
        """Detach and return embeddings weights.

        Returns:
            torch.Tensor: Embedding weights.
        """
        return self.embeddings.weight.detach().cpu()

    def renorm(self):
        with torch.no_grad():
            self.mf.renorm_(self.embeddings.weight)


class EntailmentConesEmbeddings(nn.Module):
    """Hyperbolic Entailment Cones model.
        See https://arxiv.org/pdf/1804.01882.pdf paper for more details.

    Args:
        vocab_size (int): vocabulary size.
        emb_size (int): embedding size.
        manifold (Union[PoincareBall, LorentzManifold]): manifold used.
        k (float, optional): k const value. Defaults to 0.1.
        gamma (int, optional): gamma const value. Defaults to 1.

    Shapes:
        - **word**: `(B)`
        - **expl**: `(B)`
        - **negative**: `(B, N)`

        Where B is batch size, N is a number of negative samples.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        manifold: Union[PoincareBall, LorentzManifold],
        k=0.1,
        gamma=1,
    ):

        super().__init__()

        self.k, self.gamma = k, gamma
        self.mf = manifold
        self.embeddings = znn.Embedding(vocab_size, emb_size, manifold)
        if isinstance(manifold, PoincareBall):
            self.forward = self.poincare_forward
        elif isinstance(manifold, LorentzManifold):
            self.forward = self.lorentz_forward
        else:
            raise RuntimeError(f"Entailment cones model does not support {manifold}")

    def p_psi(self, x):
        x_norm = torch.clamp(x.norm(dim=-1), min=0.00001)
        arg = self.k * (1 - x_norm ** 2) / x_norm
        return torch.asin(arg.clamp(min=-0.999, max=0.999))

    def p_ksi(self, x, y):
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        x_norm2 = x_norm ** 2
        y_norm2 = y_norm ** 2
        diff_norm = torch.norm(x - y, dim=-1)
        xy_dot = torch.sum(x * y, dim=-1)
        numer = xy_dot * (1 + x_norm2) - x_norm2 * (1 + y_norm2)
        sqrt_arg = 1 + x_norm2 * y_norm2 - 2 * xy_dot
        denom = torch.clamp(x_norm * diff_norm * torch.sqrt(sqrt_arg), min=0.00001)
        arg = numer / denom
        return torch.acos(arg.clamp(min=-0.999, max=0.999))

    def l_psi(self, x):
        arg = 2 * self.k / (-x.narrow(-1, 0, 1) - 1)
        return torch.asin(torch.clamp(arg, min=-0.999, max=0.999))

    def l_ksi(self, x, y):
        *_, dim = x.shape
        dot = self.mf.dot(x, y, keepdim=True)
        narrow_y = y.narrow(-1, 1, dim - 1)
        numer = y.narrow(-1, 0, 1) + x.narrow(-1, 0, 1) * dot
        denom = narrow_y.norm(dim=-1, keepdim=True) * torch.sqrt(
            torch.clamp(dot ** 2 - 1, min=1e-5)
        )
        arg = numer / denom
        return torch.acos(arg.clamp(min=-0.99999, max=0.99999))

    def p_e(self, u, v):
        k = self.p_ksi(u, v)
        p = self.p_psi(u)
        return torch.clamp(k - p, min=0)

    def l_e(self, u, v):
        k = self.l_ksi(u, v)
        p = self.l_psi(u)
        return torch.clamp(k - p, min=0)

    def poincare_forward(self, word, expl, negative):
        word_emb = self.embeddings(word)
        expl_emb = self.embeddings(expl)
        negative_emb = self.embeddings(negative)
        expl_negative_emb = expl_emb.unsqueeze(1).expand_as(negative_emb)

        e_pos = self.p_e(word_emb, expl_emb).unsqueeze(1)
        e_neg = torch.clamp(
            self.gamma - self.p_e(expl_negative_emb, negative_emb), min=0
        )
        loss = torch.cat((e_pos, e_neg), dim=-1).mean()
        return loss

    def lorentz_forward(self, word, expl, negative):
        word_emb = self.embeddings(word)
        expl_emb = self.embeddings(expl)
        negative_emb = self.embeddings(negative)
        expl_negative_emb = expl_emb.unsqueeze(1).expand_as(negative_emb)

        e_pos = self.l_e(expl_emb, word_emb).unsqueeze(1)
        e_neg = torch.clamp(
            self.gamma - self.l_e(expl_negative_emb, negative_emb), min=0
        )
        loss = torch.cat((e_pos, e_neg), dim=1).mean()
        return loss

    def dump(self):
        """Detach and return embeddings weights.

        Returns:
            torch.Tensor: Embedding weights.
        """
        return self.embeddings.weight.detach().cpu()

    def renorm(self):
        with torch.no_grad():
            self.mf.renorm_(self.embeddings.weight)
