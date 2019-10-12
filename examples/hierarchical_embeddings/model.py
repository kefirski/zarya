import torch
from torch import nn

from zarya import nn as znn


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
        self.embedding = znn.Embedding(vocab_size, emb_size, manifold)
        self.objective = nn.CrossEntropyLoss()

    def forward(self, succ, pred, neg):
        """Calculate hierarchy embedding loss value.

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

        union = torch.cat((succ, neg), dim=-1)
        pred_emb = self.embedding(pred)
        union_emb = self.embedding(union)
        distances = -self.mf.distance(pred_emb, union_emb)

        return self.objective(distances, target)

    def dump(self):
        """Detach and return embedding weights.

        Returns:
            torch.Tensor: Embedding weights.
        """
        return self.embedding.weight.detach().cpu()

    def renorm(self):
        self.mf.renorm(self.embedding.weight)
