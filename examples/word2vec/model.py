import torch
from torch import nn


class Word2Vec(nn.Module):
    """Word2Vec classic model.

    Args:
        vocab_size (int): vocabulary size.
        emb_size (int): embedding size.

    Inputs:
        center (torch.Tensor): center words.
        context (torch.Tensor): context words.
        negative (torch.Tensor): negative words.

    Shapes:
        center: (B, )
        context: (B, )
        negative: (B, N)
        Where B - batch size, N - number of negative samples.
    """

    def __init__(self, vocab_size: int, emb_size: int):

        super(Word2Vec, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, emb_size, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, emb_size, sparse=True)

        a = 0.5 / emb_size
        nn.init.uniform_(self.center_embeddings.weight.data, -a, a)
        nn.init.uniform_(self.context_embeddings.weight.data, -a, a)

    def forward(
        self, center: torch.Tensor, context: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """Model forward pass.

        Args:
            center (torch.Tensor): center words.
            context (torch.Tensor): context words.
            negative (torch.Tensor): negative words.

        Returns:
            torch.Tensor: loss value.
        """
        center_emb = self.center_embeddings(center).unsqueeze(1)
        context_emb = self.context_embeddings(context).unsqueeze(1)
        negative_emb = self.context_embeddings(negative)
        center_context_dot = nn.functional.logsigmoid(
            torch.sum(context_emb * center_emb, -1)
        )
        center_negative_dot = nn.functional.logsigmoid(
            -torch.sum(negative_emb * center_emb, -1)
        )
        loss = -torch.cat((center_context_dot, center_negative_dot), dim=-1).mean()
        return loss
