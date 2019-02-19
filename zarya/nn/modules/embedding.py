import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from zarya.nn.parameter import Parameter


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        manifold,
        padding_idx=None,
        scale_grad_by_freq=False,
    ):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx

        self.mf = manifold
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight, -0.001, 0.001)
        self.mf.proj_(self.weight)

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            scale_grad_by_freq=self.scale_grad_by_freq,
        )
