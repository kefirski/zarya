import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from zarya import HTensor
from zarya.nn import HParameter, HModule


class Embedding(HModule):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        manifold,
        padding_idx=None,
        scale_grad_by_freq=False,
        sparse=False,
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

        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.weight = HParameter(
            nn.Parameter(torch.Tensor(num_embeddings, embedding_dim)),
            manifold=manifold,
            project=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight.tensor, -0.001, 0.001)
        self.weight.proj_()

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight.tensor[self.padding_idx].fill_(0)

    def forward(self, input):
        return HTensor(
            F.embedding(
                input,
                self.weight.tensor,
                self.padding_idx,
                None,
                None,
                self.scale_grad_by_freq,
                self.sparse,
            ),
            self.weight.manifold,
            project=False,
        )
