import torch

import zarya
import zarya.nn as znn
from zarya.manifolds import PoincareBall


class Model(znn.HModule):
    def __init__(
        self, vocab_size, embedding_size, hidden_size, manifold=PoincareBall()
    ):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.manifold = manifold

        self.embeddings = znn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            manifold=self.manifold,
            padding_idx=0,
            sparse=True,
        )
        self.gru = znn.GRUCell(embedding_size, hidden_size, self.manifold)
        self.out = znn.Hyperplane(hidden_size, vocab_size, self.manifold)

    def forward(self, input):
        """
        :param input: Long tensor with shape [batch_size, seq_len]
        :return: Float tensor with shape [batch_size * seq_len, vocab_size] with result probs logits
        """

        batch_size, seq_len = input.size()

        input = self.embeddings(input)

        res = []
        hx = zarya.HTensor(
            torch.zeros(batch_size, self.hidden_size),
            manifold=self.manifold,
            project=False,
        )

        for x in input.split(1, 1):
            hx = self.gru(x.view(-1, self.embedding_size), hx)
            res += [hx]
        res = zarya.HTensor.stack(res, 1).view(-1, self.hidden_size)
        return self.out(res)
