from math import log

import torch
import torch.nn as nn

import zarya.nn as znn


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, manifold):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.mf = manifold

        self.embeddings = znn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            manifold=self.mf,
            padding_idx=0,
        )
        self.gru = nn.ModuleList(
            [
                znn.GRUCell(
                    embedding_size if i == 0 else hidden_size, hidden_size, self.mf
                )
                for i in range(3)
            ]
        )
        self.out = znn.Hyperplane(hidden_size, vocab_size, self.mf)

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def logits(self, input):
        """
        :param input: Long tensor with shape [batch_size, seq_len]
        :return: Float tensor with shape [batch_size * seq_len, vocab_size] with result probs logits
        """

        batch_size, seq_len = input.size()

        input = self.embeddings(input)

        for gru in self.gru:

            res = []
            hx = torch.zeros(batch_size, self.hidden_size, device=input.device)

            for x in input.split(1, 1):
                hx = gru(x.view(-1, self.embedding_size), hx)
                res += [hx]

            input = torch.stack(res, 1)

        return self.out(input.view(-1, self.hidden_size))

    def forward(self, input, target, log_base=None):
        out = self.logits(input)
        loss = self.criterion(out, target.view(-1))

        if log_base is not None:
            loss = loss / log(log_base)

        return loss.mean().unsqueeze(0)
