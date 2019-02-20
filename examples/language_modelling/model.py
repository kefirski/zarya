import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.gru = znn.GRUCell(embedding_size, hidden_size, self.mf)
        self.out = znn.Hyperplane(hidden_size, vocab_size, self.mf)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input, target):
        out, _ = self._forward(input)
        return self.criterion(out, target.view(-1))

    def _forward(self, input, hx=None):
        """
        :param input: Long tensor with shape [batch_size, seq_len]
        :param hx: Float tensor with shape [batch_size, hidden_size]
        :return: Float tensor with shape [batch_size * seq_len, vocab_size] with result probs logits
        """

        batch_size, seq_len = input.size()

        input = self.embeddings(input)

        res = []
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=input.device)

        for x in input.split(1, 1):
            hx = self.gru(x.view(-1, self.embedding_size), hx)
            res += [hx]

        res = torch.stack(res, 1).view(-1, self.hidden_size)
        return self.out(res), hx

    def generate(self, idx, device):
        idx = torch.LongTensor([[idx]]).to(device)
        hx = None

        res = []

        for _ in range(500):

            out, hx = self._forward(idx, hx)
            out = F.softmax(
                1.5 * out.squeeze(), dim=-1
            )  # 1.5 is for increasing the temperature of sampling
            out = out.cpu().numpy()

            idx = int(np.random.choice(self.vocab_size, 1, p=out)[0])

            if idx == 2:
                break

            res += [idx]
            idx = torch.LongTensor([[idx]]).to(device)

        return res
