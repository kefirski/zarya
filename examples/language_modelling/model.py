from math import log

import torch as t
import torch.nn as nn

import zarya.nn as znn
from examples.language_modelling.multihead import MultiHeadAttention
from examples.language_modelling.position_wise import PositionWise


class Model(nn.Module):
    def __init__(
        self, vocab_size, embedding_size, n_layers, n_heads, p_s, manifold, dropout=0.1
    ):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.mf = manifold

        self.embeddings = znn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            manifold=self.mf,
            padding_idx=0,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(n_heads, embedding_size, p_s, self.mf, dropout)
                for _ in range(n_layers)
            ]
        )

        self.out = znn.Hyperplane(embedding_size, vocab_size, self.mf)

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target, log_base=None):

        b_s, s_l = input.size()

        pos = (
            t.arange(1, s_l + 1, dtype=t.float, device=input.device)
            .repeat(b_s)
            .view(b_s, -1)
        )

        input = self.embeddings(input)

        mask = self.autoregressive_mask(b_s, s_l, input.device)

        recur = None
        for i, layer in enumerate(self.layers):
            input, recur = layer(
                input, self.mf.zero_exp(self.positional(pos, i + 1)), recur, mask
            )

        out = self.out(input.view(-1, self.embedding_size))
        loss = self.criterion(out, target.view(-1))

        if log_base is not None:
            loss = loss / log(log_base)

        return loss.mean().unsqueeze(0)

    def positional(self, indices, time):

        b_s, s_l = indices.size()
        indices = indices.unsqueeze(-1).repeat(1, 1, self.embedding_size)

        js = t.arange(0, self.embedding_size, dtype=t.float, device=indices.device)
        js = 10000 ** (2 * js / self.embedding_size)
        js = js.unsqueeze(0).repeat(s_l, 1).unsqueeze(0).repeat(b_s, 1, 1)

        positional = indices / js

        positional[:, :, 0::2] = t.sin(positional[:, :, 0::2])
        positional[:, :, 1::2] = t.cos(positional[:, :, 1::2])

        additional = time / js

        additional[:, :, 0::2] = t.sin(additional[:, :, 0::2])
        additional[:, :, 1::2] = t.cos(additional[:, :, 1::2])

        return positional + additional

    @staticmethod
    def autoregressive_mask(batch_size, length, device):
        mask = t.ones(length, length, dtype=t.uint8, device=device).tril_(-1)
        return (
            mask.transpose(0, 1).repeat(batch_size, 1).view(batch_size, length, length)
        )


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, h_s, p_s, manifold, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mf = manifold

        self.attention = MultiHeadAttention(n_heads, h_s, p_s, self.mf, dropout)
        self.position_wise = PositionWise(h_s, h_s * 4, self.mf, dropout)

    def forward(self, input, pos, recur=None, mask=None):
        if recur is None:
            recur = t.zeros_like(input, requires_grad=True)

        residual = input
        input = self.mf.add(input, pos)

        result = self.attention(q=input, k=input, v=input, residual=residual, mask=mask)

        return self.position_wise(result, recur), result
