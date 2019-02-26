import torch as t
import torch.nn as nn
import torch.nn.init as init

from zarya.nn import Hyperbolic
from .product_attention import ProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, h_s, p_s, manifold, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.h_s = h_s

        self.mf = manifold

        self.q_proj = nn.Parameter(t.FloatTensor(n_heads, h_s, p_s))
        self.k_v_proj = nn.Parameter(t.FloatTensor(2 * n_heads, h_s, p_s))
        for param in [self.q_proj, self.k_v_proj]:
            init.kaiming_normal_(param.data)

        self.attention = ProductAttention(p_s)

        self.out = nn.Linear(n_heads * p_s, h_s)
        self.layer_norm = Hyperbolic(nn.LayerNorm(h_s, eps=1e-12), self.mf)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, residual=None, mask=None):
        """
        :param q: Float tensor with shape of [batch_size, query_len, h_s]
        :param k: Float tensor with shape of [batch_size, seq_len, h_s]
        :param v: Float tensor with shape of [batch_size, seq_len, h_s]
        :param mask: Byte tensor with shape of [batch_size, query_len, seq_len]
        :return: Float tensor with shape of [batch_size, query_len, h_s]
        """

        batch_size = q.size(0)

        q_len = q.size(1)
        seq_len = k.size(1)

        if residual is None:
            residual = q

        q = self.repeat_heads(q)
        k = self.repeat_heads(k)
        v = self.repeat_heads(v)
        k_v = t.cat([k, v], 0)

        q = self.mf.zero_log(q)
        k_v = self.mf.zero_log(k_v)

        q = t.bmm(q, self.q_proj).view(-1, q_len, self.q_proj.size(2))
        k, v = t.split(t.bmm(k_v, self.k_v_proj), self.n_heads, 0)
        k = k.view(-1, seq_len, self.k_v_proj.size(2))
        v = k.view(-1, seq_len, self.k_v_proj.size(2))

        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        result = self.attention(q, k, v, mask)
        result = t.split(result, batch_size, dim=0)
        result = t.cat(result, dim=-1)

        result = self.out(result)
        result = self.dropout(result)
        result = self.mf.zero_exp(result)

        return self.layer_norm(self.mf.add(result, residual))

    def repeat_heads(self, input):
        """
        :param input: Float tensor with shape of [batch_size, seq_len, hidden_size]
        :return: Float tensor with shape of [n_heads, batch_size * seq_len, hidden_size]
        """
        return input.repeat(self.n_heads, 1, 1).view(self.n_heads, -1, self.h_s)
