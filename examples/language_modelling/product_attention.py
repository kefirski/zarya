from math import sqrt

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ProductAttention(nn.Module):
    def __init__(self, s):
        """
        :param s: size of query
        """
        super(ProductAttention, self).__init__()

        self.scaling = 1 / (sqrt(s))

    def forward(self, q, k, v, mask=None):
        """
        :param q: Float tensor with shape of [batch_size, query_len, s]
        :param k: Float tensor with shape of [batch_size, seq_len, s]
        :param v: Float tensor with shape of [batch_size, seq_len, value_s]
        :param mask: Byte tensor with shape of [batch_size, query_len, seq_len]
        :return: Float tensor with shape of [batch_size, query_len, value_s]
        """

        attention = t.bmm(q, k.transpose(1, 2)) * self.scaling

        if mask is not None:
            attention.data.masked_fill_(mask.data, -float("inf"))

        attention = F.softmax(attention, dim=2)

        return t.bmm(attention, v)
