import random

import pytest
import torch

import zarya.nn as znn
from zarya.manifolds import PoincareBall

torch.manual_seed(42)
random.seed(42)

manifold = PoincareBall()
hdim = 1


def mul_coef(n, start, end):
    b = start
    a = (end - start) / n
    return lambda x: a * x + b


def proj(x):
    manifold.proj_(x, hdim)
    return x


def generate_input(tuple_size, n=1000):
    mul_f = mul_coef(n, 0.01, 0.5)
    return [
        tuple(
            [
                proj(torch.randn(5, 10, 2) * random.uniform(1e-5, mul_f(i)))
                for _ in range(tuple_size)
            ]
            + [mul_f(i)]
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("input", generate_input(1))
def test_norm(input):
    x, _ = input
    assert torch.norm(x, dim=hdim).max().item() <= 1 - 1e-5 + 1e-6


@pytest.mark.parametrize("input", generate_input(1))
def test_sum_mul(input):
    x, _ = input

    zero = torch.zeros_like(x)

    assert (
        manifold.add(x, x, dim=hdim) - manifold.mul(x, 2, dim=hdim)
    ).abs().max().item() <= 1e-5
    assert (manifold.add(x, zero, dim=hdim) - x).abs().max().item() == 0
    assert (manifold.add(zero, x, dim=hdim) - x).abs().max().item() == 0
    assert (zero - manifold.mul(x, 0, dim=hdim)).abs().max().item() == 0


@pytest.mark.parametrize("input", generate_input(2))
def test_log_exp(input):
    x, y, mul = input

    zero = torch.zeros_like(x)

    log = manifold.log(x, y, hdim)
    exp = manifold.exp(x, log, hdim)

    assert (y - exp).abs().mean().item() <= 2e-5

    log = manifold.zero_log(x, hdim)
    exp = manifold.zero_exp(log, hdim)

    assert (x - exp).abs().mean().item() <= 2e-5


@pytest.mark.parametrize("input", generate_input(1))
def test_linear(input):
    x, mul = input

    x = x.transpose(1, 2).contiguous()
    x = x.view(-1, x.size(-1))

    layer = znn.Linear(x.size(-1), 10, manifold, bias=False)
    result = manifold.zero_exp(manifold.zero_log(x).matmul(layer.weight.t()), dim=-1)
    assert (layer(x) - result).abs().mean().item() <= 1e-5

    eps = 1e-4 if mul < 0.2 else 5e-3

    layer = znn.Linear(x.size(-1), 10, manifold, bias=True)

    result = manifold.zero_exp(
        manifold.zero_log(x).matmul(layer.weight.t()) + layer.bias, dim=-1
    )
    assert (layer(x) - result).abs().mean().item() <= eps


def test_embed():
    embed = znn.Embedding(100, 5, manifold, 0)
    x = torch.LongTensor([[1, 2, 3, 0], [4, 5, 2, 0]])
    embed(x)


def test_gru():
    gru = znn.GRUCell(5, 10, manifold)

    x = torch.randn(3, 5) * 0.2
    manifold.proj_(x)

    hidden_state = torch.randn(3, 10) * 0.2
    manifold.proj_(hidden_state)

    res = gru(x, hidden_state)
