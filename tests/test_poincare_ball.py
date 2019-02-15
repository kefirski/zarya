import random

import pytest
import torch

import zarya.nn as nn
from zarya import HTensor
from zarya.manifolds import PoincareBall

torch.manual_seed(42)
random.seed(42)


def mul_coef(n, start, end):
    b = start
    a = (end - start) / n
    return lambda x: a * x + b


def generate_input(tuple_size, n=1000):
    mul_f = mul_coef(n, 0.01, 0.5)
    return [
        tuple(
            [
                HTensor(
                    torch.randn(5, 10, 2) * random.uniform(1e-5, mul_f(i)),
                    manifold=PoincareBall(),
                    hdim=1,
                )
                for _ in range(tuple_size)
            ]
            + [mul_f(i)]
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("input", generate_input(1))
def test_norm(input):
    x, _ = input
    assert torch.norm(x.tensor, dim=x.hdim).max().item() <= 1 - 1e-5 + 1e-6


@pytest.mark.parametrize("input", generate_input(1, n=10))
def test_view(input):
    x, _ = input
    x.transpose(1, 2).view(-1, x.tensor.size(1))


@pytest.mark.parametrize("input", generate_input(1, n=10))
def test_transpose(input):
    x, _ = input
    assert x.transpose(0, 1).hdim == 0
    assert x.transpose(1, 2).hdim == 2
    assert x.transpose(0, 2).hdim == 1


@pytest.mark.parametrize("input", generate_input(1))
def test_sum_mul(input):
    x, _ = input

    zero = HTensor(torch.zeros_like(x.tensor), manifold=x.manifold, hdim=1)

    assert ((x + x).tensor - (2 * x).tensor).abs().max().item() <= 1e-5
    assert ((x + zero).tensor - (x).tensor).abs().max().item() == 0
    assert ((zero + x).tensor - (x).tensor).abs().max().item() == 0
    assert ((zero).tensor - (x * 0).tensor).abs().max().item() == 0


@pytest.mark.parametrize("input", generate_input(2))
def test_log_exp(input):
    x, y, mul = input

    zero = HTensor(torch.zeros_like(x.tensor), hdim=1, project=False)

    log = x.log(y)
    exp = x.exp(log)

    assert (y.tensor - exp.tensor).abs().mean().item() <= 2e-5

    log = x.zero_log()
    exp = x.manifold.zero_exp(log, dim=x.hdim)

    assert (x.tensor - exp).abs().mean().item() <= 2e-5


@pytest.mark.parametrize("input", generate_input(1))
def test_linear(input):
    x, mul = input

    x = x.transpose(1, 2)
    x = x.view(-1, x.tensor.size(-1))

    layer = nn.Linear(x.tensor.size(-1), 10, bias=False)
    result = x.manifold.zero_exp(x.zero_log().matmul(layer.weight.t()), dim=-1)
    assert (layer(x).tensor - result).abs().mean().item() <= 1e-5

    eps = 1e-4 if mul < 0.2 else 5e-3

    layer = nn.Linear(x.tensor.size(-1), 10, bias=True)

    result = x.manifold.zero_exp(
        x.zero_log().matmul(layer.weight.t()) + layer.bias, dim=-1
    )
    assert (layer(x).tensor - result).abs().mean().item() <= eps


def test_embed():
    embed = nn.Embedding(100, 5, PoincareBall(), 0)
    x = torch.LongTensor([[1, 2, 3, 0], [4, 5, 2, 0]])
    embed(x)


def test_gru():
    gru = nn.GRUCell(5, 10, PoincareBall())

    x = HTensor(torch.randn(3, 5) * 0.2)
    hidden_state = HTensor(torch.randn(3, 10) * 0.2)
    res = gru(x, hidden_state)
