import random

import pytest
import torch

from zarya.manifolds import LorentzManifold

torch.manual_seed(42)
random.seed(42)

manifold = LorentzManifold()
dim = 3


def generate_input(*size, n=100):
    mul_f = torch.linspace(-0.5, 0.5, n).tolist()
    return [torch.randn(*size) * random.uniform(1e-5, mul_f[i]) for i in range(n)]


@pytest.mark.parametrize("input", generate_input(4, 5))
def test_norm(input):
    norm = manifold.norm(input)
    assert norm.shape == (4,)
    assert not torch.isnan(norm).any()


@pytest.mark.parametrize("x,y", zip(generate_input(4, 5), generate_input(4, 5)))
def test_distance(x, y):
    dist = manifold.distance(x, y)
    assert dist.shape == (4,)
    assert not torch.isnan(dist).any()


@pytest.mark.parametrize("x,v", zip(generate_input(4, 5), generate_input(4, 5)))
def test_exp(x, v):
    manifold.renorm_(x)
    exp = manifold.exp(x, v)
    assert exp.shape == (4, 5)
    assert not torch.isnan(exp).any()


@pytest.mark.parametrize("p,p_grad", zip(generate_input(4, 5), generate_input(4, 5)))
def test_grad_proj(p, p_grad):
    grad_proj = manifold.grad_proj(p, p_grad)
    assert grad_proj.shape == (4, 5)
    assert not torch.isnan(grad_proj).any()


@pytest.mark.parametrize("t", generate_input(4, 5))
def test_to_poincare(t):
    manifold.renorm_(t)
    poincare_t = manifold.to_poincare(t)
    assert poincare_t.shape == (4, 4)
    assert all(poincare_t.norm(dim=-1) <= 1)
