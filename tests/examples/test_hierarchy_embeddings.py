import torch

from zarya.manifolds import PoincareBall, LorentzManifold
from examples.hierarchical_embeddings.model import (
    HierarchicalEmbeddings,
    EntailmentConesEmbeddings,
)

torch.manual_seed(42)

manifold = PoincareBall()


def test_forward():
    batch_size, negative_size = 8, 15

    pred, succ = torch.randint(10, (2, batch_size))
    negatives = torch.randint(10, (batch_size, negative_size))

    HierarchicalEmbeddings(10, 30, manifold).forward(pred, succ, negatives)


def test_entcon_poincare_forward():
    batch_size, negative_size = 8, 15

    pred, succ = torch.randint(10, (2, batch_size))
    negatives = torch.randint(10, (batch_size, negative_size))

    model = EntailmentConesEmbeddings(10, 30, PoincareBall())

    assert not torch.isnan(model(pred, succ, negatives))


def test_entcon_lorentz_forward():
    batch_size, negative_size = 8, 15

    pred, succ = torch.randint(10, (2, batch_size))
    negatives = torch.randint(10, (batch_size, negative_size))

    model = EntailmentConesEmbeddings(10, 30, LorentzManifold())

    assert not torch.isnan(model(pred, succ, negatives))
