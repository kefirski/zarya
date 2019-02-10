import unittest

import torch

from zarya import HTensor
from zarya.manifolds import PoincareBall


class PoincareBallTest(unittest.TestCase):
    def setUp(self):
        self.x = HTensor(
            torch.randn(5, 3, requires_grad=True), manifold=PoincareBall(), hdim=-1
        )
        self.y = HTensor(torch.randn(5, 3), manifold=PoincareBall(), hdim=-1)
        self.z = HTensor(
            torch.randn(5, 3, 2, requires_grad=True), manifold=PoincareBall(), hdim=0
        )

    def test_norm(self):
        for var in [self.x, self.y, self.z]:
            self.assertLess(torch.norm(var.tensor, dim=var.hdim).max().item(), 1.0)

    def test_transpose(self):
        transposed = self.x.transpose(0, 1)
        self.assertEqual(transposed.hdim, 0)

        transposed = self.z.transpose(1, 2)
        self.assertEqual(transposed.hdim, 0)

    def test_sum_and_mul(self):
        _sum = self.x + self.x + self.x
        _mul_1 = self.x * 3
        _mul_2 = 3 * self.x

        a = (_sum.tensor - _mul_1.tensor).abs()
        b = (_mul_1.tensor - _mul_2.tensor).abs()

        self.assertLessEqual(a.max().item(), 1e-5)
        self.assertLessEqual(b.max().item(), 1e-5)

    def test_log_exp(self):
        log = self.x.log(self.y)
        exp = self.x.exp(log)

        a = (self.y.tensor - exp.tensor).abs()

        self.assertLessEqual(a.max().item(), 1e-3)

        log = self.x.manifold.zero_log(self.x.tensor, dim=-1)
        exp = self.x.manifold.zero_exp(log, dim=-1)

        a = (self.x.tensor - exp).abs()

        self.assertLessEqual(a.max().item(), 1e-5)
