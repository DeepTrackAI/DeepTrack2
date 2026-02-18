# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

import torch

from deeptrack.backend.array_api_compat_ext.torch import random as rnd


class TestRandom(unittest.TestCase):
    def test_rand(self):
        torch.manual_seed(0)

        # No-arg call should return a scalar (NumPy-compatible behavior).
        x0 = rnd.rand()
        self.assertIsInstance(x0, torch.Tensor)
        self.assertEqual(x0.ndim, 0)
        self.assertGreaterEqual(float(x0), 0.0)
        self.assertLess(float(x0), 1.0)

        # Shape via positional arguments.
        x = rnd.rand(2, 3)
        self.assertIsInstance(x, torch.Tensor)
        self.assertEqual(tuple(x.shape), (2, 3))

        # Values should lie in [0, 1).
        self.assertTrue(torch.all(x >= 0.0).item())
        self.assertTrue(torch.all(x < 1.0).item())

    def test_random(self):
        torch.manual_seed(0)

        # Scalar case.
        x0 = rnd.random()
        self.assertIsInstance(x0, torch.Tensor)
        self.assertEqual(x0.ndim, 0)
        self.assertGreaterEqual(float(x0), 0.0)
        self.assertLess(float(x0), 1.0)

        # Tuple size.
        x = rnd.random((2, 4))
        self.assertEqual(tuple(x.shape), (2, 4))
        self.assertTrue(torch.all(x >= 0.0).item())
        self.assertTrue(torch.all(x < 1.0).item())

        # Empty dimension should be supported.
        x_empty = rnd.random((0,))
        self.assertEqual(tuple(x_empty.shape), (0,))

    def test_randn(self):
        torch.manual_seed(0)

        # Scalar case.
        x0 = rnd.randn()
        self.assertIsInstance(x0, torch.Tensor)
        self.assertEqual(x0.ndim, 0)

        # Shape case.
        x = rnd.randn(2, 4)
        self.assertEqual(tuple(x.shape), (2, 4))

        # Should contain finite values.
        self.assertTrue(torch.isfinite(x).all().item())

    def test_beta_tensor_parameters(self):
        torch.manual_seed(0)

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([5.0, 7.0])

        # size=None -> broadcasted parameter shape.
        x = rnd.beta(a, b)
        self.assertEqual(tuple(x.shape), (2,))
        self.assertTrue(torch.all(x >= 0.0).item())
        self.assertTrue(torch.all(x <= 1.0).item())

        # size prepends sample shape: size + batch_shape.
        y = rnd.beta(a, b, (4,))
        self.assertEqual(tuple(y.shape), (4, 2))
        self.assertTrue(torch.all(y >= 0.0).item())
        self.assertTrue(torch.all(y <= 1.0).item())

    def test_binomial(self):
        torch.manual_seed(0)

        # Scalar case.
        x0 = rnd.binomial(10, 0.5)
        self.assertIsInstance(x0, torch.Tensor)
        self.assertEqual(x0.ndim, 0)
        self.assertEqual(x0.dtype, torch.int64)
        self.assertGreaterEqual(int(x0), 0)
        self.assertLessEqual(int(x0), 10)

        # Explicit size.
        x = rnd.binomial(10, 0.5, (3, 4))
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertEqual(x.dtype, torch.int64)
        self.assertTrue(torch.all(x >= 0).item())
        self.assertTrue(torch.all(x <= 10).item())

        # Tensor parameters (broadcasted).
        n = torch.tensor([5, 10], dtype=torch.int64)
        p = torch.tensor([0.2, 0.8])

        x_tensor = rnd.binomial(n, p)
        self.assertEqual(tuple(x_tensor.shape), (2,))
        self.assertEqual(x_tensor.dtype, torch.int64)
        self.assertTrue(torch.all(x_tensor >= 0).item())
        self.assertTrue(torch.all(x_tensor <= n).item())

    def test_choice(self):
        torch.manual_seed(0)

        a = torch.tensor([10, 20, 30, 40])

        # Scalar case (tensor population).
        x0 = rnd.choice(a)
        self.assertEqual(x0.ndim, 0)
        self.assertIn(int(x0), a.tolist())

        # Shape case (tensor population).
        x = rnd.choice(a, (2, 3))
        self.assertEqual(tuple(x.shape), (2, 3))
        for val in x.flatten():
            self.assertIn(int(val), a.tolist())

        # With probabilities (tensor population).
        p_tensor = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x_prob = rnd.choice(a, (5,), p=p_tensor)
        self.assertTrue(torch.all(x_prob == 30).item())

        # Without replacement (tensor population).
        x_no_rep = rnd.choice(a, (4,), replace=False)
        self.assertEqual(len(torch.unique(x_no_rep)), 4)

        # Integer population parity: samples from range(a).
        y = rnd.choice(5, (20,))
        self.assertEqual(tuple(y.shape), (20,))
        self.assertTrue(torch.all(y >= 0).item())
        self.assertTrue(torch.all(y < 5).item())

        # Integer population with probabilities.
        p_int = torch.tensor([0.0, 0.0, 1.0, 0.0])
        y_prob = rnd.choice(4, (6,), p=p_int)
        self.assertTrue(torch.all(y_prob == 2).item())

    def test_multinomial(self):
        torch.manual_seed(0)

        p = torch.tensor([0.2, 0.8])

        # Single draw.
        x = rnd.multinomial(5, p)
        self.assertEqual(tuple(x.shape), (2,))
        self.assertEqual(x.dtype, torch.int64)
        self.assertEqual(int(x.sum()), 5)

        # Multiple draws.
        y = rnd.multinomial(5, p, (4,))
        self.assertEqual(tuple(y.shape), (4, 2))
        self.assertEqual(y.dtype, torch.int64)
        self.assertTrue(torch.all(y.sum(dim=1) == 5).item())

    def test_randint(self):
        torch.manual_seed(0)

        # Scalar case.
        x0 = rnd.randint(5)
        self.assertEqual(x0.ndim, 0)
        self.assertEqual(x0.dtype, torch.int64)
        self.assertGreaterEqual(int(x0), 0)
        self.assertLess(int(x0), 5)

        # Explicit bounds.
        x = rnd.randint(2, 10, (3, 4))
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertEqual(x.dtype, torch.int64)
        self.assertTrue(torch.all(x >= 2).item())
        self.assertTrue(torch.all(x < 10).item())

    def test_shuffle(self):
        torch.manual_seed(0)

        # 1D shuffle should be in-place and preserve all elements.
        x = torch.arange(10)
        original = x.clone()
        rnd.shuffle(x)

        self.assertEqual(tuple(x.shape), (10,))
        self.assertTrue(torch.all(torch.sort(x).values == original).item())
        self.assertFalse(torch.all(x == original).item())

        # 2D shuffle should permute rows (axis=0), preserving row contents.
        x2 = torch.arange(12).reshape(3, 4)
        original2 = x2.clone()
        rnd.shuffle(x2)

        self.assertEqual(tuple(x2.shape), (3, 4))
        self.assertTrue(
            torch.all(
                torch.sort(x2, dim=0).values
                == torch.sort(original2, dim=0).values
            ).item()
        )

    def test_permutation(self):
        torch.manual_seed(0)

        # Integer input should permute range(n).
        p = rnd.permutation(10)
        self.assertEqual(tuple(p.shape), (10,))
        self.assertEqual(len(torch.unique(p)), 10)
        self.assertTrue(torch.all(p >= 0).item())
        self.assertTrue(torch.all(p < 10).item())

        # Tensor input should return a permuted copy (not in-place).
        x = torch.arange(12).reshape(3, 4)
        original = x.clone()
        y = rnd.permutation(x)

        self.assertEqual(tuple(y.shape), (3, 4))
        self.assertTrue(
            torch.all(
                torch.sort(y[:, 0]).values == torch.sort(original[:, 0]).values
            ).item()
        )
        self.assertTrue(torch.all(x == original).item())

    def test_uniform(self):
        torch.manual_seed(0)

        # Scalar case.
        x0 = rnd.uniform(0.0, 1.0)
        self.assertEqual(x0.ndim, 0)
        self.assertGreaterEqual(float(x0), 0.0)
        self.assertLess(float(x0), 1.0)

        # Shape case.
        x = rnd.uniform(0.0, 1.0, (3, 4))
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertTrue(torch.all(x >= 0.0).item())
        self.assertTrue(torch.all(x < 1.0).item())

        # Tensor broadcasting (size=None): broadcast(low, high).
        low = torch.tensor([0.0, 1.0])
        high = torch.tensor([1.0, 2.0])
        y = rnd.uniform(low, high)
        self.assertEqual(tuple(y.shape), (2,))

        # Tensor broadcasting with size: size + broadcast(low, high).
        y2 = rnd.uniform(low, high, (3,))
        self.assertEqual(tuple(y2.shape), (3, 2))

    def test_normal(self):
        torch.manual_seed(0)

        # Scalar case.
        x0 = rnd.normal(0.0, 1.0)
        self.assertEqual(x0.ndim, 0)

        # Shape case.
        x = rnd.normal(0.0, 1.0, (3, 4))
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertTrue(torch.isfinite(x).all().item())

        # Tensor broadcasting (size=None): broadcast(loc, scale).
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 2.0])
        y = rnd.normal(loc, scale)
        self.assertEqual(tuple(y.shape), (2,))

        # Tensor broadcasting with size: size + broadcast(loc, scale).
        y2 = rnd.normal(loc, scale, (3,))
        self.assertEqual(tuple(y2.shape), (3, 2))

    def test_poisson(self):
        torch.manual_seed(0)

        # Scalar case.
        x0 = rnd.poisson(3.0)
        self.assertEqual(x0.ndim, 0)
        self.assertGreaterEqual(int(x0), 0)
        self.assertEqual(x0.dtype, torch.int64)

        # Shape case.
        x = rnd.poisson(3.0, (3, 4))
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertTrue(torch.all(x >= 0).item())
        self.assertEqual(x.dtype, torch.int64)

        # Tensor broadcasting (size=None): broadcast(lam).
        lam = torch.tensor([1.0, 4.0])
        y = rnd.poisson(lam)
        self.assertEqual(tuple(y.shape), (2,))
        self.assertEqual(y.dtype, torch.int64)

        # Tensor broadcasting with size: size + broadcast(lam).
        y2 = rnd.poisson(lam, (3,))
        self.assertEqual(tuple(y2.shape), (3, 2))
        self.assertEqual(y2.dtype, torch.int64)
