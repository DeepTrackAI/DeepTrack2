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

        # Scalar case
        x0 = rnd.random()
        self.assertIsInstance(x0, torch.Tensor)
        self.assertEqual(x0.ndim, 0)
        self.assertGreaterEqual(float(x0), 0.0)
        self.assertLess(float(x0), 1.0)

        # Tuple size
        x = rnd.random((2, 4))
        self.assertEqual(tuple(x.shape), (2, 4))

        self.assertTrue(torch.all(x >= 0.0).item())
        self.assertTrue(torch.all(x < 1.0).item())

        # Empty dimension
        x_empty = rnd.random((0,))
        self.assertEqual(tuple(x_empty.shape), (0,))

    def test_randn(self):
        torch.manual_seed(0)

        # Scalar case
        x0 = rnd.randn()
        self.assertIsInstance(x0, torch.Tensor)
        self.assertEqual(x0.ndim, 0)

        # Shape case
        x = rnd.randn(2, 4)
        self.assertEqual(tuple(x.shape), (2, 4))

        # Should contain finite values
        self.assertTrue(torch.isfinite(x).all().item())
