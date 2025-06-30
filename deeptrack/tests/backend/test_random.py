import unittest

import numpy as np
import torch
from deeptrack.backend.array_api_compat_ext.torch import random
"""
TODO: Implement tests for all of these functions to start with.
    "rand",
    "random",
    "random_sample",
    "randn",
    "beta",
    "binomial",
    "choice",
    "multinomial",
    "randint",
    "shuffle",
    "uniform",
    "normal",
    "poisson",

"""

class TestRandom(unittest.TestCase):
    def test_rand(self):
        shapes = [(2, ), (3, 4)]
        dtypes = [torch.float32, torch.float64]
        devices = [torch.device("cpu"), "cpu"]

        for shape, dtype, device in zip(shapes, dtypes, devices):

            expected = np.random.rand(*shape)
            generated = random.rand(*shape, dtype=dtype, device=device)
            self.assertEqual(generated.shape, expected.shape)
            self.assertEqual(generated.dtype, dtype)

        a = random.rand(100, dtype=torch.float32, device="cpu")
        b = np.random.rand(100)
        self.assertAlmostEqual(a.mean(), np.mean(b), delta=1)  # Use a different rand
