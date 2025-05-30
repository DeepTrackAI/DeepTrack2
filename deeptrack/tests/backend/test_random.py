import unittest

import numpy as np
from deeptrack.backend.array_api_compat_ext.torch import random


class TestRandomNumpy(unittest.TestCase):
    def test_rand(self):
        shapes = [(2, ), (3, 4)]
        dtypes = [torch.float32, torch.float64]
        devices = [torch.device("cpu"), "cpu"]

        for shape, dtype, device in zip(shapes, dtypes, devices):

            expected = np.random.rand(*shape)
            generated = rand(*shape, dtype=dtype, device=device)
            self.assertEqual(generated.shape, expected.shape)
            self.assertEqual(generated.dtype, dtype)

        a = rand(100, dtype=torch.float32, device="cpu")
        b = np.random.rand(100)
        self.assertAlmostEqual(a.mean(), np.mean(b), delta = 1)
