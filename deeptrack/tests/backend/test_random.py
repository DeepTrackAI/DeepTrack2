import unittest

import torch

from deeptrack.backend.array_api_compat_ext.torch import random


class TestRandom(unittest.TestCase):
  def test_rand(self):
    shapes = [(2, ) , (3, 4) ]
    dtypes = [torch.float32, torch.float64]
    devices = [torch.device("cpu"), "cpu"]
    for i in range(len(shapes)):
        shape = shapes[i]
        dtype = dtypes[i]
        device = devices[i]
        
        torch.manual_seed(1)
        expected = torch.rand(*shape, dtype=dtype, device=device)

        torch.manual_seed(1)
        generated = random.rand(*shape, dtype=dtype, device=device)
        
        self.assertEqual(generated.shape, expected.shape)
        self.assertEqual(generated.dtype, expected.dtype)
        self.assertEqual(generated.device, expected.device)
        self.assertTrue(torch.equal(generated, expected))
