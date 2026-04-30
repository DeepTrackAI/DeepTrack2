# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import unittest

import numpy as np
import torch

from deeptrack.pytorch import features


class TestTorchFeatures(unittest.TestCase):

    def test_ToTensor_numpy(self):
        f = features.ToTensor()
        x = np.ones((4, 5), dtype=np.float32)
        y = f(x)

        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(tuple(y.shape), (4, 5))

    def test_ToTensor_torch_tensor_passthrough(self):
        f = features.ToTensor()
        x = torch.ones((4, 5), dtype=torch.float32)
        y = f(x)

        self.assertIsInstance(y, torch.Tensor)
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(x.dtype, y.dtype)

    def test_ToTensor_numpy_negative_stride(self):
        f = features.ToTensor()
        x = np.arange(12).reshape(3, 4)[:, ::-1]
        y = f(x)

        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(tuple(y.shape), (3, 4))

    def test_ToTensor_scalar_add_dim(self):
        f = features.ToTensor(add_dim_to_number=True)
        y = f(3.0)

        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(tuple(y.shape), (1,))

    def test_ToTensor_scalar_no_add_dim(self):
        f = features.ToTensor(add_dim_to_number=False)
        y = f(3.0)

        self.assertIsInstance(y, float)

    def test_ToTensor_permute_always(self):
        f = features.ToTensor(permute_mode="always")
        x = np.zeros((10, 11, 3), dtype=np.float32)
        y = f(x)

        self.assertEqual(tuple(y.shape), (3, 10, 11))

    def test_ToTensor_permute_never(self):
        f = features.ToTensor(permute_mode="never")
        x = np.zeros((10, 11, 3), dtype=np.float32)
        y = f(x)

        self.assertEqual(tuple(y.shape), (10, 11, 3))

    def test_ToTensor_permute_numpy_only(self):
        f = features.ToTensor(permute_mode="numpy")
        x_np = np.zeros((10, 11, 3), dtype=np.float32)
        y_np = f(x_np)

        x_torch = torch.zeros((10, 11, 3), dtype=torch.float32)
        y_torch = f(x_torch)

        self.assertEqual(tuple(y_np.shape), (3, 10, 11))
        self.assertEqual(tuple(y_torch.shape), (10, 11, 3))

    def test_ToTensor_permute_numpy_and_not_int(self):
        f = features.ToTensor(permute_mode="numpy_and_not_int")

        x_float = np.zeros((10, 11, 3), dtype=np.float32)
        y_float = f(x_float)
        self.assertEqual(tuple(y_float.shape), (3, 10, 11))

        x_int = np.zeros((10, 11, 3), dtype=np.int32)
        y_int = f(x_int)
        self.assertEqual(tuple(y_int.shape), (10, 11, 3))

    def test_ToTensor_dtype(self):
        f = features.ToTensor(dtype=torch.float64)
        x = np.ones((2, 2), dtype=np.float32)
        y = f(x)

        self.assertEqual(y.dtype, torch.float64)
