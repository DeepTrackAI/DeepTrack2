# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

import unittest

import numpy as np
import torch

import deeptrack as dt
from deeptrack.pytorch.data import Dataset


class TestPyTorchData(unittest.TestCase):

    def test_Dataset_len_from_length(self):
        pipeline = dt.Value(value=np.ones((2, 3), dtype=np.float32))
        ds = Dataset(pipeline, length=5)
        self.assertEqual(len(ds), 5)

    def test_Dataset_len_from_inputs(self):
        pipeline = dt.Value(value=np.ones((2, 3), dtype=np.float32))
        inputs = [[], [], []]
        ds = Dataset(pipeline, inputs=inputs)
        self.assertEqual(len(ds), 3)

    def test_Dataset_requires_inputs_or_length(self):
        pipeline = dt.Value(value=np.ones((2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            Dataset(pipeline)

    def test_Dataset_getitem_returns_tuple_of_tensors(self):
        pipeline = dt.Value(value=np.ones((2, 3), dtype=np.float32))
        ds = Dataset(pipeline, length=2)
        out = ds[0]

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], torch.Tensor)
        self.assertEqual(tuple(out[0].shape), (2, 3))

    def test_Dataset_caches_when_replace_false(self):
        pipeline = dt.Value(
            value=lambda: np.random.rand(2, 3).astype(np.float32)
        )
        ds = Dataset(pipeline, length=1, replace=False)

        out1 = ds[0][0].clone()
        out2 = ds[0][0].clone()

        self.assertTrue(torch.equal(out1, out2))

    def test_Dataset_replaces_when_replace_true(self):
        pipeline = dt.Value(
            value=lambda: np.random.rand(2, 3).astype(np.float32)
        )
        ds = Dataset(pipeline, length=1, replace=True)

        out1 = ds[0][0].clone()
        out2 = ds[0][0].clone()

        self.assertFalse(torch.equal(out1, out2))

    def test_Dataset_replace_probability_zero_and_one(self):
        pipeline = dt.Value(
            value=lambda: np.random.rand(2, 3).astype(np.float32)
        )

        ds0 = Dataset(pipeline, length=1, replace=0.0)
        a1 = ds0[0][0].clone()
        a2 = ds0[0][0].clone()
        self.assertTrue(torch.equal(a1, a2))

        ds1 = Dataset(pipeline, length=1, replace=1.0)
        b1 = ds1[0][0].clone()
        b2 = ds1[0][0].clone()
        self.assertFalse(torch.equal(b1, b2))

    def test_Dataset_replace_callable_no_args(self):
        pipeline = dt.Value(
            value=lambda: np.random.rand(2, 3).astype(np.float32)
        )
        ds = Dataset(pipeline, length=1, replace=lambda: True)

        out1 = ds[0][0].clone()
        out2 = ds[0][0].clone()

        self.assertFalse(torch.equal(out1, out2))

    def test_Dataset_replace_callable_with_index(self):
        pipeline = dt.Value(
            value=lambda: np.random.rand(2, 3).astype(np.float32)
        )

        def replace_fn(index):
            return index == 0

        ds = Dataset(pipeline, length=2, replace=replace_fn)

        out1 = ds[0][0].clone()
        out2 = ds[0][0].clone()
        self.assertFalse(torch.equal(out1, out2))

        out3 = ds[1][0].clone()
        out4 = ds[1][0].clone()
        self.assertTrue(torch.equal(out3, out4))

    def test_Dataset_as_tensor_negative_stride_numpy(self):
        pipeline = dt.Value(value=np.arange(12).reshape(3, 4)[:, ::-1])
        ds = Dataset(pipeline, length=1)

        out = ds[0][0]
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_Dataset_as_tensor_permute_numpy_ndim_gt_2(self):
        x = np.zeros((10, 11, 3), dtype=np.float32)
        pipeline = dt.Value(value=x)
        ds = Dataset(pipeline, length=1)

        out = ds[0][0]
        self.assertEqual(tuple(out.shape), (3, 10, 11))

    def test_Dataset_as_tensor_no_permute_for_uint(self):
        x = np.zeros((10, 11, 3), dtype=np.uint8)
        pipeline = dt.Value(value=x)
        ds = Dataset(pipeline, length=1)

        out = ds[0][0]
        self.assertEqual(tuple(out.shape), (10, 11, 3))

    def test_Dataset_float_dtype_cast_default(self):
        pipeline = dt.Value(value=np.ones((2, 2), dtype=np.float32))
        ds = Dataset(pipeline, length=1, float_dtype="default")

        out = ds[0][0]
        self.assertTrue(out.is_floating_point())
        self.assertEqual(out.dtype, torch.get_default_dtype())

    def test_Dataset_float_dtype_cast_explicit(self):
        pipeline = dt.Value(value=np.ones((2, 2), dtype=np.float32))
        ds = Dataset(pipeline, length=1, float_dtype=torch.float64)

        out = ds[0][0]
        self.assertEqual(out.dtype, torch.float64)

    def test_Dataset_int_casts_to_long(self):
        pipeline = dt.Value(value=np.ones((2, 2), dtype=np.int32))
        ds = Dataset(pipeline, length=1)

        out = ds[0][0]
        self.assertEqual(out.dtype, torch.long)

    def test_Dataset_replace_invalid_raises(self):
        pipeline = dt.Value(value=np.ones((2, 2), dtype=np.float32))
        ds = Dataset(pipeline, length=1, replace="nope")

        _ = ds[0]  # First call populates cache; replace is not validated yet.

        with self.assertRaises(TypeError):
            _ = ds[0]  # Second call evaluates replace and should raise.