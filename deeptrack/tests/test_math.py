# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import array_api_compat as apc
import numpy as np
from scipy.ndimage import uniform_filter

from deeptrack import math
from deeptrack.backend import OPENCV_AVAILABLE, TORCH_AVAILABLE, xp
from deeptrack.tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch


class TestMath_Numpy(BackendTestBase):
    BACKEND = "numpy"

    @property
    def array_type(self):
        if self.BACKEND == "numpy":
            return np.ndarray
        elif self.BACKEND == "torch":
            return torch.Tensor
        else:
            raise ValueError(f"Unsupported backend: {self.BACKEND}")

    def test_Average(self):
        input_image0 = xp.ones((10, 30, 20)) * 2
        input_image1 = xp.ones((10, 30, 20)) * 4
        feature = math.Average(axis=0)
        average = feature.resolve([input_image0, input_image1])

        self.assertIsInstance(average, self.array_type)
        self.assertTrue(xp.all(average == 3), True)
        self.assertEqual(average.shape, (10, 30, 20))


    def test_Clip(self):
        input_image = xp.asarray([[10, 4], [4, -10]])
        feature = math.Clip(min=-5, max=5)
        clipped_feature = feature.resolve(input_image)

        self.assertIsInstance(clipped_feature, self.array_type)
        self.assertTrue(
            xp.all(clipped_feature == xp.asarray([[5, 4], [4, -5]]))
        )

        input_image = xp.asarray(np.array([[5, 6], [7, 8]]))
        feature = math.Clip(min=0, max=10)
        clipped_feature = feature.resolve(input_image)

        self.assertIsInstance(clipped_feature, self.array_type)
        self.assertTrue(
            xp.all(clipped_feature == xp.asarray([[5, 6], [7, 8]]))
        )


    def test_NormalizeMinMax(self):
        input_image = xp.asarray([[10, 4], [4, -10]])
        feature = math.NormalizeMinMax(min=-5, max=5)
        normalized_image = feature.resolve(input_image, featurewise=False)
        self.assertIsInstance(normalized_image, self.array_type)
        self.assertTrue(
            xp.all(normalized_image == xp.asarray([[5, 2], [2, -5]]))
        )

        x = xp.asarray(
            [
                [[0.0, 10.0], [5.0, 20.0]],
                [[10.0, 30.0], [20.0, 40.0]],
            ]
        )
        out_featurewise = math.NormalizeMinMax(
            min=0,
            max=1,
            featurewise=True,
        ).resolve(x)

        zero = xp.asarray(0.0)
        one = xp.asarray(1.0)

        # featurewise
        self.assertIsInstance(out_featurewise, self.array_type)
        self.assertTrue(xp.allclose(xp.min(out_featurewise[..., 0]), zero))
        self.assertTrue(xp.allclose(xp.max(out_featurewise[..., 0]), one))
        self.assertTrue(xp.allclose(xp.min(out_featurewise[..., 1]), zero))
        self.assertTrue(xp.allclose(xp.max(out_featurewise[..., 1]), one))

        out_global = math.NormalizeMinMax(
            min=0,
            max=1,
            featurewise=False,
        ).resolve(x)
        # global
        self.assertIsInstance(out_global, self.array_type)
        self.assertTrue(xp.allclose(xp.min(out_global), zero))
        self.assertTrue(xp.allclose(xp.max(out_global), one))

        # channel_axis
        x = xp.asarray(
            [
                [[0.0, 10.0], [5.0, 20.0]],
                [[10.0, 30.0], [20.0, 40.0]],
            ]
        )  # shape (2,2,2)
        # move channels to axis 0
        x = xp.moveaxis(x, -1, 0)  # shape (2,2,2)
        out = math.NormalizeMinMax(
            featurewise=True,
            channel_axis=0,
        ).resolve(x)
        zero = xp.asarray(0.0)
        one = xp.asarray(1.0)
        self.assertTrue(xp.allclose(xp.min(out[0]), zero))
        self.assertTrue(xp.allclose(xp.max(out[0]), one))
        self.assertTrue(xp.allclose(xp.min(out[1]), zero))
        self.assertTrue(xp.allclose(xp.max(out[1]), one))

    def test_NormalizeStandard(self):
        # --- basic correctness ---
        x = xp.asarray([[1, 2], [3, 4]], dtype=float)
        out = math.NormalizeStandard().resolve(x)

        self.assertIsInstance(out, self.array_type)
        self.assertTrue(xp.allclose(xp.mean(out), xp.asarray(0.0, dtype=out.dtype)))

        if apc.is_torch_array(out):
            self.assertTrue(torch.allclose(
                torch.std(out, unbiased=False),
                torch.tensor(1.0, dtype=out.dtype),
            ))
        else:
            self.assertTrue(xp.allclose(xp.std(out), xp.asarray(1.0)))

        # --- shape preservation ---
        self.assertEqual(out.shape, x.shape)

        # --- constant input (numerical stability) ---
        x = xp.ones((4, 4))
        out = math.NormalizeStandard().resolve(x)
        self.assertTrue(xp.all(xp.isfinite(out)))

        # --- featurewise with channel axis ---
        x = xp.asarray(
            [
                [[1, 10], [2, 20]],
                [[3, 30], [4, 40]],
            ],
            dtype=float,
        )  # shape (2,2,2)

        out = math.NormalizeStandard(
            featurewise=True,
            channel_axis=-1,
        ).resolve(x)

        zero = xp.asarray(0.0, dtype=out.dtype)
        one = xp.asarray(1.0, dtype=out.dtype)

        self.assertTrue(xp.allclose(xp.mean(out[..., 0]), zero))
        self.assertTrue(xp.allclose(xp.mean(out[..., 1]), zero))

        if apc.is_torch_array(out):
            self.assertTrue(torch.allclose(
                torch.std(out[..., 0], unbiased=False),
                torch.tensor(1.0, dtype=out.dtype),
            ))
        else:
            self.assertTrue(xp.allclose(xp.std(out[..., 0]), one))
            self.assertTrue(xp.allclose(xp.std(out[..., 1]), one))

        # --- global vs featurewise difference ---
        out_global = math.NormalizeStandard(
            featurewise=False,
        ).resolve(x)

        self.assertFalse(xp.allclose(out, out_global))

    def test_NormalizeQuantile(self):
        # --- basic correctness ---
        x = xp.asarray([[1, 2], [3, 100]], dtype=float)
        out = math.NormalizeQuantile(quantiles=(0.25, 0.75)).resolve(x)

        self.assertIsInstance(out, self.array_type)

        # median -> 0
        self.assertTrue(xp.allclose(
            xp.quantile(out, 0.5),
            xp.asarray(0.0, dtype=out.dtype),
            atol=1e-5,
        ))

        # --- shape preservation ---
        self.assertEqual(out.shape, x.shape)

        # --- scale normalization ---
        q_low = xp.quantile(out, 0.25)
        q_high = xp.quantile(out, 0.75)
        self.assertTrue(q_high > q_low)

        # --- constant input (numerical stability) ---
        x = xp.ones((4, 4))
        out = math.NormalizeQuantile().resolve(x)
        self.assertTrue(xp.all(xp.isfinite(out)))

        # --- featurewise behavior ---
        x = xp.asarray(
            [
                [[1, 10], [2, 20]],
                [[3, 30], [4, 40]],
            ],
            dtype=float,
        )

        out = math.NormalizeQuantile(
            featurewise=True,
            channel_axis=-1,
        ).resolve(x)

        self.assertTrue(xp.allclose(
            xp.quantile(out[..., 0], 0.5),
            xp.asarray(0.0, dtype=out.dtype),
            atol=1e-5,
        ))
        self.assertTrue(xp.allclose(
            xp.quantile(out[..., 1], 0.5),
            xp.asarray(0.0, dtype=out.dtype),
            atol=1e-5,
        ))

        # --- global vs featurewise difference ---
        out_global = math.NormalizeQuantile(
            featurewise=False,
        ).resolve(x)

        self.assertFalse(xp.allclose(out, out_global))


    def test_Blur(self):
        blur = math.Blur()
        with self.assertRaises(NotImplementedError):
            blur.resolve(xp.zeros((2,2)))

        class DummyBlur(math.Blur):
            def _get_numpy(self, image, **kwargs):
                return image + 1
            def _get_torch(self, image, **kwargs):
                return image + 1

        image = xp.zeros((2,2))
        out = DummyBlur().resolve(image)
        self.assertIsInstance(out, self.array_type)
        self.assertTrue(xp.all(out == 1))

    def test_GaussianBlur(self):

        #  --- impulse response ---
        impulse = xp.zeros((7, 7))
        impulse[3, 3] = 1
        out = math.GaussianBlur(sigma=1).resolve(impulse)
        # symmetry
        self.assertTrue(xp.allclose(out, xp.flip(out, axis=0)))
        self.assertTrue(xp.allclose(out, xp.flip(out, axis=1)))
        # normalization
        tol = 1e-5 if self.BACKEND == "numpy" else 2e-2 # reflect padding works differently in torch vs scipy
        self.assertTrue(xp.allclose(xp.sum(out), xp.asarray(1.0, dtype=out.dtype), atol=tol))
        # center is maximum
        self.assertTrue(out[3, 3] == xp.max(out))

        # --- sigma = 0 (identity) ---
        input_image = xp.asarray([[1, 2], [3, 4]], dtype=float)
        feature = math.GaussianBlur(sigma=0)
        blurred_image = feature.resolve(input_image)
        self.assertTrue(xp.all(blurred_image == xp.asarray([[1, 2], [3, 4]])))
        self.assertIsInstance(blurred_image, self.array_type)

        # --- sigma → large (mean image) ---
        input_image = xp.asarray([[1, 2], [3, 4]], dtype=float)
        feature = math.GaussianBlur(sigma=1000)
        blurred_image = feature.resolve(input_image)
        mean_val = xp.mean(input_image)
        self.assertTrue(
            xp.allclose(
                blurred_image,
                xp.full_like(input_image, mean_val),
                atol=1e-2,
            )
        )
        # --- sigma small ---
        feature = math.GaussianBlur(sigma=2)
        blurred = feature.resolve(input_image)

        # variance decreases
        self.assertTrue(xp.var(blurred) < xp.var(input_image))
        # values move toward mean
        mean_val = xp.mean(input_image)
        self.assertTrue(
            xp.all(
                xp.abs(blurred - mean_val)
                <= xp.abs(input_image - mean_val) + 1e-6
            )
        )
        # sum approximately preserved
        self.assertTrue(xp.allclose(xp.sum(blurred), xp.sum(input_image), atol=1e-5))
        # no new extrema
        self.assertTrue(xp.min(blurred) >= xp.min(input_image))
        self.assertTrue(xp.max(blurred) <= xp.max(input_image))

    def test_MedianBlur(self):
        # --- ksize = 1 (identity) ---
        image = xp.asarray([[1, 2], [3, 4]], dtype=float)
        out = math.MedianBlur(ksize=1).resolve(image)
        self.assertTrue(xp.all(out == image))

        # --- ksize = 3 (removes outliers) ---
        image = xp.asarray([
            [1, 100, 1],
            [1, 1,   1],
            [1, 1,   1],
        ], dtype=float)
        out = math.MedianBlur(ksize=3).resolve(image)
        # central pixel should remove outlier
        self.assertEqual(float(out[1, 1]), 1.0)

        #--- ksize = 3 (preserves edges) ---
        image = xp.zeros((16, 16))
        out = math.MedianBlur(ksize=3).resolve(image)
        self.assertEqual(out.shape, image.shape)

        #--- ksize = 3 (multi-channel) ---
        image = xp.asarray([[1, 2], [3, 4]], dtype=float)
        out = math.MedianBlur(ksize=3).resolve(image)
        self.assertEqual(out.dtype, image.dtype)

        #--- ksize = 3 (no new extrema) ---
        image = xp.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=float)
        out = math.MedianBlur(ksize=3).resolve(image)
        self.assertTrue(xp.min(out) >= xp.min(image))
        self.assertTrue(xp.max(out) <= xp.max(image))

        # --- ksize = 3 (impulse response) ---
        image = xp.zeros((5, 5))
        image[2, 2] = 100  # impulse noise
        out = math.MedianBlur(ksize=3).resolve(image)
        # impulse should be removed
        self.assertTrue(float(out[2, 2]) < 100)

        # --- ksize = 3 (multi-channel) ---
        image = xp.zeros((8, 8, 3))
        image[..., 0] = 1
        image[..., 1] = 2
        image[..., 2] = 3
        out = math.MedianBlur(ksize=3).resolve(image)
        self.assertEqual(out.shape, image.shape)
        # channels processed independently
        self.assertTrue(xp.all(out[..., 0] == 1))
        self.assertTrue(xp.all(out[..., 1] == 2))
        self.assertTrue(xp.all(out[..., 2] == 3))

        #--- ksize = 3 (3D) ---
        image = xp.zeros((8, 8, 8))
        image[4, 4, 4] = 10
        out = math.MedianBlur(ksize=3).resolve(image)
        self.assertEqual(out.shape, image.shape)
        # spike should be reduced
        self.assertTrue(float(out[4, 4, 4]) <= 10)

        # --- ksize even (should raise error) ---
        with self.assertRaises(ValueError):
            math.MedianBlur(ksize=4).resolve(xp.zeros((5, 5)))
        feature = math.MedianBlur(ksize=lambda: 3)
        image = xp.zeros((5, 5))
        out = feature.resolve(image)
        self.assertEqual(out.shape, image.shape)

    def test_Pool(self):
        class Dummy_Pool(math.Pool):
            def _get_numpy(self, image, **kwargs): return image
            def _get_torch(self, image, **kwargs): return image

        p = Dummy_Pool(ksize=(2, 3, 4))
        
        # --- pool size logic ---
        image = xp.zeros((10, 10))
        self.assertEqual(p._get_pool_size(image), (2, 3, 1))

        image = xp.zeros((10, 10, 3))  # channels-last
        self.assertEqual(p._get_pool_size(image), (2, 3, 1))

        image = xp.zeros((8, 10, 12))  # 3D
        self.assertEqual(p._get_pool_size(image), (2, 3, 4))

        # --- cropping behavior ---
        image = xp.zeros((9, 10))
        cropped = p._crop_center(image)
        self.assertEqual(cropped.shape, (8, 9))

        image = xp.zeros((9, 10, 3))
        cropped = p._crop_center(image)
        self.assertEqual(cropped.shape, (8, 9, 3))

        image = xp.zeros((7, 9, 10))
        cropped = p._crop_center(image)
        self.assertEqual(cropped.shape, (6, 9, 8))

    def test_AveragePooling(self):

        # --- basic 2D pooling ---
        image = xp.asarray([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], dtype=float)

        out = math.AveragePooling(ksize=2).resolve(image)
        expected = xp.asarray([[3.5, 5.5]], dtype=image.dtype)
        self.assertTrue(xp.allclose(out, expected))
        self.assertEqual(out.shape, (1, 2))

        # --- shape reduction ---
        image = xp.zeros((8, 8))
        out = math.AveragePooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (4, 4))

        # --- cropping (non-divisible size) ---
        image = xp.ones((5, 5))
        out = math.AveragePooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2))  # cropped to 4x4 → pooled

        # --- multi-channel (no mixing) ---
        image = xp.zeros((4, 4, 3))
        image[..., 0] = 1
        image[..., 1] = 2
        image[..., 2] = 3

        out = math.AveragePooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertTrue(xp.all(out[..., 0] == 1))
        self.assertTrue(xp.all(out[..., 1] == 2))
        self.assertTrue(xp.all(out[..., 2] == 3))

        # --- 3D pooling (true volume) ---
        image = xp.ones((4, 4, 8))
        out = math.AveragePooling(ksize=(2, 2, 2)).resolve(image)
        self.assertEqual(out.shape, (2, 2, 4))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- channels (no z pooling) ---
        image = xp.ones((4, 4, 3))
        out = math.AveragePooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- z ignored when treated as channels ---
        image = xp.ones((4, 4, 3))
        out = math.AveragePooling(ksize=(2, 2, 2)).resolve(image)
        self.assertEqual(out.shape, (2, 2, 3))

        # --- value correctness ---
        image = xp.asarray([
            [0, 0],
            [0, 4],
        ], dtype=float)

        out = math.AveragePooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(1.0, dtype=out.dtype)))

        # --- dtype preserved ---
        image = xp.asarray([[1, 2], [3, 4]], dtype=float)
        out = math.AveragePooling(ksize=2).resolve(image)
        self.assertEqual(out.dtype, image.dtype)

        # --- ksize = 1 (identity) ---
        image = xp.random.rand(4, 4)
        out = math.AveragePooling(ksize=1).resolve(image)
        self.assertTrue(xp.allclose(out, image))
        self.assertEqual(out.shape, image.shape)

        # --- singleton channel dimension ---
        image = xp.ones((4, 4, 1))
        out = math.AveragePooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2, 1))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- anisotropic pooling ---
        image = xp.arange(16, dtype=float).reshape(4, 4)
        out = math.AveragePooling(ksize=(2, 1)).resolve(image)
        self.assertEqual(out.shape, (2, 4))
        expected = image.reshape(2, 2, 4).mean(axis=1)
        self.assertTrue(xp.allclose(out, expected))

        # --- random input vs reference ---
        image = xp.random.rand(10, 10)
        k = 2
        out = math.AveragePooling(ksize=k).resolve(image)

        # reference (numpy-style reshape)
        ref = image[:10 - 10 % k, :10 - 10 % k]
        ref = ref.reshape(10 // k, k, 10 // k, k).mean(axis=(1, 3))

        self.assertTrue(xp.allclose(out, ref))

        # --- axis correctness (critical) ---
        image = xp.zeros((4, 4, 6), dtype=float)

        # encode variation ONLY along z
        for i in range(6):
            image[:, :, i] = float(i)

        out = math.AveragePooling(ksize=(2, 2, 3)).resolve(image)

        # if z is pooled:
        # blocks [0,1,2] → mean = 1.0
        # blocks [3,4,5] → mean = 4.0
        expected = xp.asarray([
            [[1.0, 4.0],
            [1.0, 4.0]],
            [[1.0, 4.0],
            [1.0, 4.0]],
        ], dtype=out.dtype)
        self.assertTrue(xp.allclose(out, expected))

    def test_MaxPooling(self):

        # --- basic 2D pooling ---
        image = xp.asarray([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], dtype=float)

        out = math.MaxPooling(ksize=2).resolve(image)
        expected = xp.asarray([[6, 8]], dtype=image.dtype)
        self.assertTrue(xp.allclose(out, expected))
        self.assertEqual(out.shape, (1, 2))

        # --- shape reduction ---
        image = xp.zeros((8, 8))
        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (4, 4))

        # --- cropping (non-divisible size) ---
        image = xp.ones((5, 5))
        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2))

        # --- multi-channel (no mixing) ---
        image = xp.zeros((4, 4, 3))
        image[..., 0] = 1
        image[..., 1] = 2
        image[..., 2] = 3

        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertTrue(xp.all(out[..., 0] == 1))
        self.assertTrue(xp.all(out[..., 1] == 2))
        self.assertTrue(xp.all(out[..., 2] == 3))

        # --- 3D pooling (true volume) ---
        image = xp.ones((4, 4, 8))
        out = math.MaxPooling(ksize=(2, 2, 2)).resolve(image)
        self.assertEqual(out.shape, (2, 2, 4))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- channels (no z pooling) ---
        image = xp.ones((4, 4, 3))
        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- z ignored when treated as channels ---
        image = xp.ones((4, 4, 3))
        out = math.MaxPooling(ksize=(2, 2, 2)).resolve(image)
        self.assertEqual(out.shape, (2, 2, 3))

        # --- value correctness ---
        image = xp.asarray([
            [0, 0],
            [0, 4],
        ], dtype=float)

        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(4.0, dtype=out.dtype)))

        # --- distinct values (critical) ---
        image = xp.asarray([
            [1, 2],
            [3, 100],
        ], dtype=float)

        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(100.0, dtype=out.dtype)))

        # --- dtype preserved ---
        image = xp.asarray([[1, 2], [3, 4]], dtype=float)
        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertEqual(out.dtype, image.dtype)

        # --- ksize = 1 (identity) ---
        image = xp.random.rand(4, 4)
        out = math.MaxPooling(ksize=1).resolve(image)
        self.assertTrue(xp.allclose(out, image))
        self.assertEqual(out.shape, image.shape)

        # --- singleton channel ---
        image = xp.ones((4, 4, 1))
        out = math.MaxPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2, 2, 1))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- anisotropic pooling ---
        image = xp.arange(16, dtype=float).reshape(4, 4)
        out = math.MaxPooling(ksize=(2, 1)).resolve(image)
        self.assertEqual(out.shape, (2, 4))

        # --- random input vs reference ---
        image = xp.random.rand(10, 10)
        k = 2
        out = math.MaxPooling(ksize=k).resolve(image)

        ref_np = np.asarray(image)
        ref_np = ref_np[:10 - 10 % k, :10 - 10 % k]
        ref_np = ref_np.reshape(10//k, k, 10//k, k).max(axis=(1,3))

        self.assertTrue(xp.allclose(out, xp.asarray(ref_np)))

        # --- axis correctness (critical) ---
        image = xp.zeros((4, 4, 6))

        # encode axis identity
        for i in range(6):
            image[:, :, i] = i  # variation ONLY along z

        out = math.MaxPooling(ksize=(2, 2, 3)).resolve(image)

        # if z is pooled → values should change
        # if z is treated as channel → values preserved

        # expected if z is pooled:
        # blocks: [0,1,2] → 2 ; [3,4,5] → 5
        expected = xp.asarray([[[2, 5],
                                [2, 5]],
                            [[2, 5],
                                [2, 5]]], dtype=out.dtype)
        self.assertTrue(xp.allclose(out, expected))

    def test_MinPooling(self):

        # --- basic 2D pooling ---
        image = xp.asarray([[1,2,3,4],[5,6,7,8]], dtype=float)
        out = math.MinPooling(ksize=2).resolve(image)
        expected = xp.asarray([[1,3]], dtype=image.dtype)
        self.assertTrue(xp.allclose(out, expected))

        # --- multi-channel (no mixing) ---
        image = xp.zeros((4,4,3))
        image[...,0] = 1
        image[...,1] = 2
        image[...,2] = 3

        out = math.MinPooling(ksize=2).resolve(image)
        self.assertTrue(xp.all(out[...,0] == 1))
        self.assertTrue(xp.all(out[...,1] == 2))
        self.assertTrue(xp.all(out[...,2] == 3))

        # --- value correctness ---
        image = xp.asarray([[0,0],[0,4]], dtype=float)
        out = math.MinPooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(0.0, dtype=out.dtype)))

        # --- distinct values ---
        image = xp.asarray([[5,2],[3,100]], dtype=float)
        out = math.MinPooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(2.0, dtype=out.dtype)))

        # --- random vs reference ---
        image = xp.random.rand(10,10)
        k = 2
        out = math.MinPooling(ksize=k).resolve(image)

        ref_np = np.asarray(image)
        ref_np = ref_np[:10 - 10 % k, :10 - 10 % k]
        ref_np = ref_np.reshape(10//k, k, 10//k, k).min(axis=(1,3))

        self.assertTrue(xp.allclose(out, xp.asarray(ref_np)))

        # --- axis correctness ---
        image = xp.zeros((4,4,6))
        for i in range(6):
            image[:,:,i] = i

        out = math.MinPooling(ksize=(2,2,3)).resolve(image)

        expected = xp.asarray([[[0,3],[0,3]],
                            [[0,3],[0,3]]], dtype=out.dtype)

        self.assertTrue(xp.allclose(out, expected))

    def test_SumPooling(self):

        # --- basic 2D pooling ---
        image = xp.asarray([[1,2,3,4],[5,6,7,8]], dtype=float)
        out = math.SumPooling(ksize=2).resolve(image)
        expected = xp.asarray([[14,22]], dtype=image.dtype)
        self.assertTrue(xp.allclose(out, expected))
        self.assertEqual(out.shape, (1,2))

        # --- shape reduction ---
        image = xp.zeros((8,8))
        out = math.SumPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (4,4))

        # --- cropping ---
        image = xp.ones((5,5))
        out = math.SumPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2,2))

        # --- multi-channel ---
        image = xp.zeros((4,4,3))
        image[...,0] = 1
        image[...,1] = 2
        image[...,2] = 3

        out = math.SumPooling(ksize=2).resolve(image)
        self.assertTrue(xp.all(out[...,0] == 4))
        self.assertTrue(xp.all(out[...,1] == 8))
        self.assertTrue(xp.all(out[...,2] == 12))

        # --- 3D pooling ---
        image = xp.ones((4,4,6))
        out = math.SumPooling(ksize=(2,2,3)).resolve(image)
        self.assertEqual(out.shape, (2,2,2))
        self.assertTrue(xp.allclose(out, xp.asarray(12.0)))

        # --- channels ---
        image = xp.ones((4,4,3))
        out = math.SumPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2,2,3))
        self.assertTrue(xp.allclose(out, xp.asarray(4.0)))

        # --- value correctness ---
        image = xp.asarray([[0,0],[0,4]], dtype=float)
        out = math.SumPooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(4.0, dtype=out.dtype)))

        # --- dtype preserved ---
        image = xp.asarray([[1,2],[3,4]], dtype=float)
        out = math.SumPooling(ksize=2).resolve(image)
        self.assertEqual(out.dtype, image.dtype)

        # --- ksize = 1 ---
        image = xp.random.rand(4,4)
        out = math.SumPooling(ksize=1).resolve(image)
        self.assertTrue(xp.allclose(out, image))

        # --- anisotropic ---
        image = xp.arange(16, dtype=float).reshape(4,4)
        out = math.SumPooling(ksize=(2,1)).resolve(image)
        self.assertEqual(out.shape, (2,4))

        # --- random vs reference ---
        image = xp.random.rand(10,10)
        k = 2
        out = math.SumPooling(ksize=k).resolve(image)

        ref_np = np.asarray(image)
        ref_np = ref_np[:10 - 10 % k, :10 - 10 % k]
        ref_np = ref_np.reshape(10//k, k, 10//k, k).sum(axis=(1,3))

        self.assertTrue(xp.allclose(out, xp.asarray(ref_np, dtype=out.dtype)))

        # --- axis correctness ---
        image = xp.zeros((4,4,6))
        for i in range(6):
            image[:,:,i] = i

        out = math.SumPooling(ksize=(2,2,3)).resolve(image)

        expected = xp.asarray([[[12,48],[12,48]],
                            [[12,48],[12,48]]], dtype=out.dtype)
        self.assertTrue(xp.allclose(out, expected))

    def test_MedianPooling(self):
        
        # --- ksize = 2 (simple case) ---
        image = xp.asarray([
            [1, 2],
            [3, 4],
        ], dtype=float)
        out = math.MedianPooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(2.5, dtype=image.dtype)))

        # --- basic 2D pooling ---
        image = xp.asarray([[1,2,3,4],[5,6,7,8]], dtype=float)
        out = math.MedianPooling(ksize=2).resolve(image)
        expected = xp.asarray([[3.5,5.5]], dtype=image.dtype)
        self.assertTrue(xp.allclose(out, expected))
        self.assertEqual(out.shape, (1,2))

        # --- shape reduction ---
        image = xp.zeros((8,8))
        out = math.MedianPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (4,4))

        # --- cropping ---
        image = xp.ones((5,5))
        out = math.MedianPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2,2))

        # --- multi-channel ---
        image = xp.zeros((4,4,3))
        image[...,0] = 1
        image[...,1] = 2
        image[...,2] = 3

        out = math.MedianPooling(ksize=2).resolve(image)
        self.assertTrue(xp.all(out[...,0] == 1))
        self.assertTrue(xp.all(out[...,1] == 2))
        self.assertTrue(xp.all(out[...,2] == 3))

        # --- 3D pooling ---
        image = xp.ones((4,4,6))
        out = math.MedianPooling(ksize=(2,2,3)).resolve(image)
        self.assertEqual(out.shape, (2,2,2))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- channels ---
        image = xp.ones((4,4,3))
        out = math.MedianPooling(ksize=2).resolve(image)
        self.assertEqual(out.shape, (2,2,3))
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- value correctness ---
        image = xp.asarray([[0,0],[0,4]], dtype=float)
        out = math.MedianPooling(ksize=2).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(0.0, dtype=out.dtype)))

        # --- odd kernel ---
        image = xp.asarray([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
        out = math.MedianPooling(ksize=3).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(5.0, dtype=out.dtype)))

        # --- dtype preserved ---
        image = xp.asarray([[1,2],[3,4]], dtype=float)
        out = math.MedianPooling(ksize=2).resolve(image)
        self.assertEqual(out.dtype, image.dtype)

        # --- ksize = 1 ---
        image = xp.random.rand(4,4)
        out = math.MedianPooling(ksize=1).resolve(image)
        self.assertTrue(xp.allclose(out, image))

        # --- random vs reference ---
        image = xp.random.rand(10,10)
        k = 2
        out = math.MedianPooling(ksize=k).resolve(image)
        ref_np = np.asarray(image)
        ref_np = ref_np[:10 - 10 % k, :10 - 10 % k]
        ref_np = ref_np.reshape(10//k, k, 10//k, k)
        ref_np = ref_np.transpose(0, 2, 1, 3)   # (H',W',k,k)
        ref_np = ref_np.reshape(10//k, 10//k, -1)
        ref_np = np.median(ref_np, axis=-1)
        self.assertTrue(xp.allclose(out, xp.asarray(ref_np, dtype=out.dtype)))

        # --- axis correctness ---
        image = xp.zeros((4,4,6))
        for i in range(6):
            image[:,:,i] = i

        out = math.MedianPooling(ksize=(2,2,3)).resolve(image)

        expected = xp.asarray([[[1,4],[1,4]],
                            [[1,4],[1,4]]], dtype=out.dtype)

        self.assertTrue(xp.allclose(out, expected))

    
    def test_Resize(self):

        # --- ksize = 1 (identity) ---
        image = xp.asarray([
            [1, 2],
            [3, 4],
        ], dtype=float)

        out = math.Resize(dsize=(2, 2)).resolve(image)

        # identity case must be exact
        self.assertTrue(xp.allclose(out, image))

        # --- basic 2D ---
        image = xp.random.rand(16, 8)
        out = math.Resize(dsize=(4, 2)).resolve(image)
        self.assertEqual(out.shape, (2, 4))

        # --- channels (H,W,C) ---
        image = xp.zeros((16, 8, 3))
        image[..., 0] = 1
        image[..., 1] = 2
        image[..., 2] = 3

        out = math.Resize(dsize=(4, 2)).resolve(image)

        self.assertEqual(out.shape, (2, 4, 3))
        self.assertTrue(xp.allclose(out[..., 0], xp.asarray(1.0)))
        self.assertTrue(xp.allclose(out[..., 1], xp.asarray(2.0)))
        self.assertTrue(xp.allclose(out[..., 2], xp.asarray(3.0)))

        # --- volume (H,W,Z) ---
        image = xp.zeros((16, 8, 5))
        for i in range(5):
            image[..., i] = i

        out = math.Resize(dsize=(4, 2)).resolve(image)

        self.assertEqual(out.shape, (2, 4, 5))

        # slices must remain constant → detects axis errors
        for i in range(5):
            self.assertTrue(xp.allclose(out[..., i], xp.asarray(float(i))))

        # --- volume + channels (H,W,Z,C) ---
        image = xp.zeros((16, 8, 4, 3))
        for z in range(4):
            for c in range(3):
                image[..., z, c] = z + 10*c

        out = math.Resize(dsize=(4, 2)).resolve(image)

        self.assertEqual(out.shape, (2, 4, 4, 3))

        for z in range(4):
            for c in range(3):
                self.assertTrue(xp.allclose(out[..., z, c], xp.asarray(float(z + 10 * c))))

        # --- identity resize ---
        image = xp.random.rand(10, 12)
        out = math.Resize(dsize=(12, 10)).resolve(image)
        self.assertTrue(xp.allclose(out, image))

        # --- constant image invariance ---
        image = xp.ones((16, 16))
        out = math.Resize(dsize=(8, 8)).resolve(image)
        self.assertTrue(xp.allclose(out, xp.asarray(1.0)))

        # --- dtype preserved ---
        image = xp.random.rand(8, 8)
        image = xp.asarray(image, dtype=xp.float32)
        out = math.Resize(dsize=(4, 4)).resolve(image)
        self.assertEqual(out.dtype, image.dtype)

    def test_isotropic_dilation(self):
        mask = xp.asarray([[0, 1], [0, 0]], dtype=bool)
        out = math.isotropic_dilation(mask, radius=0, backend=self.BACKEND)
        self.assertTrue(xp.all(out == mask))

        mask = xp.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = math.isotropic_dilation(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(xp.sum(out) >= xp.sum(mask))

        mask = xp.random.rand(5, 5) > 0.5
        out = math.isotropic_dilation(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(xp.all((out == 0) | (out == 1)))

        mask = xp.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        out = math.isotropic_dilation(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(out[3, 3])
        self.assertTrue(xp.sum(out) > 1)

        mask = xp.ones((5, 5), dtype=bool)
        out = math.isotropic_dilation(mask, radius=2, backend=self.BACKEND)
        self.assertTrue(xp.all(out))
        
        mask = xp.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        out = math.isotropic_dilation(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(out[2, 2, 2])
        self.assertTrue(xp.sum(out) > 1)

        # activate one plane only
        mask = xp.zeros((5, 5, 5), dtype=bool)
        mask[2, :, :] = True
        out = math.isotropic_dilation(mask, radius=1, backend=self.BACKEND)
        # must expand along Z
        self.assertTrue(xp.sum(out[1]) > 0)
        self.assertTrue(xp.sum(out[3]) > 0)
        
        mask = xp.zeros((7, 7, 7))
        mask[3, 3, 3] = 1
        out = math.isotropic_dilation(mask, radius=1, backend=self.BACKEND)
        self.assertEqual(out.ndim, mask.ndim)
        self.assertGreater(xp.sum(out), 1)

        mask = xp.zeros((7, 7, 1))
        mask[3, 3, 0] = 1
        out = math.isotropic_dilation(mask, radius=1, backend=self.BACKEND)
        self.assertEqual(out.ndim, mask.ndim)

    def test_isotropic_erosion(self):
        mask = xp.asarray([[0, 1], [1, 1]], dtype=bool)
        out = math.isotropic_erosion(mask, radius=0, backend=self.BACKEND)
        self.assertTrue(xp.all(out == mask))

        mask = xp.ones((5, 5), dtype=bool)
        out = math.isotropic_erosion(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(xp.sum(out) <= xp.sum(mask))

        mask = xp.random.rand(5, 5) > 0.5
        out = math.isotropic_erosion(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(xp.all((out == 0) | (out == 1)))

        mask = xp.ones((7, 7), dtype=bool)
        out = math.isotropic_erosion(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(xp.sum(out) < xp.sum(mask))

        mask = xp.ones((5, 5, 5), dtype=bool)
        out = math.isotropic_erosion(mask, radius=1, backend=self.BACKEND)
        self.assertTrue(xp.sum(out) < xp.sum(mask))

        # thick slab (3 voxels in Z)
        mask = xp.zeros((5, 5, 5), dtype=bool)
        mask[1:4, :, :] = True
        out = math.isotropic_erosion(mask, radius=1, backend=self.BACKEND)
        # should shrink but not disappear
        self.assertTrue(xp.sum(out) > 0)
        # still centered
        self.assertTrue(xp.sum(out[2]) > 0)

        mask = xp.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True  # 3x3x3 cube
        out = math.isotropic_erosion(mask, radius=1, backend=self.BACKEND)
        # should shrink to 1 voxel
        self.assertTrue(xp.sum(out) > 0)

        mask = xp.ones((7, 7, 7))
        out = math.isotropic_erosion(mask, radius=1, backend=self.BACKEND)
        self.assertLess(xp.sum(out), xp.sum(mask))

# Extending the test and setting the backend to torch
@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestMath_Torch(TestMath_Numpy):
    BACKEND = "torch"

class TestMath_NumpyOnly(unittest.TestCase):
  
    @unittest.skipUnless(OPENCV_AVAILABLE, "OpenCV is not installed.")
    def test_Resize(self):
        input_image = np.random.rand(16, 16)
        feature = math.Resize(dsize=(8, 4))
        resized = feature.resolve(input_image)

        self.assertIsInstance(resized, np.ndarray)
        self.assertEqual(resized.shape, (4, 8))


    @unittest.skipUnless(OPENCV_AVAILABLE, "OpenCV is not installed.")
    def test_BlurCV2_GaussianBlur(self):
        import cv2

        input_image = np.random.rand(32, 32).astype(np.float32)
        expected_output = cv2.GaussianBlur(
            input_image, ksize=(5, 5), sigmaX=1, borderType=cv2.BORDER_REFLECT
        )
        feature = math.BlurCV2(
            filter_function=cv2.GaussianBlur, ksize=(5, 5), sigmaX=1, mode="reflect"
        )
        output_image = feature.resolve(input_image)
        self.assertTrue(output_image.shape == expected_output.shape)
        self.assertIsNone(
            np.testing.assert_allclose(
                output_image,
                expected_output,
                rtol=1e-5,
                atol=1e-6,
            )
        )

    @unittest.skipUnless(OPENCV_AVAILABLE, "OpenCV is not installed.")
    def test_BlurCV2_bilateralFilter(self):
        import cv2

        input_image = np.random.rand(32, 32).astype(np.float32)
        expected_output = cv2.bilateralFilter(
            input_image,
            d=9,
            sigmaColor=75,
            sigmaSpace=75,
            borderType=cv2.BORDER_REFLECT,
        )
        feature = math.BlurCV2(
            filter_function=cv2.bilateralFilter,
            d=9,
            sigmaColor=75,
            sigmaSpace=75,
            mode="reflect",
        )
        output_image = feature.resolve(input_image)
        self.assertTrue(output_image.shape == expected_output.shape)
        self.assertIsNone(
            np.testing.assert_allclose(
                output_image,
                expected_output,
                rtol=1e-5,
                atol=1e-6,
            )
        )

    @unittest.skipUnless(OPENCV_AVAILABLE, "OpenCV is not installed.")
    def test_BilateralBlur(self):
        import cv2

        input_image = np.random.rand(32, 32).astype(np.float32)
        expected_output = cv2.bilateralFilter(
            input_image,
            d=9,
            sigmaColor=75,
            sigmaSpace=75,
            borderType=cv2.BORDER_REFLECT,
        )
        feature = math.BilateralBlur(
            d=9, sigma_color=75, sigma_space=75, mode="reflect"
        )
        output_image = feature.resolve(input_image)
        self.assertTrue(output_image.shape == expected_output.shape)
        self.assertIsNone(
            np.testing.assert_allclose(
                output_image,
                expected_output,
                rtol=1e-5,
                atol=1e-6,
            )
        )
    

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestMath_TorchOnly(BackendTestBase):
    BACKEND = "torch"

    def test_Resize_torch_backend(self):
        feature = math.Resize(dsize=(4, 8))

        x = torch.rand(16, 16)
        out = feature.resolve(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (8, 4))

if __name__ == "__main__":
    unittest.main()
