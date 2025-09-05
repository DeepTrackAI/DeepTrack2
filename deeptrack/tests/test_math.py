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

    def test_Average(self):
        input_image0 = xp.ones((10, 30, 20)) * 2
        input_image1 = xp.ones((10, 30, 20)) * 4
        feature = math.Average(axis=0)
        average = feature.resolve([input_image0, input_image1])
        self.assertTrue(xp.all(average == 3), True)
        self.assertEqual(average.shape, (10, 30, 20))


    def test_Clip(self):
        input_image = xp.asarray([[10, 4], [4, -10]])
        feature = math.Clip(min=-5, max=5)
        clipped_feature = feature.resolve(input_image)
        self.assertTrue(
            xp.all(clipped_feature == xp.asarray([[5, 4], [4, -5]]))
        )

        input_image = xp.asarray(np.array([[5, 6], [7, 8]]))
        feature = math.Clip(min=0, max=10)
        clipped_feature = feature.resolve(input_image)
        self.assertTrue(
            xp.all(clipped_feature == xp.asarray([[5, 6], [7, 8]]))
        )


    def test_NormalizeMinMax(self):
        input_image = xp.asarray([[10, 4], [4, -10]])
        feature = math.NormalizeMinMax(min=-5, max=5)
        normalized_image = feature.resolve(input_image)
        self.assertTrue(
            xp.all(normalized_image == xp.asarray([[5, 2], [2, -5]]))
        )


    def test_NormalizeStandard(self):
        input_image = xp.asarray([[1, 2], [3, 4]], dtype=float)
        feature = math.NormalizeStandard()
        normalized_image = feature.resolve(input_image)
        self.assertEqual(xp.mean(normalized_image), 0)
        if apc.is_torch_array(normalized_image):
            # By default, torch.std() is unbiased, i.e., divides by N-1
            self.assertEqual(torch.std(normalized_image, unbiased=False), 1)
        else:
            self.assertEqual(xp.std(normalized_image), 1)


    def test_NormalizeQuantile(self):
        input_image = xp.asarray([[1, 2], [3, 100]], dtype=float)
        feature = math.NormalizeQuantile(quantiles=(0.25, 0.75))
        output = feature.resolve(input_image)
        self.assertAlmostEqual(xp.quantile(output, 0.5), 0, places=5)


    def test_Blur(self):
        # TODO: check this test with torch
        pass
        #input_image = xp.asarray(np.array([[1, 2], [3, 4]], dtype=float))
        #expected_output = xp.asarray(np.array([[1, 1.5], [2, 2.5]]))

        #eature = math.Blur(filter_function=uniform_filter, size=2)
        #blurred_image = feature.resolve(input_image)
        #self.assertTrue(xp.all(blurred_image == expected_output))

    def test_MinPooling(self):
        input_image = xp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        feature = math.MinPooling(ksize=2)
        pooled_image = feature.resolve(input_image)

        expected = xp.asarray([[1.0, 3.0]], dtype=float)

        self.assertEqual(pooled_image.shape, (1, 2))
        self.assertTrue(xp.all(pooled_image == expected))


# Extending the test and setting the backend to torch
@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestMath_Torch(TestMath_Numpy):
    BACKEND = "torch"
    pass


class TestMath(unittest.TestCase):

    def test_GaussianBlur(self):
        input_image = np.array([[1, 2], [3, 4]], dtype=float)
        feature = math.GaussianBlur(sigma=0)
        blurred_image = feature.resolve(input_image)
        self.assertTrue(np.all(blurred_image == [[1, 2], [3, 4]]))

        input_image = np.array([[1, 2], [3, 4]], dtype=float)
        feature = math.GaussianBlur(sigma=1000)
        blurred_image = feature.resolve(input_image)
        self.assertTrue(np.all(blurred_image - [[2.5, 2.5], [2.5, 2.5]] <= 0.01))

    def test_AveragePooling(self):
        input_image = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        feature = math.AveragePooling(ksize=2)
        pooled_image = feature.resolve(input_image)
        self.assertTrue(np.all(pooled_image == [[3.5, 5.5]]))

    def test_MaxPooling(self):
        input_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        feature = math.MaxPooling(ksize=2)
        pooled_image = feature.resolve(input_image)
        self.assertTrue(np.all(pooled_image == [[5, 6], [8, 9]]))

    def test_MinPooling(self):
        input_image = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        feature = math.MinPooling(ksize=2)
        pooled_image = feature.resolve(input_image)
        self.assertTrue(np.all(pooled_image == [[1, 3]]))

    def test_MedianBlur(self):
        input_image = np.random.rand(32, 32)
        feature = math.MedianBlur(ksize=3)
        output = feature.resolve(input_image)
        self.assertEqual(output.shape, input_image.shape)

    def test_MedianPooling(self):
        input_image = np.array([[1, 3, 2, 4], [5, 7, 6, 8]], dtype=float)
        feature = math.MedianPooling(ksize=2)
        pooled = feature.resolve(input_image)
        self.assertEqual(pooled.shape, (1, 2))

    @unittest.skipUnless(OPENCV_AVAILABLE, "OpenCV is not installed.")
    def test_Resize(self):
        input_image = np.random.rand(16, 16)
        feature = math.Resize(dsize=(8, 8))
        resized = feature.resolve(input_image)
        self.assertEqual(resized.shape, (8, 8))

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


if __name__ == "__main__":
    unittest.main()
