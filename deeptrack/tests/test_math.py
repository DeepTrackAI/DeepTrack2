# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np
from scipy.ndimage import uniform_filter

from deeptrack import math


class TestMath(unittest.TestCase):
    def test_Clip(self):
        input_image = np.array([[10, 4], [4, -10]])
        feature = math.Clip(min=-5, max=5)
        clipped_feature = feature.resolve(input_image)
        self.assertTrue(np.all(clipped_feature == [[5, 4], [4, -5]]))

        input_image = np.array([[5, 6], [7, 8]])
        feature = math.Clip(min=0, max=10)
        clipped_feature = feature.resolve(input_image)
        self.assertTrue(np.all(clipped_feature == [[5, 6], [7, 8]]))

    def test_NormalizeMinMax(self):
        feature = math.NormalizeMinMax(min=-5, max=5)
        input_image = np.array([[10, 4], [4, -10]])
        normalized_image = feature.resolve(input_image)
        self.assertTrue(np.all(normalized_image == [[5, 2], [2, -5]]))

    def test_NormalizeStandard(self):
        feature = math.NormalizeStandard()
        input_image = np.array([[1, 2], [3, 4]], dtype=float)
        normalized_image = feature.resolve(input_image)
        self.assertEqual(np.mean(normalized_image), 0)
        self.assertEqual(np.std(normalized_image), 1)

    def test_Blur(self):
        input_image = np.array([[1, 2], [3, 4]], dtype=float)
        feature = math.Blur(filter_function=uniform_filter, size=2)
        blurred_image = feature.resolve(input_image)
        self.assertTrue(np.all(blurred_image == [[1, 1.5], [2, 2.5]]))

    def test_GaussianBlur(self):
        input_image = np.array([[1, 2], [3, 4]], dtype=float)
        feature = math.GaussianBlur(sigma=0)
        blurred_image = feature.resolve(input_image)
        self.assertTrue(np.all(blurred_image == [[1, 2], [3, 4]]))

        input_image = np.array([[1, 2], [3, 4]], dtype=float)
        feature = math.GaussianBlur(sigma=1000)
        blurred_image = feature.resolve(input_image)
        self.assertTrue(
            np.all(blurred_image - [[2.5, 2.5], [2.5, 2.5]] <= 0.01)
        )

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
     

if __name__ == "__main__":
    unittest.main()