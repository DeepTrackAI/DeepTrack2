import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import math

import numpy as np

class TestMath(unittest.TestCase):
    def test_Clip(self):
        input_image = np.array([[10, 4], [4, -10]])
        feature = math.Clip(min=-5, max=5)
        clipped_feature = feature.resolve(input_image)
        self.assertTrue(np.all(clipped_feature == [[5, 4], [4, -5]]))

    def test_NormalizeMinMax(self):

        feature = math.NormalizeMinMax(min=-5, max=5)

        input_image = np.array([[10, 4], [4, -10]])
        normalized_image = feature(input_image)
        self.assertTrue(np.all(normalized_image == [[5, 2], [2, -5]]))

if __name__ == "__main__":
    unittest.main()