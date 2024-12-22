import unittest

import numpy as np

from deeptrack.extras import radialcenter


class TestRadialCenter(unittest.TestCase):
    
    def test_noise(self):
        intensity_map = np.random.normal(0, 0.005, (100, 100))
        x, y = radialcenter(intensity_map)
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertAlmostEqual(x, 50.0,delta=5)
        self.assertAlmostEqual(y, 50.0,delta=5)


if __name__ == "__main__":
    unittest.main()
