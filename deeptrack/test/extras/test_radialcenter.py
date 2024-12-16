import unittest

from deeptrack.extras import radialcenter as rc

import numpy as np

class TestRadialCenter(unittest.TestCase):
    def test_noise(self):
        intensity_map = np.random.normal(0, 0.005, (100, 100))
        x, y = rc.radialcenter(intensity_map)
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertAlmostEqual(x, 50.0,delta=5)
        self.assertAlmostEqual(y, 50.0,delta=5)

if __name__ == "__main__":
    unittest.main()
