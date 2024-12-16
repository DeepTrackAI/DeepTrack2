import unittest

import numpy as np
from deeptrack.extras import radialcenter


class TestRadialCenter(unittest.TestCase):

    def test_gaussian(self):

        linspace = np.linspace(-10, 10, 100)
        gaussian = np.exp(-0.5 * (
            linspace[:, None] ** 2 + linspace[None, :] ** 2)
        )
        x, y = radialcenter.radialcenter(gaussian)

        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertAlmostEqual(x, 50.0, delta=1)
        self.assertAlmostEqual(y, 50.0, delta=1)

if __name__ == "__main__":
    unittest.main()
