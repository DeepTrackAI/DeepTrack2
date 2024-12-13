# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np
from deeptrack.backend import mie

class TestMie(unittest.TestCase):

    def test_mie_coefficients(self):
        m = 1.5 + 0.01j
        a = 0.5
        L = 5
        A, B = mie.mie_coefficients(m, a, L)

        # Check the shape of the coefficients
        self.assertEqual(A.shape, (L,))
        self.assertEqual(B.shape, (L,))

        # Check the type of the coefficients
        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)

    def test_stratified_mie_coefficients(self):
        m = [1.5 + 0.01j, 1.2 + 0.02j]
        a = [0.5, 0.3]
        L = 5
        an, bn = mie.stratified_mie_coefficients(m, a, L)

        # Check the shape of the coefficients
        self.assertEqual(an.shape, (L,))
        self.assertEqual(bn.shape, (L,))

        # Check the type of the coefficients
        self.assertIsInstance(an, np.ndarray)
        self.assertIsInstance(bn, np.ndarray)

    def test_mie_harmonics(self):
        x = np.linspace(-1, 1, 100)
        L = 5
        PI, TAU = mie.mie_harmonics(x, L)

        # Check the shape of the harmonics
        self.assertEqual(PI.shape, (L, 100))
        self.assertEqual(TAU.shape, (L, 100))

        # Check the type of the harmonics
        self.assertIsInstance(PI, np.ndarray)
        self.assertIsInstance(TAU, np.ndarray)

if __name__ == "__main__":
    unittest.main()
