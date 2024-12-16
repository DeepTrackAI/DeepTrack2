# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np
from deeptrack.backend.mie import *

class TestMie(unittest.TestCase):

    def test_mie_coefficients(self):
        m = 1.5 + 0.01j
        a = 0.5
        L = 5
        A, B = mie_coefficients(m, a, L)

        # Check the shape of the coefficients
        self.assertEqual(A.shape, (L,))
        self.assertEqual(B.shape, (L,))

        # Check the type of the coefficients
        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)

        # Check against test values from Sergio Aragon's Mie Scattering 
        # in Mathematica
        m = 4.0 / 3.0
        a = 50
        L = 1
        A, B = mie_coefficients(m, a, L)
        self.assertAlmostEqual(A.real, 0.5311058892948411929, delta=1e-8)
        self.assertAlmostEqual(A.imag, -0.4990314856310943073, delta=1e-8)
        self.assertAlmostEqual(B.real, 0.7919244759352004773, delta=1e-8)
        self.assertAlmostEqual(B.imag, -0.4059311522289938238, delta=1e-8)

        m = 1.5 + 1j
        a = 2
        L = 1
        A, B = mie_coefficients(m, a, L)
        self.assertAlmostEqual(A.real, 0.5465202033970914511, delta=1e-8)
        self.assertAlmostEqual(A.imag, -0.1523738572575972279, delta=1e-8)
        self.assertAlmostEqual(B.real, 0.3897147278879423235, delta=1e-8)
        self.assertAlmostEqual(B.imag, 0.2278960752564908264, delta=1e-8)

        m = 1.1 + 25j
        a = 2
        L = 2
        A, B = mie_coefficients(m, a, L)
        self.assertAlmostEqual(A[1].real, 0.324433578437, delta=1e-8)
        self.assertAlmostEqual(A[1].imag, -0.465627763266, delta=1e-8)
        self.assertAlmostEqual(B[1].real, 0.060464399088, delta=1e-8)
        self.assertAlmostEqual(B[1].imag, 0.236805417045, delta=1e-8)

    def test_stratified_mie_coefficients(self):
        m = [1.5 + 0.01j, 1.2 + 0.02j]
        a = [0.5, 0.3]
        L = 5
        an, bn = stratified_mie_coefficients(m, a, L)

        # Check the shape of the coefficients
        self.assertEqual(an.shape, (L,))
        self.assertEqual(bn.shape, (L,))

        # Check the type of the coefficients
        self.assertIsInstance(an, np.ndarray)
        self.assertIsInstance(bn, np.ndarray)

    def test_mie_harmonics(self):
        x = np.linspace(-1, 1, 100)
        L = 5
        PI, TAU = mie_harmonics(x, L)

        # Check the shape of the harmonics
        self.assertEqual(PI.shape, (L, 100))
        self.assertEqual(TAU.shape, (L, 100))

        # Check the type of the harmonics
        self.assertIsInstance(PI, np.ndarray)
        self.assertIsInstance(TAU, np.ndarray)

        # Check against test values
        x = np.array([0.4])
        L = 4
        PI_expected = np.array([[1], [1.2], [-0.3],[-1.88]])
        TAU_expected = np.array([[0.4], [-2.04], [-5.16],[-1.508]])
        PI, TAU = mie_harmonics(x, L)
        self.assertTrue(np.allclose(PI, PI_expected))
        self.assertTrue(np.allclose(TAU, TAU_expected))
        
        x = np.array([0])
        L = 5
        PI_expected = np.array([[1], [0], [-1.5], [0], [1.875]])
        TAU_expected = np.array([[0], [-3],[0],[7.5],[0]])
        PI, TAU = mie_harmonics(x, L)
        self.assertTrue(np.allclose(PI, PI_expected))
        self.assertTrue(np.allclose(TAU, TAU_expected))
        
        x = np.array([-0.5])
        L = 3
        PI_expected = np.array([[1], [-1.5], [0.375]])
        TAU_expected = np.array([[-0.5], [-1.5], [5.4375]])
        PI, TAU = mie_harmonics(x, L)
        self.assertTrue(np.allclose(PI, PI_expected))
        self.assertTrue(np.allclose(TAU, TAU_expected))


if __name__ == "__main__":
    unittest.main()
