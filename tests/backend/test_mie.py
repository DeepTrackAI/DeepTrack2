# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack.backend import config, mie, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class TestMie(unittest.TestCase):

    def setUp(self):
        config.set_backend("numpy")

    def test_coefficients(self):
        m = 1.5 + 0.01j
        a = 0.5
        L = 5
        A, B = mie.coefficients(m, a, L)

        # Check the shape of the coefficients.
        self.assertEqual(A.shape, (L,))
        self.assertEqual(B.shape, (L,))

        # Check the type of the coefficients.
        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)

        # Check against test values from Sergio Aragon's Mie Scattering 
        # in Mathematica.
        m = 4.0 / 3.0
        a = 50
        L = 1
        A, B = mie.coefficients(m, a, L)
        self.assertAlmostEqual(A.real, 0.5311058892948411929, delta=1e-8)
        self.assertAlmostEqual(A.imag, -0.4990314856310943073, delta=1e-8)
        self.assertAlmostEqual(B.real, 0.7919244759352004773, delta=1e-8)
        self.assertAlmostEqual(B.imag, -0.4059311522289938238, delta=1e-8)

        m = 1.5 + 1j
        a = 2
        L = 1
        A, B = mie.coefficients(m, a, L)
        self.assertAlmostEqual(A.real, 0.5465202033970914511, delta=1e-8)
        self.assertAlmostEqual(A.imag, -0.1523738572575972279, delta=1e-8)
        self.assertAlmostEqual(B.real, 0.3897147278879423235, delta=1e-8)
        self.assertAlmostEqual(B.imag, 0.2278960752564908264, delta=1e-8)

        m = 1.1 + 25j
        a = 2
        L = 2
        A, B = mie.coefficients(m, a, L)
        self.assertAlmostEqual(A[1].real, 0.324433578437, delta=1e-8)
        self.assertAlmostEqual(A[1].imag, -0.465627763266, delta=1e-8)
        self.assertAlmostEqual(B[1].real, 0.060464399088, delta=1e-8)
        self.assertAlmostEqual(B[1].imag, 0.236805417045, delta=1e-8)


    def test_stratified_coefficients(self):
        m = [1.5 + 0.01j, 1.2 + 0.02j]
        a = [0.5, 0.3]
        L = 5
        an, bn = mie.stratified_coefficients(m, a, L)

        # Check the shape of the coefficients.
        self.assertEqual(an.shape, (L,))
        self.assertEqual(bn.shape, (L,))

        # Check the type of the coefficients.
        self.assertIsInstance(an, np.ndarray)
        self.assertIsInstance(bn, np.ndarray)


    def test_harmonics(self):
        x = np.linspace(-1, 1, 100)
        L = 5
        PI, TAU = mie.harmonics(x, L)

        # Check the shape of the harmonics.
        self.assertEqual(PI.shape, (L, 100))
        self.assertEqual(TAU.shape, (L, 100))

        # Check the type of the harmonics.
        self.assertIsInstance(PI, np.ndarray)
        self.assertIsInstance(TAU, np.ndarray)

        # Check against test values.
        x = np.array([0.4])
        L = 4
        PI_expected = np.array([[1], [1.2], [-0.3],[-1.88]])
        TAU_expected = np.array([[0.4], [-2.04], [-5.16],[-1.508]])
        PI, TAU = mie.harmonics(x, L)
        self.assertTrue(np.allclose(PI, PI_expected))
        self.assertTrue(np.allclose(TAU, TAU_expected))

        x = np.array([0])
        L = 5
        PI_expected = np.array([[1], [0], [-1.5], [0], [1.875]])
        TAU_expected = np.array([[0], [-3],[0],[7.5],[0]])
        PI, TAU = mie.harmonics(x, L)
        self.assertTrue(np.allclose(PI, PI_expected))
        self.assertTrue(np.allclose(TAU, TAU_expected))

        x = np.array([-0.5])
        L = 3
        PI_expected = np.array([[1], [-1.5], [0.375]])
        TAU_expected = np.array([[-0.5], [-1.5], [5.4375]])
        PI, TAU = mie.harmonics(x, L)
        self.assertTrue(np.allclose(PI, PI_expected))
        self.assertTrue(np.allclose(TAU, TAU_expected))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestMieTorch(unittest.TestCase):

    def setUp(self):
        config.set_backend("numpy")

    def test_coefficients_matches_numpy_and_autodiff(self):
        m = 1.5 + 0.01j
        a_np = 0.5
        L = 5

        A_expected, B_expected = mie.coefficients(m, a_np, L)

        with config.with_backend("torch"):
            a = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
            A, B = mie.coefficients(m, a, L)

            self.assertIsInstance(A, torch.Tensor)
            self.assertIsInstance(B, torch.Tensor)
            self.assertEqual(A.shape, (L,))
            self.assertEqual(B.shape, (L,))

            self.assertTrue(
                np.allclose(
                    A.detach().numpy(), A_expected, rtol=1e-10, atol=1e-10
                )
            )
            self.assertTrue(
                np.allclose(
                    B.detach().numpy(), B_expected, rtol=1e-10, atol=1e-10
                )
            )

            loss = torch.abs(A).sum() + torch.abs(B).sum()
            loss.backward()

            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad))
            self.assertGreater(abs(float(a.grad)), 0)

    def test_coefficients_refractive_index_autodiff(self):
        with config.with_backend("torch"):
            m = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
            a = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

            A, B = mie.coefficients(m, a, 5)

            loss = torch.abs(A).sum() + torch.abs(B).sum()
            loss.backward()

            self.assertIsNotNone(m.grad)
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(m.grad))
            self.assertTrue(torch.isfinite(a.grad))
            self.assertGreater(abs(float(m.grad)), 0)
            self.assertGreater(abs(float(a.grad)), 0)

    def test_stratified_coefficients_matches_numpy_and_autodiff(self):
        m = [1.5 + 0.01j, 1.2 + 0.02j]
        a_np = [0.5, 0.3]
        L = 5

        an_expected, bn_expected = mie.stratified_coefficients(m, a_np, L)

        with config.with_backend("torch"):
            a = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
            an, bn = mie.stratified_coefficients(m, a, L)

            self.assertIsInstance(an, torch.Tensor)
            self.assertIsInstance(bn, torch.Tensor)
            self.assertEqual(an.shape, (L,))
            self.assertEqual(bn.shape, (L,))

            self.assertTrue(
                np.allclose(
                    an.detach().numpy(), an_expected, rtol=1e-10, atol=1e-10
                )
            )
            self.assertTrue(
                np.allclose(
                    bn.detach().numpy(), bn_expected, rtol=1e-10, atol=1e-10
                )
            )

            loss = torch.abs(an).sum() + torch.abs(bn).sum()
            loss.backward()

            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertGreater(float(torch.linalg.vector_norm(a.grad)), 0)

    def test_harmonics_matches_numpy_and_autodiff(self):
        x_np = np.array([0.4])
        L = 4
        PI_expected, TAU_expected = mie.harmonics(x_np, L)

        with config.with_backend("torch"):
            x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
            PI, TAU = mie.harmonics(x, L)

            self.assertIsInstance(PI, torch.Tensor)
            self.assertIsInstance(TAU, torch.Tensor)
            self.assertEqual(PI.shape, (L, 1))
            self.assertEqual(TAU.shape, (L, 1))

            self.assertTrue(np.allclose(PI.detach().numpy(), PI_expected))
            self.assertTrue(np.allclose(TAU.detach().numpy(), TAU_expected))

            loss = PI.sum() + TAU.sum()
            loss.backward()

            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(torch.linalg.vector_norm(x.grad)), 0)


if __name__ == "__main__":
    unittest.main()
