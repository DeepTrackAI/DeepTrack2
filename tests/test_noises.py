# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack.optical import noises

from deeptrack.backend import TORCH_AVAILABLE, xp
from tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch


class TestNoises_NumPy(BackendTestBase):
    BACKEND = "numpy"

    @property
    def array_type(self):
        if self.BACKEND == "numpy":
            return np.ndarray
        elif self.BACKEND == "torch":
            return torch.Tensor
        else:
            raise ValueError(f"Unsupported backend: {self.BACKEND}")

    def test___all__(self):
        from deeptrack import (
            Noise,
            Background,
            Offset,
            Gaussian,
            ComplexGaussian,
            Poisson,
        )

    def test_Offset(self):
        noise = noises.Offset(offset=0.5)
        input_image = xp.zeros((256, 256))
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(xp.all(xp.asarray(output_image) == 0.5))

    def test_Background(self):
        # Test with DeepTrack image
        noise = noises.Background(offset=0.5)
        input_image = xp.zeros((256, 256))
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(xp.all(xp.asarray(output_image) == 0.5))

        # Test with arrays
        noise = noises.Background(offset=0.5)
        input_image = xp.ones((10, 10))
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (10, 10))
        self.assertTrue(xp.all(xp.asarray(output_image) == 1.5))

    def test_Gaussian(self):
        noise = noises.Gaussian(mu=0.1, sigma=0.05)
        input_image = xp.zeros((256, 256))
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (256, 256))

    def test_Gaussian_zero_sigma(self):
        noise = noises.Gaussian(mu=1, sigma=0)
        image = xp.zeros((10, 10))
        out = noise.resolve(image)

        self.assertTrue(xp.all(out == 1))

    def test_ComplexGaussian(self):
        noise = noises.ComplexGaussian(mu=0.1, sigma=0.05)
        input_image = xp.zeros((256, 256))
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(xp.any(output_image.imag != 0))

        if self.BACKEND == "numpy":
            self.assertTrue(np.iscomplexobj(output_image))
        elif self.BACKEND == "torch":
            self.assertTrue(torch.is_complex(output_image))

    def test_Poisson(self):
        noise = noises.Poisson(snr=20)
        input_image = xp.ones((256, 256)) * 0.1
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (256, 256))

    def test_Poisson_negative_input(self):
        noise = noises.Poisson(snr=10)
        image = xp.asarray([-1, -0.5, 0, 1])

        out = noise.resolve(image)

        self.assertEqual(out.shape, image.shape)

    def test_Poisson_zero_signal(self):
        noise = noises.Poisson(snr=10, background=0)
        image = xp.zeros((10, 10))

        out = noise.resolve(image)

        self.assertEqual(out.shape, image.shape)

    def test_PropertyLike(self):
        noise = noises.Gaussian(mu=lambda: 1, sigma=lambda: 0)
        image = xp.zeros((5, 5))

        out = noise.resolve(image)

        self.assertTrue(xp.all(out == 1))

    def test_Device(self):
        if self.BACKEND == "torch":

            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")

            for device in devices:
                image = torch.zeros((10, 10), device=device)
                noise = noises.Gaussian()

                out = noise.resolve(image)

                self.assertEqual(out.device, image.device)


# Extending the test and setting the backend to torch
@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestNoises_PyTorch(TestNoises_NumPy):
    BACKEND = "torch"


if __name__ == "__main__":
    unittest.main()
