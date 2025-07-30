# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack.image import Image
from deeptrack import noises, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

class TestNoises(unittest.TestCase):
    def test_Offset(self):
        noise = noises.Offset(offset=0.5)
        input_image = Image(np.zeros((256, 256)))
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(np.all(np.array(output_image) == 0.5))

    def test_Background(self):
        # Test with DeepTrack Image
        noise = noises.Background(offset=0.5)
        input_image = Image(np.zeros((256, 256)))
        output_image = noise.resolve(input_image)

        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(np.all(np.array(output_image) == 0.5))

        # Test with NumPy array
        noise = noises.Background(offset=0.5)
        input_image = np.ones((10, 10))
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (10, 10))
        self.assertTrue(np.all(np.array(output_image) == 1.5))

        ### Test with PyTorch tensor (if available)
        if TORCH_AVAILABLE:
            noise = noises.Background(offset=0.25)
            input_image = torch.zeros(5,5)
            output_image = noise.resolve(input_image)

            self.assertIsInstance(output_image, torch.Tensor)
            self.assertEqual(output_image.shape, (5,5))
            self.assertTrue(torch.all(output_image == 0.25).item())

    def test_Gaussian(self):
        noise = noises.Gaussian(mu=0.1, sigma=0.05)
        input_image = Image(np.zeros((256, 256)))
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))

    def test_ComplexGaussian(self):
        noise = noises.ComplexGaussian(mu=0.1, sigma=0.05)
        input_image = Image(np.zeros((256, 256)))
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(np.iscomplexobj(output_image))

    def test_Poisson(self):
        noise = noises.Poisson(snr=20)
        input_image = Image(np.ones((256, 256)) * 0.1)
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))


if __name__ == "__main__":
    unittest.main()