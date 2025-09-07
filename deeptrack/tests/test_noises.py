# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import array_api_compat as apc
import numpy as np

from deeptrack.image import Image
from deeptrack import noises

from deeptrack.backend import OPENCV_AVAILABLE, TORCH_AVAILABLE, xp
from deeptrack.tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch

class TestNoises_Numpy(BackendTestBase):
    BACKEND = "numpy"

    def test_Offset(self):
        noise = noises.Offset(offset=0.5)
        input_image = Image(xp.zeros((256, 256)))
        output_image = noise.resolve(input_image)

        #self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(xp.all(xp.asarray(output_image) == 0.5))

    def test_Background(self):
        # Test with DeepTrack Image
        noise = noises.Background(offset=0.5)
        input_image = Image(xp.zeros((256, 256)))
        output_image = noise.resolve(input_image)

        #self.assertIsInstance(output_image, input_image)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(xp.all(xp.asarray(output_image) == 0.5))

        # Test with arrays
        noise = noises.Background(offset=0.5)
        input_image = xp.ones((10, 10))
        output_image = noise.resolve(input_image)

        #self.assertIsInstance(output_image, input_image)
        self.assertEqual(output_image.shape, (10, 10))
        self.assertTrue(xp.all(xp.asarray(output_image) == 1.5))

    def test_Gaussian(self):
        noise = noises.Gaussian(mu=0.1, sigma=0.05)
        input_image = Image(xp.zeros((256, 256)))
        output_image = noise.resolve(input_image)
        
        #self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))

    def test_ComplexGaussian(self):
        noise = noises.ComplexGaussian(mu=0.1, sigma=0.05)
        input_image = Image(xp.zeros((256, 256)))
        output_image = noise.resolve(input_image)

        #self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))
        #TODO: Add xp function to check complex values
        self.assertTrue(xp.any(output_image.imag != 0))

    def test_Poisson(self):
        noise = noises.Poisson(snr=20)
        input_image = xp.ones((256, 256)) * 0.1
        output_image = noise.resolve(input_image)

        #self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))


# Extending the test and setting the backend to torch
@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestNoises_Torch(TestNoises_Numpy):
    BACKEND = "torch"
    pass

if __name__ == "__main__":
    unittest.main()
