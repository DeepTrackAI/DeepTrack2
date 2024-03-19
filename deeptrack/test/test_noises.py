import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import noises

from ..image import Image
import numpy as np


class TestNoises(unittest.TestCase):
    def test_Offset(self):
        noise = noises.Offset(offset=0.5)
        input_image = Image(np.zeros((256, 256)))
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(np.all(np.array(output_image) == 0.5))

    def test_Background(self):
        noise = noises.Background(offset=0.5)
        input_image = Image(np.zeros((256, 256)))
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))
        self.assertTrue(np.all(np.array(output_image) == 0.5))

    def test_Gaussian(self):
        noise = noises.Gaussian(mu=0.1, sigma=0.05)
        input_image = Image(np.zeros((256, 256)))
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))

    def test_Poisson(self):
        noise = noises.Poisson(snr=20)
        input_image = Image(np.ones((256, 256)) * 0.1)
        output_image = noise.resolve(input_image)
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (256, 256))


if __name__ == "__main__":
    unittest.main()