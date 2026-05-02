# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack import holography

class TestOpticalFieldFunctions(unittest.TestCase):
    
    def test_get_propagation_matrix(self):        
        propagation_matrix = holography.get_propagation_matrix(
            shape=(128, 128),
            to_z=1.0,
            pixel_size=0.1,
            wavelength=0.65e-6,
            dx=0,
            dy=0
        )
        self.assertEqual(propagation_matrix.shape, (128, 128))
        self.assertTrue(np.iscomplexobj(propagation_matrix))

    def test_rescale(self):
        rescale_factor = 0.5
        image = np.random.rand(128, 128, 2) 
        rescale_op = holography.Rescale(rescale=rescale_factor)
        scaled_image = rescale_op(image)
        mean_value = (image[..., 0].mean() - 1) * rescale_factor + 1
        self.assertAlmostEqual(scaled_image[..., 0].mean(), mean_value)
        expected_image = image[..., 1] * rescale_factor
        self.assertTrue(np.allclose(scaled_image[..., 1], expected_image))

    def test_fourier_transform(self):
        image = np.random.rand(128, 128, 2) 
        ft_op = holography.FourierTransform()
        transformed_image = ft_op(image)
        self.assertTrue(np.iscomplexobj(transformed_image))
        self.assertEqual(transformed_image.shape, (192, 192)) # 128+2*32 = 192

    def test_inverse_fourier_transform(self):
        image = np.random.rand(128, 128, 2)
        ft_op = holography.FourierTransform()
        transformed_image = ft_op(image)
        ift_op = holography.InverseFourierTransform()
        reconstr_image = ift_op(transformed_image)
        self.assertTrue(np.allclose(image, reconstr_image, atol=1e-5))
        

if __name__ == '__main__':
    unittest.main()
