# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name
 
# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import deeptrack as dt
import numpy as np
from numpy.random import Generator, PCG64

class TestDLCC(unittest.TestCase):

    def test_3B(self):
        ## PART 1
        # Image pipeline with reproducible randomness.
        rng = Generator(PCG64(42))

        image_size = 3

        particle = dt.scatterers.MieSphere(
            position=lambda: rng.uniform(image_size / 2 - 5,
                                         image_size / 2 + 5, 2),
            z=lambda: rng.uniform(-1, 1),
            radius=lambda: rng.uniform(500, 600) * 1e-9,
            refractive_index=lambda: rng.uniform(1.37, 1.42),
            position_unit="pixel",
        )

        brightfield_microscope = dt.optics.Brightfield(
            wavelength=630e-9,
            NA=0.8,
            resolution=1e-6,
            magnification=15,
            refractive_index_medium=1.33,
            output_region=(0, 0, image_size, image_size),
        )

        imaged_particle = brightfield_microscope(particle)

        # First resolve
        expected_first = np.array(
            [[[0.84823867], [0.84193078], [0.85179172]],
            [[0.80131109], [0.79478238], [0.80499449]],
            [[0.76094944], [0.75431347], [0.76469805]]]
        )
        actual_first = imaged_particle()
        np.testing.assert_allclose(actual_first, expected_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        np.testing.assert_allclose(imaged_particle(), expected_first,
                                   rtol=1e-7, atol=1e-7)

        # Change after update
        expected_second = np.array(
            [[[1.02220767], [0.98545219], [0.95156275]],
            [[0.98707199], [0.94484236], [0.90718955]],
            [[0.94221739], [0.89565235], [0.85521306]]]
        )
        actual_second = imaged_particle.update()()
        np.testing.assert_allclose(actual_second, expected_second,
                                   rtol=1e-7, atol=1e-7)

        ## PART 2
        # Poisson uses np.random so no randomness reproducibility,
        # so images are not reproducible.
        noise = dt.Poisson(
            min_snr=5,
            max_snr=20,
            background=1,
            snr=lambda min_snr, max_snr: rng.uniform(min_snr, max_snr),
        )
        noisy_imaged_particle = imaged_particle >> noise

        normalization = dt.NormalizeMinMax(
            lambda: rng.uniform(0.0, 0.2),
            lambda: rng.uniform(0.8, 1.0),
        )
        image_pipeline = noisy_imaged_particle >> normalization

        # First resolve
        actual_first = image_pipeline()

        # No change when resolving again
        np.testing.assert_allclose(image_pipeline(), actual_first,
                                rtol=1e-7, atol=1e-7)

        # Change after update
        actual_second = image_pipeline.update()()
        assert float(np.max(np.abs(actual_first - actual_second))) > 1e-6

        ## PART 3
        # Images are not reproducible becaus eof Poisson, but positions are.
        rng = Generator(PCG64(42))

        pipeline = image_pipeline & particle.position

        # First resolve
        _, actual_position_first = pipeline.update()()

        expected_position_first = np.array([4.23956049, 0.8887844 ])
        np.testing.assert_allclose(actual_position_first,
                                   expected_position_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        _, actual_position_first_2 = pipeline()
        np.testing.assert_allclose(actual_position_first_2,
                                   expected_position_first,
                                   rtol=1e-7, atol=1e-7)

        # Change after update
        expected_position_second = np.array([-2.21886367,  1.00385938])
        _, actual_position_second = pipeline.update()()
        np.testing.assert_allclose(actual_position_second,
                                   expected_position_second,
                                   rtol=1e-7, atol=1e-7)

if __name__ == "__main__":
    unittest.main()
