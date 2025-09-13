# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name
 
# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import deeptrack as dt
from deeptrack import TORCH_AVAILABLE
import numpy as np
from numpy.random import Generator, PCG64

if TORCH_AVAILABLE:
    import torch

class TestDLCC(unittest.TestCase):

    def test_3_B(self):
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

    def test_4_1(self):
        ## PART 1
        # Deterministic pipeline.

        particle = dt.Sphere(
            position=np.array([0.5, 0.5]) * 4,
            position_unit="pixel",
            radius=500 * dt.units.nm,
            refractive_index=1.45 + 0.02j,
        )

        brightfield_microscope = dt.Brightfield(
            wavelength=500 * dt.units.nm,
            NA=1.0,
            resolution=1 * dt.units.um,
            magnification=10,
            refractive_index_medium=1.33,
            output_region=(0, 0, 4, 4),
        )

        illuminated_sample = brightfield_microscope(particle)

        # First resolve
        expected_first = np.array(
            [[[0.55382582], [0.55944586], [0.54341977], [0.55944587]],
            [[0.55944586], [0.48773998], [0.43907532], [0.48773999]],
            [[0.54341977], [0.43907532], [0.37978135], [0.43907532]],
            [[0.55944587], [0.48773999], [0.43907532], [0.48773999]]]
        )
        actual_first = illuminated_sample()
        np.testing.assert_allclose(actual_first, expected_first,
                                   rtol=1e-7, atol=1e-7)

        # No change when resolving again
        np.testing.assert_allclose(illuminated_sample(), expected_first,
                                   rtol=1e-7, atol=1e-7)

        # No change also after update (deterministic pipeline)
        np.testing.assert_allclose(illuminated_sample.update()(),
                                   expected_first,
                                   rtol=1e-7, atol=1e-7)

        ## PART 2
        # Non-reproducible randomness for noisy_particle.
        if TORCH_AVAILABLE:
            clean_particle = (
                illuminated_sample
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )

            noise = dt.Poisson(snr=lambda: 2.0 + np.random.rand())

            noisy_particle = (
                illuminated_sample >> noise
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )

            pip = noisy_particle & clean_particle

            # First resolve
            actual_noisy_first, actual_clean_first = pip()
            expected_clean_first = torch.tensor(
                [[[0.9687, 1.0000, 0.9108, 1.0000],
                [1.0000, 0.6009, 0.3300, 0.6009],
                [0.9108, 0.3300, 0.0000, 0.3300],
                [1.0000, 0.6009, 0.3300, 0.6009]]]
            ).to(device=actual_clean_first.device,
                 dtype=actual_clean_first.dtype)
            torch.testing.assert_close(actual_clean_first,
                                       expected_clean_first,
                                       rtol=1e-7, atol=1e-4)

            # No change after resolving again
            actual_noisy_first_2, actual_clean_first_2 = pip()
            torch.testing.assert_close(actual_clean_first_2,
                                       expected_clean_first,
                                       rtol=1e-7, atol=1e-4)
            torch.testing.assert_close(actual_noisy_first_2,
                                       actual_noisy_first,
                                       rtol=1e-7, atol=1e-4)

            # No change for clean also after update (deterministic pipeline),
            # but change for noisy
            actual_noisy_second, actual_clean_second = pip.update().resolve()
            torch.testing.assert_close(actual_clean_second,
                                       expected_clean_first,
                                       rtol=1e-7, atol=1e-4)
            self.assertFalse(torch.allclose(
                actual_noisy_first, actual_noisy_second, rtol=1e-7, atol=1e-4
            ))

        ## PART 3
        # Verify generation of blank image.
        if TORCH_AVAILABLE:
            blank = brightfield_microscope(particle ^ 0)
            blank_pip = (
                blank # >> noise >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )

            expected = torch.tensor(
                [[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]]
            )
            torch.testing.assert_close(blank_pip(), expected,
                                       rtol=1e-7, atol=1e-4)

        ## PART 4
        # Check diverse particle pipeline.
        if TORCH_AVAILABLE:
            diverse_particle = dt.Sphere(
                position=lambda: np.array([.2, .2]
                                          + np.random.rand(2) * .6) * 4,
                radius=lambda: 500 * dt.units.nm * (1 + np.random.rand()),
                position_unit="pixel",
                refractive_index=1.45 + 0.02j,
            )
            diverse_illuminated_sample = \
                brightfield_microscope(diverse_particle)
            diverse_clean_particle = (
                diverse_illuminated_sample
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )
            diverse_noisy_particle = (
                diverse_illuminated_sample
                >> noise
                >> dt.NormalizeMinMax()
                >> dt.MoveAxis(2, 0)
                >> dt.pytorch.ToTensor(dtype=torch.float)
            )
            diverse_pip = diverse_noisy_particle & diverse_clean_particle

            # First resolve
            diverse_noisy_first, diverse_clean_first = diverse_pip()

            # Idempotent without update()
            diverse_noisy_first_2, diverse_clean_first_2 = diverse_pip()
            torch.testing.assert_close(diverse_clean_first_2,
                                       diverse_clean_first,
                                       rtol=1e-7, atol=1e-4)
            torch.testing.assert_close(diverse_noisy_first_2,
                                       diverse_noisy_first,
                                       rtol=1e-7, atol=1e-4)

            # After update(), BOTH should change (geometry + noise)
            diverse_noisy_second, diverse_clean_second = \
                diverse_pip.update().resolve()
            self.assertFalse(torch.allclose(
                diverse_clean_second, diverse_clean_first, rtol=1e-7, atol=1e-4
            ))
            self.assertFalse(torch.allclose(
                diverse_noisy_second, diverse_noisy_first, rtol=1e-7, atol=1e-4
            ))

    def test_4_A(self):
        pass


if __name__ == "__main__":
    unittest.main()
