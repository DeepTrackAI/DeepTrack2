# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path

import unittest

import numpy as np

from deeptrack import optics
from deeptrack.scatterers import PointParticle, Sphere
from deeptrack import units_registry as u

from deeptrack.backend import TORCH_AVAILABLE, xp
from deeptrack.tests import BackendTestBase

if TORCH_AVAILABLE:
    import torch


class TestOptics_NumPy(BackendTestBase):
    BACKEND = "numpy"

    @property
    def array_type(self):
        if self.BACKEND == "numpy":
            return np.ndarray
        elif self.BACKEND == "torch":
            return torch.Tensor
        else:
            raise ValueError(f"Unsupported backend: {self.BACKEND}")

    def test_Microscope(self):
        microscope_type = optics.Fluorescence()
        scatterer = PointParticle(intensity=100)
        microscope = optics.Microscope(
            sample=scatterer,
            objective=microscope_type,
        )
        output_image = microscope.get(None)
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (128, 128, 1))

    def test_Optics(self):
        microscope = optics.Optics()
        scatterer = PointParticle()
        image = microscope(scatterer)
        self.assertIsInstance(image, optics.Microscope)

    def test_Fluorescence(self):
        microscope = optics.Fluorescence(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            padding=(10, 10, 10, 10),
            output_region=(0, 0, 64, 64),
        )
        scatterer = PointParticle(
            intensity=100,
            position_unit="pixel",
            position=(32, 32),
        )
        output_image = microscope(scatterer).resolve()

        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))
        self.assertEqual(microscope.NA(), 0.7)

        img = output_image[..., 0]
        peak = np.unravel_index(int(xp.argmax(img)), img.shape)
        self.assertLessEqual(abs(peak[0] - 32), 1)
        self.assertLessEqual(abs(peak[1] - 32), 1)

    def test_Brightfield(self):
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_Holography(self):
        microscope = optics.Holography(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_Brightfield_Holography_equivalence(self):
        bf = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        hg = optics.Holography(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )

        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )

        img_bf = bf(scatterer).resolve()
        img_hg = hg(scatterer).resolve()

        err = float(xp.mean(xp.abs(img_bf - img_hg)))
        self.assertLess(err, 1e-10)

    def test_ISCAT(self):
        microscope = optics.ISCAT(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertEqual(microscope.illumination_angle(), 3.141592653589793)
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))
        self.assertEqual(microscope.amp_factor(), 1)

    def test_Darkfield(self):
        microscope = optics.Darkfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertEqual(microscope.illumination_angle(), 1.5707963267948966)
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_IlluminationGradient(self):
        illumination_gradient = optics.IlluminationGradient(
            gradient=(5e-5, 5e-5)
        )
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            illumination=illumination_gradient,
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, self.array_type)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_upscale_Brightfield(self):
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=5,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = Sphere(
            refractive_index=1.45,
            radius=1e-6,
            z=2 * u.um,
            position_unit="pixel",
            position=(32, 32),
        )

        imaged_scatterer = microscope(scatterer)
        output_image_no_upscale = imaged_scatterer.update()(upscale=1)

        output_image_2x_upscale = imaged_scatterer.update()(upscale=(2, 2, 1))

        self.assertEqual(output_image_no_upscale.shape, (64, 64, 1))
        self.assertEqual(output_image_2x_upscale.shape, (64, 64, 1))
        # Ensure the upscaled image is almost the same as the original image

        rel_error = xp.abs(
            output_image_2x_upscale - output_image_no_upscale
        ).mean() / xp.mean(
            output_image_no_upscale
        )  # Mean relative error
        self.assertLess(rel_error, 0.1)

    def test_upscale_fluorescence(self):
        microscope = optics.Fluorescence(
            NA=0.5,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
        )
        scatterer = Sphere(
            intensity=100,
            radius=1e-6,
            z=2 * u.um,
            position_unit="pixel",
            position=(32, 32),
        )

        imaged_scatterer = microscope(scatterer)
        output_image_no_upscale = imaged_scatterer.update()(upscale=1)

        output_image_2x_upscale = imaged_scatterer.update()(upscale=(2, 2, 1))

        self.assertEqual(output_image_no_upscale.shape, (64, 64, 1))
        self.assertEqual(output_image_2x_upscale.shape, (64, 64, 1))
        # Ensure the upscaled image is almost the same as the original image

        rel_error = xp.abs(
            output_image_2x_upscale - output_image_no_upscale
        ).mean() / xp.mean(
            output_image_no_upscale
        )  # Mean relative error
        self.assertLess(rel_error, 0.1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed.")
class TestOptics_PyTorch(TestOptics_NumPy):
    BACKEND = "torch"


if __name__ == "__main__":
    unittest.main()
