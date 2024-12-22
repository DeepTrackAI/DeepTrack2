import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from deeptrack import features
from deeptrack import units as u
from .. import optics

from ..scatterers import PointParticle, Sphere
from ..image import Image


import numpy as np


class TestOptics(unittest.TestCase):
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
            aberration=None,
        )
        scatterer = PointParticle(
            intensity=100,  # Squared magnitude of the field.
            position_unit="pixel",  # Units of position (default meter)
            position=(32, 32),  # Position of the particle
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (64, 64, 1))

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
            aberration=None,
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_IlluminationGradient(self):
        illumination_gradient = optics.IlluminationGradient(gradient=(5e-5, 5e-5))
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            aberration=None,
            illumination=illumination_gradient,
        )
        scatterer = PointParticle(
            refractive_index=1.45 + 0.1j,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = microscope(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, np.ndarray)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_upscale_fluorescence(self):
        microscope = optics.Brightfield(
            NA=0.7,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=5,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            aberration=None,
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

        output_image_2x_upscale = imaged_scatterer.update()(upscale=(2, 2, 2))

        self.assertEqual(output_image_no_upscale.shape, (64, 64, 1))
        self.assertEqual(output_image_2x_upscale.shape, (64, 64, 1))
        # Ensure the upscaled image is almost the same as the original image

        error = np.abs(
            output_image_2x_upscale - output_image_no_upscale
        ).mean()  # Mean absolute error
        self.assertLess(error, 0.01)

    def test_upscale_brightfield(self):
        microscope = optics.Fluorescence(
            NA=0.5,
            wavelength=660e-9,
            resolution=1e-6,
            magnification=10,
            refractive_index_medium=1.33,
            upscale=2,
            output_region=(0, 0, 64, 64),
            padding=(10, 10, 10, 10),
            aberration=None,
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

        error = np.abs(
            output_image_2x_upscale - output_image_no_upscale
        ).mean()  # Mean absolute error
        self.assertLess(error, 0.01)


if __name__ == "__main__":
    unittest.main()