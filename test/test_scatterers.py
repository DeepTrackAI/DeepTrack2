import sys

sys.path.append("..")  # Adds the module to path

import unittest

import deeptrack.scatterers as scatterers

import numpy as np
from deeptrack.optics import Fluorescence, Brightfield
from deeptrack.image import Image


class TestScatterers(unittest.TestCase):
    def test_PointParticle(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
        )
        scatterer = scatterers.PointParticle(
            intensity=100,
            position_unit="pixel",
            position=(32, 32),
        )
        imaged_scatterer = optics(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_Ellipse(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
        )
        scatterer = scatterers.Ellipse(
            intensity=100,
            position_unit="pixel",
            position=(32, 32),
            radius=(1e-6, 0.5e-6),
            rotation=np.pi / 4,
            upsample=4,
        )
        imaged_scatterer = optics(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_Sphere(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
        )
        scatterer = scatterers.Sphere(
            intensity=100,
            position_unit="pixel",
            position=(32, 32),
            radius=1e-6,
            upsample=4,
        )
        imaged_scatterer = optics(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_Ellipsoid(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 64, 64),
        )
        scatterer = scatterers.Ellipsoid(
            intensity=100,
            position_unit="pixel",
            position=(32, 32),
            radius=(1e-6, 0.5e-6, 0.25e-6),
            rotation=(np.pi / 4, 0, 0),
            upsample=4,
        )
        imaged_scatterer = optics(scatterer)
        output_image = imaged_scatterer.resolve()
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image.shape, (64, 64, 1))

    def test_MieSphere(self):
        optics_1 = Brightfield(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=5,
            output_region=(0, 0, 64, 64),
            padding=(128, 128, 128, 128),
            return_field=True,
            upscale=2,
        )
        scatterer = scatterers.MieSphere(radius=1e-6, refractive_index=1.45 + 0.1j)

        imaged_scatterer_1 = optics_1(scatterer)

        image_1 = imaged_scatterer_1.resolve()


if __name__ == "__main__":
    unittest.main()