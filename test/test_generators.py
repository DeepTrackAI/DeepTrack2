import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.generators as generators

from deeptrack.optics import Fluorescence
from deeptrack.scatterers import PointParticle
import numpy as np


class TestGenerators(unittest.TestCase):
    def test_Generator(self):
        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 128, 128),
        )
        scatterer = PointParticle(
            intensity=100,
            position_unit="pixel",
            position=lambda: np.random.rand(2) * 128,
        )
        imaged_scatterer = optics(scatterer)

        def get_particle_position(result):
            for property in result.properties:
                if "position" in property:
                    return property["position"]

        generator = generators.Generator()
        particle_generator = generator.generate(
            imaged_scatterer, get_particle_position, batch_size=4
        )
        particles, positions = next(particle_generator)
        for particle, position in zip(particles, positions):
            self.assertEqual(particle.shape, (128, 128, 1))
            self.assertTrue(np.all(position >= 0))
            self.assertTrue(np.all(position <= 128))


if __name__ == "__main__":
    unittest.main()