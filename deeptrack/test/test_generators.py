import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import generators
from ..optics import Fluorescence
from ..scatterers import PointParticle
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

    def test_ContinuousGenerator(self):
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

        generator = generators.ContinuousGenerator(
            imaged_scatterer, get_particle_position, min_data_size=10, max_data_size=20
        )

        with generator:
            self.assertGreater(len(generator.data), 10)
            self.assertLess(len(generator.data), 21)

    def test_CappedContinuousGenerator(self):

        optics = Fluorescence(
            NA=0.7,
            wavelength=680e-9,
            resolution=1e-6,
            magnification=10,
            output_region=(0, 0, 128, 128),
            index=range(0, 200),
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

        generator = generators.ContinuousGenerator(
            imaged_scatterer,
            get_particle_position,
            batch_size=1,
            min_data_size=10,
            max_data_size=20,
            max_epochs_per_sample=5,
        )

        # with generator:
        #     self.assertGreater(len(generator.data), 10)
        #     self.assertLess(len(generator.data), 21)
        #     for _ in range(10):
        #         generator.on_epoch_end()
        #         for idx in range(len(generator)):
        #             a = generator[idx]

        #         [self.assertLess(d[-1], 8) for d in generator.data]


if __name__ == "__main__":
    unittest.main()