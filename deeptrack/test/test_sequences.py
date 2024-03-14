import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from matplotlib import pyplot

from .. import sequences

from ..optics import Fluorescence
from ..scatterers import Ellipse
import numpy as np


class TestSequences(unittest.TestCase):
    def test_Sequence(self):
        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=(16, 16),
            intensity=1,
            radius=(1.5e-6, 1e-6),
            rotation=0,  # This will be the value at time 0.
            upsample=2,
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 2 * np.pi / sequence_length

        rotating_ellipse = sequences.Sequential(ellipse, rotation=get_rotation)
        imaged_rotating_ellipse = optics(rotating_ellipse)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse, sequence_length=5
        )
        imaged_rotating_ellipse_sequence.store_properties()

        self.assertIsInstance(imaged_rotating_ellipse_sequence, sequences.Sequence)

        outputs = imaged_rotating_ellipse_sequence()

        for i, out in enumerate(outputs):

            self.assertAlmostEqual(out.get_property("rotation"), 2 * i * np.pi / 5)

    def test_Dependent_Sequential(self):

        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=(16, 16),
            radius=(1.5e-6, 1e-6),
            rotation=0,  # This will be the value at time 0.
            upsample=2,
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 2 * np.pi / sequence_length

        def get_intensity(rotation):
            return rotation * 2

        rotating_ellipse = sequences.Sequential(
            ellipse, rotation=get_rotation, intensity=get_intensity
        )
        imaged_rotating_ellipse = optics(rotating_ellipse)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse, sequence_length=5
        )
        imaged_rotating_ellipse_sequence.store_properties()

        self.assertIsInstance(imaged_rotating_ellipse_sequence, sequences.Sequence)

        outputs = imaged_rotating_ellipse_sequence()

        for i, out in enumerate(outputs):
            self.assertAlmostEqual(out.get_property("rotation"), 2 * i * np.pi / 5)
            self.assertAlmostEqual(out.get_property("intensity"), 4 * i * np.pi / 5)

    def test_RepeatedParticle(self):

        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=lambda: np.random.randn(2) * 4 + (16, 16),
            radius=(1.5e-6, 1e-6),
            rotation=0,  # This will be the value at time 0.
            upsample=2,
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 2 * np.pi / sequence_length

        def get_intensity(rotation):
            return rotation * 2

        rotating_ellipse = sequences.Sequential(
            ellipse, rotation=get_rotation, intensity=get_intensity
        )
        imaged_rotating_ellipse = optics(rotating_ellipse ^ 2)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse, sequence_length=5
        )
        imaged_rotating_ellipse_sequence.store_properties()

        self.assertIsInstance(imaged_rotating_ellipse_sequence, sequences.Sequence)
        imaged_rotating_ellipse_sequence.update()
        outputs = imaged_rotating_ellipse_sequence()

        for i, out in enumerate(outputs):
            rotations = out.get_property("rotation", get_one=False)
            intensity = out.get_property("intensity", get_one=False)
            positions = out.get_property("position", get_one=False)
            self.assertEqual(len(rotations), 2)
            self.assertEqual(len(intensity), 2)
            self.assertEqual(len(positions), 2)
            self.assertAlmostEqual(rotations[0], 2 * i * np.pi / 5)
            self.assertAlmostEqual(rotations[1], 2 * i * np.pi / 5)
            self.assertAlmostEqual(intensity[0], 4 * i * np.pi / 5)
            self.assertAlmostEqual(intensity[1], 4 * i * np.pi / 5)

            self.assertNotEqual(positions[0][0], positions[1][0])
            self.assertNotEqual(positions[0][1], positions[1][1])

    def test_DistributedRepeatedParticle(self):

        positions = [(16, 25), (15, 24)]
        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=lambda _ID: positions[_ID[-1]],
            radius=(1.5e-6, 1e-6),
            rotation=0,  # This will be the value at time 0.
            upsample=2,
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 2 * np.pi / sequence_length

        def get_intensity(rotation):
            return rotation * 2

        rotating_ellipse = sequences.Sequential(
            ellipse, rotation=get_rotation, intensity=get_intensity
        )
        imaged_rotating_ellipse = optics(rotating_ellipse ^ 2)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse, sequence_length=5
        )
        imaged_rotating_ellipse_sequence.store_properties()

        self.assertIsInstance(imaged_rotating_ellipse_sequence, sequences.Sequence)
        imaged_rotating_ellipse_sequence.update()
        outputs = imaged_rotating_ellipse_sequence()

        for i, out in enumerate(outputs):
            rotations = out.get_property("rotation", get_one=False)
            intensity = out.get_property("intensity", get_one=False)
            p_positions = out.get_property("position", get_one=False)
            self.assertEqual(len(rotations), 2)
            self.assertEqual(len(intensity), 2)
            self.assertEqual(len(positions), 2)
            self.assertAlmostEqual(rotations[0], 2 * i * np.pi / 5)
            self.assertAlmostEqual(rotations[1], 2 * i * np.pi / 5)
            self.assertAlmostEqual(intensity[0], 4 * i * np.pi / 5)
            self.assertAlmostEqual(intensity[1], 4 * i * np.pi / 5)

            self.assertSequenceEqual(list(p_positions[0]), list(positions[0]))
            self.assertSequenceEqual(list(p_positions[1]), list(positions[1]))


if __name__ == "__main__":
    unittest.main()