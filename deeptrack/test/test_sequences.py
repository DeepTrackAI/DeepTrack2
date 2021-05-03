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
            NA=0.6,
            magnification=10,
            resolution=1e-6,
            wavelength=633e-9,
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
        self.assertIsInstance(imaged_rotating_ellipse_sequence, sequences.Sequence)

        outputs = imaged_rotating_ellipse_sequence()

        for i, out in enumerate(outputs):

            self.assertAlmostEqual(out.get_property("rotation"), 2 * i * np.pi / 5)

    def test_Dependent_Sequential(self):

        optics = Fluorescence(
            NA=0.6,
            magnification=10,
            resolution=1e-6,
            wavelength=633e-9,
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
        self.assertIsInstance(imaged_rotating_ellipse_sequence, sequences.Sequence)

        outputs = imaged_rotating_ellipse_sequence()

        for i, out in enumerate(outputs):
            self.assertAlmostEqual(out.get_property("rotation"), 2 * i * np.pi / 5)
            self.assertAlmostEqual(out.get_property("intensity"), 4 * i * np.pi / 5)


if __name__ == "__main__":
    unittest.main()