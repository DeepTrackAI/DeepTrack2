import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.sequences as sequences

from deeptrack.optics import Fluorescence
from deeptrack.scatterers import Ellipse
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
            imaged_rotating_ellipse, sequence_length=50
        )
        self.assertIsInstance(imaged_rotating_ellipse_sequence, sequences.Sequence)


if __name__ == "__main__":
    unittest.main()