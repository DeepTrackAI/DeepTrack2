# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack import features, sequences, TORCH_AVAILABLE
from deeptrack.optics import Fluorescence
from deeptrack.scatterers import Ellipse


if TORCH_AVAILABLE:
    import torch


class TestSequences(unittest.TestCase):

    def test___all__(self):
        from deeptrack import Sequence

    def test_Sequence__negative_sequence_length_raises(self):
        class Dummy(features.Feature):
            __distributed__ = False

            def get(self, input_list, **kwargs):
                return 1

        seq = sequences.Sequence(Dummy(), sequence_length=1)

        with self.assertRaises(ValueError):
            seq.get([], sequence_length=-1)

    def test_Sequence__zero_sequence_length_returns_empty(self):
        class Dummy(features.Feature):
            __distributed__ = False

            def get(self, input_list, **kwargs):
                return 1

        seq = sequences.Sequence(Dummy(), sequence_length=0)
        out = seq()

        self.assertEqual(out, [])

    def test_Sequence__to_sequential_increments(self):
        class PositionFeature(features.Feature):
            __distributed__ = False

            def __init__(self, position, **kwargs):
                super().__init__(position=position, **kwargs)

            def get(self, input_list, position, **kwargs):
                return position

        def increment(previous_value):
            if previous_value is None:
                return 0
            return previous_value + 1

        feature = PositionFeature(position=0)
        feature.to_sequential(position=increment)

        seq = sequences.Sequence(feature, sequence_length=5)
        out = seq()

        self.assertEqual(out, [0, 1, 2, 3, 4])

    def test_Sequence__tuple_outputs_are_transposed(self):
        class PairFeature(features.Feature):
            __distributed__ = False

            def __init__(self, a, b, **kwargs):
                super().__init__(a=a, b=b, **kwargs)

            def get(self, input_list, a, b, **kwargs):
                return a, b

        def inc_a(previous_value):
            if previous_value is None:
                return 0
            return previous_value + 1

        def inc_b(previous_value):
            if previous_value is None:
                return 10
            return previous_value + 10

        feature = PairFeature(a=0, b=10)
        feature.to_sequential(a=inc_a, b=inc_b)

        seq = sequences.Sequence(feature, sequence_length=3)
        out = seq()

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [0, 1, 2])
        self.assertEqual(out[1], [10, 20, 30])

    def test_Sequence__ID_isolation(self):
        class PositionFeature(features.Feature):
            __distributed__ = False

            def __init__(self, position, **kwargs):
                super().__init__(position=position, **kwargs)

            def get(self, input_list, position, **kwargs):
                return position

        def increment(previous_value):
            if previous_value is None:
                return 0
            return previous_value + 1

        feature = PositionFeature(position=0)
        feature.to_sequential(position=increment)

        seq = sequences.Sequence(feature, sequence_length=3)

        out_1 = seq(_ID=(1,))
        out_2 = seq(_ID=(2,))

        self.assertEqual(out_1, [0, 1, 2])
        self.assertEqual(out_2, [0, 1, 2])

    def test_Sequence_with_optics(self):

        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=(16, 16),
            intensity=1,
            radius=(1.5e-6, 1e-6),
            rotation=0,  # Value at time 0
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 1 / sequence_length

        rotating_ellipse = ellipse.to_sequential(rotation=get_rotation)
        imaged_rotating_ellipse = optics(rotating_ellipse)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse,
            sequence_length=5,
        )

        self.assertIsInstance(
            imaged_rotating_ellipse_sequence, sequences.Sequence
        )

        outputs = imaged_rotating_ellipse_sequence()

        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 5)

        for frame in outputs:
            frame_array = np.asarray(frame)
            self.assertGreaterEqual(len(frame_array.shape), 2)
            self.assertEqual(frame_array.shape[0], 32)
            self.assertEqual(frame_array.shape[1], 32)

        rotation_prop = rotating_ellipse.properties["rotation"]
        rotation_sequence = rotation_prop.sequence()

        np.testing.assert_allclose(
            rotation_sequence,
            [0.0, 0.2, 0.4, 0.6, 0.8],
            rtol=1e-7,
            atol=1e-12,
        )

    def test_Sequence_with_dependent(self):

        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=(16, 16),
            radius=(1.5e-6, 1e-6),
            rotation=0,  # Value at time 0
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 1 / sequence_length

        def get_intensity(rotation):
            return rotation * 2

        rotating_ellipse = ellipse.to_sequential(
            rotation=get_rotation,
            intensity=get_intensity,
        )

        imaged_rotating_ellipse = optics(rotating_ellipse)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse,
            sequence_length=5,
        )

        self.assertIsInstance(
            imaged_rotating_ellipse_sequence, sequences.Sequence
        )

        outputs = imaged_rotating_ellipse_sequence()

        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 5)

        frame_sums = []
        for frame in outputs:
            frame_array = np.asarray(frame)
            self.assertGreaterEqual(len(frame_array.shape), 2)
            self.assertEqual(frame_array.shape[0], 32)
            self.assertEqual(frame_array.shape[1], 32)
            frame_sums.append(float(np.sum(frame_array)))

        rotation_prop = rotating_ellipse.properties["rotation"]
        intensity_prop = rotating_ellipse.properties["intensity"]

        np.testing.assert_allclose(
            rotation_prop.sequence(),
            [0.0, 0.2, 0.4, 0.6, 0.8],
            rtol=1e-7,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            intensity_prop.sequence(),
            [0.0, 0.4, 0.8, 1.2, 1.6],
            rtol=1e-7,
            atol=1e-12,
        )

        for prev_sum, next_sum in zip(frame_sums, frame_sums[1:]):
            self.assertLess(prev_sum, next_sum)

    def test_Sequence_with_repeated_particle(self):

        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=lambda: np.random.randn(2) * 4 + (16, 16),
            radius=(1.5e-6, 1e-6),
            rotation=0,  # Value at time 0
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 1 / sequence_length

        def get_intensity(rotation):
            return rotation * 2

        rotating_ellipse = ellipse.to_sequential(
            rotation=get_rotation,
            intensity=get_intensity,
        )

        imaged_rotating_ellipse = optics(rotating_ellipse ^ 2)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse,
            sequence_length=5,
        )

        self.assertIsInstance(
            imaged_rotating_ellipse_sequence, sequences.Sequence
        )

        imaged_rotating_ellipse_sequence.update()
        outputs_1 = imaged_rotating_ellipse_sequence()

        self.assertIsInstance(outputs_1, list)
        self.assertEqual(len(outputs_1), 5)

        sums_1: list[float] = []
        for frame in outputs_1:
            frame_array = np.asarray(frame)
            self.assertGreaterEqual(len(frame_array.shape), 2)
            self.assertEqual(frame_array.shape[0], 32)
            self.assertEqual(frame_array.shape[1], 32)
            sums_1.append(float(np.sum(frame_array)))

        rotation_prop = rotating_ellipse.properties["rotation"]
        intensity_prop = rotating_ellipse.properties["intensity"]

        for _ID in range(2):
            np.testing.assert_allclose(
                rotation_prop.sequence(_ID=(_ID,)),
                [0.0, 0.2, 0.4, 0.6, 0.8],
                rtol=1e-7,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                intensity_prop.sequence(_ID=(_ID,)),
                [0.0, 0.4, 0.8, 1.2, 1.6],
                rtol=1e-7,
                atol=1e-12,
            )

        # Pixel sum should increase over the sequence
        # because intensity increases.
        for prev_sum, next_sum in zip(sums_1, sums_1[1:]):
            self.assertLess(prev_sum, next_sum)

        # Calling again without update should yield identical results
        # (no resample).
        outputs_2 = imaged_rotating_ellipse_sequence()
        self.assertEqual(len(outputs_2), 5)

        for frame_1, frame_2 in zip(outputs_1, outputs_2):
            np.testing.assert_allclose(
                np.asarray(frame_1),
                np.asarray(frame_2),
                rtol=0.0,
                atol=0.0,
            )

    def test_Sequence_with_distributed_repeated_particle(self):

        positions = [(16, 25), (15, 24)]
        optics = Fluorescence(
            output_region=(0, 0, 32, 32),
        )
        ellipse = Ellipse(
            position_unit="pixel",
            position=lambda _ID: positions[_ID[-1]],
            radius=(1.5e-6, 1e-6),
            rotation=0,  # Value at time 0
        )

        def get_rotation(sequence_length, previous_value):
            return previous_value + 1 / sequence_length

        def get_intensity(rotation):
            return rotation * 2

        rotating_ellipse = ellipse.to_sequential(
            rotation=get_rotation,
            intensity=get_intensity,
        )

        imaged_rotating_ellipse = optics(rotating_ellipse ^ 2)
        imaged_rotating_ellipse_sequence = sequences.Sequence(
            imaged_rotating_ellipse,
            sequence_length=5,
        )

        self.assertIsInstance(
            imaged_rotating_ellipse_sequence, sequences.Sequence
        )

        imaged_rotating_ellipse_sequence.update()
        outputs = imaged_rotating_ellipse_sequence()


if __name__ == "__main__":
    unittest.main()
