import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.properties as properties
import deeptrack as dt

import numpy as np


class TestProperties(unittest.TestCase):
    def test_Property_constant(self):
        P = properties.Property(1)
        self.assertEqual(P.current_value, 1)
        P.update()
        self.assertEqual(P.current_value, 1)

    def test_Property_iter(self):
        P = properties.Property(iter([1, 2, 3, 4, 5]))
        self.assertEqual(P.current_value, 1)
        for i in range(1, 5):
            self.assertEqual(P.current_value, i)
            P.update()

    def test_Property_random(self):
        P = properties.Property(lambda: np.random.rand())
        for _ in range(100):
            P.update()
            self.assertTrue(P.current_value >= 0 and P.current_value <= 1)

    def test_SequentialProperty_Constant(self):
        sampling_rule = 1
        P = properties.SequentialProperty(sampling_rule, initializer=0)
        P.update(sequence_length=10)
        self.assertEqual(P.current_value, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_SequentialProperty_timedependent(self):
        def steady_increase(previous_value):
            return previous_value + 1

        sampling_rule = steady_increase
        P = properties.SequentialProperty(sampling_rule, initializer=2)
        P.update(sequence_length=5)
        self.assertEqual(P.current_value, [3, 4, 5, 6, 7])

    def test_SequentialProperty_dependence_on_SequentialProperty(self):
        def steady_increase(previous_value):
            return previous_value + 1

        P1 = properties.SequentialProperty(steady_increase, initializer=1)

        def geometric_increase(previous_value, step_length):
            return previous_value + step_length

        P2 = properties.SequentialProperty(geometric_increase, initializer=1)

        P2.update(sequence_length=5, step_length=P1)
        self.assertEqual(P2.current_value, [3, 6, 10, 15, 21])

    def test_UpdateFromParent(self):
        optics = dt.Fluorescence()
        sample = dt.DummyFeature(voxel_size=optics.voxel_size)
        together = optics(sample)
        together.update()
        self.assertEqual(
            tuple(optics.voxel_size.current_value),
            tuple(sample.voxel_size.current_value),
        )

    def test_PropertyDict(self):
        property_dict = properties.PropertyDict(
            P1=properties.Property(1),
            P2=properties.Property(iter([1, 2, 3, 4, 5])),
            P3=properties.Property(lambda: np.random.rand()),
        )
        current_value_dict = property_dict.current_value_dict()
        self.assertEqual(current_value_dict["P1"], 1)
        self.assertEqual(current_value_dict["P2"], 1)
        self.assertTrue(current_value_dict["P3"] >= 0 and current_value_dict["P3"] <= 1)
        for i in range(1, 100):
            current_value_dict = property_dict.current_value_dict(is_resolving=True)
            self.assertEqual(current_value_dict["P1"], 1)
            self.assertEqual(current_value_dict["P2"], np.min((i, 5)))
            self.assertTrue(
                current_value_dict["P3"] >= 0 and current_value_dict["P3"] <= 1
            )
            property_dict.update()


if __name__ == "__main__":
    unittest.main()