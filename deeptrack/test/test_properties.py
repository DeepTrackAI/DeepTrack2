import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import properties, features, optics

import numpy as np


class TestProperties(unittest.TestCase):
    def test_Property_constant(self):
        P = properties.Property(1)
        self.assertEqual(P(), 1)
        P._update()
        self.assertEqual(P(), 1)

    def test_Property_iter(self):
        P = properties.Property(iter([1, 2, 3, 4, 5]))
        self.assertEqual(P(), 1)
        for i in range(1, 5):
            self.assertEqual(P(), i)
            P._update()

    def test_Property_random(self):
        P = properties.Property(lambda: np.random.rand())
        for _ in range(100):
            P._update()
            self.assertTrue(P() >= 0 and P() <= 1)

    def test_PropertyDict(self):
        property_dict = properties.PropertyDict(
            P1=properties.Property(1),
            P2=properties.Property(iter([1, 2, 3, 4, 5])),
            P3=properties.Property(lambda: np.random.rand()),
        )
        current_value_dict = property_dict()
        self.assertEqual(current_value_dict["P1"], 1)
        self.assertEqual(current_value_dict["P2"], 1)
        self.assertTrue(current_value_dict["P3"] >= 0 and current_value_dict["P3"] <= 1)
        for i in range(1, 100):
            current_value_dict = property_dict()
            self.assertEqual(current_value_dict["P1"], 1)
            self.assertEqual(current_value_dict["P2"], np.min((i, 5)))
            self.assertTrue(
                current_value_dict["P3"] >= 0 and current_value_dict["P3"] <= 1
            )
            property_dict._update()

    def test_AcceptsReplicateIndex(self):

        prop = properties.Property(lambda _ID: _ID)
        self.assertEqual(prop(), ())


if __name__ == "__main__":
    unittest.main()