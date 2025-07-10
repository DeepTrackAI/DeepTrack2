# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack.sources import base
from deeptrack import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class TestBase(unittest.TestCase):

    def test___all__(self):
        from deeptrack.sources import (
            Source,
            SourceItem,
            Product,
            Subset,
            Sources,
            Join,
            random_split,
        )

    def test_SourceItem(self):

        called = []

        def callback(item):
            called.append(item)

        # List input
        item_list = base.SourceItem(callbacks=[callback], a=[1, 2], b=[3, 4])
        self.assertEqual(item_list["a"], [1, 2])
        self.assertEqual(item_list["b"], [3, 4])
        returned = item_list()
        self.assertIn(item_list, called)
        self.assertIs(returned, item_list)
        self.assertIn("callback", repr(item_list))

        # Tuple input
        called.clear()
        item_tuple = base.SourceItem(callbacks=[callback], a=(1, 2), b=(3, 4))
        self.assertEqual(item_tuple["a"], (1, 2))
        self.assertEqual(item_tuple["b"], (3, 4))
        returned = item_tuple()
        self.assertIn(item_tuple, called)

        # NumPy array input
        called.clear()
        a_np = np.array([1, 2])
        b_np = np.array([3, 4])
        item_np = base.SourceItem(callbacks=[callback], a=a_np, b=b_np)
        np.testing.assert_array_equal(item_np["a"], a_np)
        np.testing.assert_array_equal(item_np["b"], b_np)
        returned = item_np()
        self.assertIn(item_np, called)

        if TORCH_AVAILABLE:
            called.clear()
            a_torch = torch.tensor([1, 2])
            b_torch = torch.tensor([3, 4])
            item_torch = base.SourceItem(
                callbacks=[callback], a=a_torch, b=b_torch,
            )
            self.assertTrue(torch.equal(item_torch["a"], a_torch))
            self.assertTrue(torch.equal(item_torch["b"], b_torch))
            returned = item_torch()
            self.assertIn(item_torch, called)

    def test_Source(self):
        # Prepare test data
        data_variants = {
            "list": ([1, 2, 3], [10, 20, 30]),
            "tuple": ((1, 2, 3), (10, 20, 30)),
            "numpy": (np.array([1, 2, 3]), np.array([10, 20, 30])),
        }

        if TORCH_AVAILABLE:
            import torch
            data_variants["torch"] = (
                torch.tensor([1, 2, 3]),
                torch.tensor([10, 20, 30]),
            )

        for name, (a, b) in data_variants.items():
            with self.subTest(dtype=name):
                source = base.Source(a=a, b=b)

                # Test length
                self.assertEqual(len(source), 3)

                # Test indexing
                item = source[1]
                self.assertEqual(item["a"], a[1])
                self.assertEqual(item["b"], b[1])

                # Test iteration
                items = list(source)
                self.assertEqual(len(items), 3)
                self.assertEqual(items[2]["a"], a[2])
                self.assertEqual(items[2]["b"], b[2])

                # Test slice
                sliced = source[1:3]
                self.assertEqual(len(sliced), 2)
                self.assertEqual(sliced[0]["a"], a[1])
                self.assertEqual(sliced[1]["b"], b[2])

                # Test dynamic field
                source.set_index(0)
                self.assertEqual(source.a(), a[0])
                self.assertEqual(source.b(), b[0])

    def test_Product(self):
        pass

    def test_Subset(self):
        pass

    def test_Sources(self):
        pass

    def test_Join(self):
        pass

    def test_random_split(self):
        pass


if __name__ == "__main__":
    unittest.main()
