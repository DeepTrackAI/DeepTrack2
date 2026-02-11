# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

import deeptrack as dt
from deeptrack import TORCH_AVAILABLE
from deeptrack.sources import base


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


    def test_SourceDeepTrackNode(self):
        pass


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
        data_variants = {
            "list": ([1, 2], [10, 20]),
            "tuple": ((1, 2), (10, 20)),
            "numpy": (np.array([1, 2]), np.array([10, 20])),
        }

        if TORCH_AVAILABLE:
            import torch
            data_variants["torch"] = (
                torch.tensor([1, 2]),
                torch.tensor([10, 20]),
            )

        for name, (a, b) in data_variants.items():
            with self.subTest(dtype=name):
                source = base.Source(a=a)
                product = base.Product(source, b=b)

                # Check length: 2 source × 2 b = 4
                self.assertEqual(len(product), 4)

                # Check content consistency
                expected_a = [a[0], a[0], a[1], a[1]]
                expected_b = [b[0], b[1], b[0], b[1]]

                for i, item in enumerate(product):
                    self.assertEqual(item["a"], expected_a[i])
                    self.assertEqual(item["b"], expected_b[i])
                    self.assertIsInstance(item, base.SourceItem)

        # Test Product without source (i.e., only kwargs)
        product = base.Product(x=[1, 2], y=[100, 200])
        self.assertEqual(len(product), 4)
        expected_pairs = [(1, 100), (1, 200), (2, 100), (2, 200)]
        for i, item in enumerate(product):
            self.assertEqual(item["x"], expected_pairs[i][0])
            self.assertEqual(item["y"], expected_pairs[i][1])

        # Test error on overlapping keys
        source = base.Source(x=[1, 2])
        with self.assertRaises(ValueError):
            base.Product(source, x=[10, 20])


    def test_Subset(self):
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
                indices = [0, 2]
                subset = base.Subset(source, indices)

                # Length
                self.assertEqual(len(subset), 2)

                # Items should match corresponding ones from original source
                for i, idx in enumerate(indices):
                    item = subset[i]
                    self.assertEqual(item["a"], a[idx])
                    self.assertEqual(item["b"], b[idx])

                # Iteration should return correct items
                for i, item in enumerate(subset):
                    self.assertEqual(item["a"], a[indices[i]])
                    self.assertEqual(item["b"], b[indices[i]])

                # Dynamic attribute access
                if TORCH_AVAILABLE and isinstance(a, torch.Tensor):
                    self.assertEqual(subset.a().item(), a[0].item())
                else:
                    self.assertEqual(subset.a(), a[0])


    def test_Sources(self):
        data_variants = {
            "list": ([1, 2], [10, 20], [3, 4], [30, 40]),
            "tuple": ((1, 2), (10, 20), (3, 4), (30, 40)),
            "numpy": (
                np.array([1, 2]), np.array([10, 20]),
                np.array([3, 4]), np.array([30, 40])
            ),
        }

        if TORCH_AVAILABLE:
            data_variants["torch"] = (
                torch.tensor([1, 2]), torch.tensor([10, 20]),
                torch.tensor([3, 4]), torch.tensor([30, 40])
            )

        for name, (a1, b1, a2, b2) in data_variants.items():
            with self.subTest(dtype=name):
                train = base.Source(a=a1, b=b1)
                val = base.Source(a=a2, b=b2)

                joined = base.Sources(train, val)

                # Verify dynamic fields exist and have callable values
                self.assertTrue(callable(joined.a))
                self.assertTrue(callable(joined.b))

                # Trigger update by activating an item
                item_train = train[0]
                item_val = val[1]

                item_train()
                self.assertEqual(joined.a(), a1[0])
                self.assertEqual(joined.b(), b1[0])

                item_val()
                self.assertEqual(joined.a(), a2[1])
                self.assertEqual(joined.b(), b2[1])

                # Feature access
                feature = dt.Value(joined.a) + dt.Value(joined.b)
                self.assertEqual(feature(train[0]), a1[0] + b1[0])
                self.assertEqual(feature(val[1]), a2[1] + b2[1])


    def test_random_split(self):
        data_variants = {
            "list": ([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]),
            "tuple": ((1, 2, 3, 4, 5), (10, 20, 30, 40, 50)),
            "numpy": (
                np.array([1, 2, 3, 4, 5]),
                np.array([10, 20, 30, 40, 50]),
            ),
        }

        if TORCH_AVAILABLE:
            data_variants["torch"] = (
                torch.tensor([1, 2, 3, 4, 5]),
                torch.tensor([10, 20, 30, 40, 50]),
            )

        for dtype, (a, b) in data_variants.items():
            with self.subTest(dtype=dtype):
                source = base.Source(a=a, b=b)

                # Test integer split
                train, val = base.random_split(source, [3, 2])
                self.assertEqual(len(train), 3)
                self.assertEqual(len(val), 2)

                train_indices = {item["a"] for item in train}
                val_indices = {item["a"] for item in val}
                self.assertTrue(train_indices.isdisjoint(val_indices))

                combined = sorted(train_indices | val_indices)
                expected = sorted(list(a))
                if TORCH_AVAILABLE and isinstance(a, torch.Tensor):
                    expected = expected  # torch.Tensor already sorted and list-like
                self.assertEqual(combined, expected)

                # Test fractional split
                splits = base.random_split(source, [0.4, 0.6])
                self.assertEqual(sum(len(s) for s in splits), 5)

                # Ensure all indices are unique and complete
                all_indices = set()
                for subset in splits:
                    for item in subset:
                        all_indices.add(item["a"])
                self.assertEqual(len(all_indices), 5)


if __name__ == "__main__":
    unittest.main()
