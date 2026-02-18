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
        source = base.SourceDeepTrackNode(
            lambda: {"a": {"b": 1}, "_x": 2},
            node_name="root",
        )

        # Access returns nodes (not values) when not calling.
        node_a_1 = source.a
        self.assertIsInstance(node_a_1, base.SourceDeepTrackNode)
        self.assertEqual(node_a_1(), {"b": 1})

        node_b_1 = source.a.b
        self.assertIsInstance(node_b_1, base.SourceDeepTrackNode)
        self.assertEqual(node_b_1(), 1)

        # Child nodes are cached.
        node_a_2 = source.a
        self.assertIs(node_a_1, node_a_2)

        node_b_2 = source.a.b
        self.assertIs(node_b_1, node_b_2)

        # Node names use dotted paths when the parent is named.
        self.assertEqual(node_a_1.node_name, "root.a")
        self.assertEqual(node_b_1.node_name, "root.a.b")

        # Private/dunder-like names are rejected as data keys.
        with self.assertRaises(AttributeError):
            _ = source._x

        # Dependencies/children are registered.
        children = source.recurse_children()
        self.assertIn(source, children)
        self.assertIn(node_a_1, children)
        self.assertIn(node_b_1, children)
        self.assertEqual(len(children), 3)

        deps_a = node_a_1.recurse_dependencies()
        self.assertIn(node_a_1, deps_a)
        self.assertIn(source, deps_a)
        self.assertEqual(len(deps_a), 2)

        deps_b = node_b_1.recurse_dependencies()
        self.assertIn(node_b_1, deps_b)
        self.assertIn(node_a_1, deps_b)
        self.assertIn(source, deps_b)
        self.assertEqual(len(deps_b), 3)

    def test_SourceItem(self):
        called: list[base.SourceItem] = []

        def callback(item):
            called.append(item)

        callbacks = [callback]
        item = base.SourceItem(callbacks=callbacks, a=1, b=2)

        # Behaves like a dict
        self.assertEqual(item["a"], 1)
        self.assertEqual(item["b"], 2)

        # Calling triggers callback and returns self
        returned = item()
        self.assertIs(returned, item)
        self.assertEqual(called, [item])

        # __repr__ includes class name and callback count
        rep = repr(item)
        self.assertIn("SourceItem", rep)
        self.assertIn("1 callback(s)", rep)

        # Callbacks list is copied (no aliasing)
        callbacks.append(lambda x: None)
        called.clear()
        item()
        self.assertEqual(called, [item])

    def test_Source(self):
        # Prepare test data
        data_variants = {
            "list": ([1, 2, 3], [10, 20, 30]),
            "tuple": ((1, 2, 3), (10, 20, 30)),
            "numpy": (np.array([1, 2, 3]), np.array([10, 20, 30])),
        }

        if TORCH_AVAILABLE:
            data_variants["torch"] = (
                torch.tensor([1, 2, 3]),
                torch.tensor([10, 20, 30]),
            )

        for a, b in data_variants.values():
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
            self.assertEqual(sliced[0]["b"], b[1])
            self.assertEqual(sliced[1]["a"], a[2])
            self.assertEqual(sliced[1]["b"], b[2])

            # Test dynamic field
            source.set_index(0)
            self.assertEqual(source.a(), a[0])
            self.assertEqual(source.b(), b[0])

            # Activation updates current index and dynamic access
            source.set_index(0)
            item = source[2]
            self.assertEqual(source.a(), a[0])
            self.assertEqual(source.b(), b[0])

            item()
            self.assertEqual(source.a(), a[2])
            self.assertEqual(source.b(), b[2])

    def test_Product(self):
        data_variants = {
            "list": ([1, 2], [10, 20]),
            "tuple": ((1, 2), (10, 20)),
            "numpy": (np.array([1, 2]), np.array([10, 20])),
        }

        if TORCH_AVAILABLE:
            data_variants["torch"] = (
                torch.tensor([1, 2]),
                torch.tensor([10, 20]),
            )

        for a, b in data_variants.values():
            source = base.Source(a=a)
            product = base.Product(source, b=b)

            # Check length: 2 source × 2 b = 4
            self.assertEqual(len(product), 4)

            # Check content consistency
            expected_a = [a[0], a[0], a[1], a[1]]
            expected_b = [b[0], b[1], b[0], b[1]]

            for i, item in enumerate(product):
                self.assertIsInstance(item, base.SourceItem)
                self.assertEqual(item["a"], expected_a[i])
                self.assertEqual(item["b"], expected_b[i])

        # Test Product without source (i.e., only kwargs)
        product = base.Product(x=[1, 2], y=[100, 200])
        self.assertEqual(len(product), 4)
        expected_pairs = [(1, 100), (1, 200), (2, 100), (2, 200)]
        for i, item in enumerate(product):
            self.assertEqual(item["x"], expected_pairs[i][0])
            self.assertEqual(item["y"], expected_pairs[i][1])

        # Test empty base source yields empty product
        empty = base.Source(a=[])
        product = base.Product(empty, b=[10, 20])
        self.assertEqual(len(product), 0)
        self.assertEqual(list(product), [])

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
            data_variants["torch"] = (
                torch.tensor([1, 2, 3]),
                torch.tensor([10, 20, 30]),
            )

        for a, b in data_variants.values():
            source = base.Source(a=a, b=b)
            subset = base.Subset(source, [0, 2])

            # Provenance exposed
            self.assertIs(subset.source, source)
            self.assertEqual(subset.indices, [0, 2])

            # Length
            self.assertEqual(len(subset), 2)

            # Indexing
            item0 = subset[0]
            self.assertIsInstance(item0, base.SourceItem)
            self.assertEqual(item0["a"], a[0])
            self.assertEqual(item0["b"], b[0])

            item1 = subset[1]
            self.assertEqual(item1["a"], a[2])
            self.assertEqual(item1["b"], b[2])

            # Iteration
            items = list(subset)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["a"], a[0])
            self.assertEqual(items[1]["b"], b[2])

            # Dynamic behavior is independent of parent
            source.set_index(1)
            self.assertEqual(source.a(), a[1])

            subset.set_index(0)
            self.assertEqual(subset.a(), a[0])
            self.assertEqual(subset.b(), b[0])

            subset.set_index(1)
            self.assertEqual(subset.a(), a[2])
            self.assertEqual(subset.b(), b[2])

            # Negative index at construction
            subset_neg = base.Subset(source, [-1])
            self.assertEqual(len(subset_neg), 1)
            self.assertEqual(subset_neg[0]["a"], a[-1])
            self.assertEqual(subset_neg[0]["b"], b[-1])

            # Out-of-range index raises
            with self.assertRaises(IndexError):
                base.Subset(source, [100])

    def test_Sources(self):
        data_variants = {
            "list": ([1, 2], [10, 20], [3, 4], [30, 40]),
            "tuple": ((1, 2), (10, 20), (3, 4), (30, 40)),
            "numpy": (
                np.array([1, 2]),
                np.array([10, 20]),
                np.array([3, 4]),
                np.array([30, 40]),
            ),
        }

        if TORCH_AVAILABLE:
            data_variants["torch"] = (
                torch.tensor([1, 2]),
                torch.tensor([10, 20]),
                torch.tensor([3, 4]),
                torch.tensor([30, 40]),
            )

        for a1, b1, a2, b2 in data_variants.values():
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

        for a, b in data_variants.values():
            source = base.Source(a=a, b=b)

            expected = list(a)
            if hasattr(a, "tolist"):
                expected = a.tolist()

            # Integer split (3, 2)
            gen = np.random.default_rng(123)
            train, val = base.random_split(
                source,
                [3, 2],
                generator=gen,
            )

            self.assertEqual(len(train), 3)
            self.assertEqual(len(val), 2)

            train_a = {item["a"] for item in train}
            val_a = {item["a"] for item in val}

            self.assertTrue(train_a.isdisjoint(val_a))
            self.assertEqual(sorted(train_a | val_a), sorted(expected))

            # Fractional split (0.4, 0.6)
            gen = np.random.default_rng(123)
            splits = base.random_split(
                source,
                [0.4, 0.6],
                generator=gen,
            )

            self.assertEqual([len(s) for s in splits], [2, 3])

            all_a: list[int] = []
            for subset in splits:
                all_a.extend(item["a"] for item in subset)

            self.assertEqual(len(all_a), len(expected))
            self.assertEqual(len(set(all_a)), len(expected))
            self.assertEqual(sorted(all_a), sorted(expected))

        if TORCH_AVAILABLE:
            source = base.Source(
                a=[1, 2, 3, 4, 5],
                b=[10, 20, 30, 40, 50],
            )
            expected = [1, 2, 3, 4, 5]

            torch_gen = torch.Generator()
            torch_gen.manual_seed(123)

            train, val = base.random_split(
                source,
                [3, 2],
                generator=torch_gen,
            )

            self.assertEqual(len(train), 3)
            self.assertEqual(len(val), 2)

            train_a = {item["a"] for item in train}
            val_a = {item["a"] for item in val}

            self.assertTrue(train_a.isdisjoint(val_a))
            self.assertEqual(sorted(train_a | val_a), sorted(expected))

            torch_gen = torch.Generator()
            torch_gen.manual_seed(123)

            splits = base.random_split(
                source,
                [0.4, 0.6],
                generator=torch_gen,
            )

            self.assertEqual([len(s) for s in splits], [2, 3])

            all_a = []
            for subset in splits:
                all_a.extend(item["a"] for item in subset)

            self.assertEqual(len(all_a), len(expected))
            self.assertEqual(len(set(all_a)), len(expected))
            self.assertEqual(sorted(all_a), sorted(expected))

    def test__accumulate(self):
        # Default cumulative sum
        self.assertEqual(
            list(base._accumulate([1, 2, 3, 4, 5])),
            [1, 3, 6, 10, 15],
        )

        # Custom operator (multiplication)
        import operator

        self.assertEqual(
            list(base._accumulate([1, 2, 3, 4, 5], fn=operator.mul)),
            [1, 2, 6, 24, 120],
        )

        # Empty iterable
        self.assertEqual(
            list(base._accumulate([])),
            [],
        )

        # Single element
        self.assertEqual(
            list(base._accumulate([7])),
            [7],
        )

        # Ensure function is called expected number of times
        calls: list[tuple[int, int]] = []

        def fn(x: int, y: int) -> int:
            calls.append((x, y))
            return x + y

        self.assertEqual(
            list(base._accumulate([1, 2, 3, 4], fn=fn)),
            [1, 3, 6, 10],
        )

        self.assertEqual(
            calls,
            [(1, 2), (3, 3), (6, 4)],
        )


if __name__ == "__main__":
    unittest.main()
