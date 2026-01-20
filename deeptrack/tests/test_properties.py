# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack import properties, TORCH_AVAILABLE
from deeptrack.backend.core import DeepTrackNode


if TORCH_AVAILABLE:
    import torch


class TestProperties(unittest.TestCase):

    def test___all__(self):
        from deeptrack import (
            Property,
            PropertyDict,
            SequentialProperty,
        )
        from deeptrack.properties import (
            Property,
            PropertyDict,
            SequentialProperty,
        )


    def test_Property_constant_list_nparray_tensor(self):
        P = properties.Property(42)
        self.assertEqual(P(), 42)
        self.assertEqual(P.new(), 42)

        P = properties.Property((1, 2, 3))
        self.assertEqual(P(), (1, 2, 3))
        self.assertEqual(P.new(), (1, 2, 3))

        P = properties.Property(np.array([1, 2, 3]))
        np.testing.assert_array_equal(P(), np.array([1, 2, 3]))
        np.testing.assert_array_equal(P.new(), np.array([1, 2, 3]))

        if TORCH_AVAILABLE:
            P = properties.Property(torch.Tensor([1, 2, 3]))
            self.assertTrue(torch.equal(P(), torch.tensor([1, 2, 3])))
            self.assertTrue(torch.equal(P.new(), torch.tensor([1, 2, 3])))

    def test_Property_function(self):

        # Lambda function.
        P = properties.Property(lambda x: x * 2, x=properties.Property(10))
        self.assertEqual(P(), 20)
        self.assertEqual(P.new(), 20)

        # Function.
        def func1(x):
            return 2 * x

        P = properties.Property(func1, x=properties.Property(10))
        self.assertEqual(P(), 20)
        self.assertEqual(P.new(), 20)

        # Lambda function with randomness.
        P = properties.Property(lambda: np.random.rand())
        for _ in range(10):
            self.assertEqual(P.new(), P())
            self.assertTrue(P() >= 0 and P() <= 1)

        # Function with randomness.
        def func2(x):
            return 2 * x

        P = properties.Property(
            func2,
            x=properties.Property(lambda: np.random.rand()),
        )
        for _ in range(10):
            self.assertEqual(P.new(), P())
            self.assertTrue(P() >= 0 and P() <= 2)

    def test_Property_slice(self):
        P = properties.Property(slice(1, lambda: 10, properties.Property(2)))
        result = P()
        self.assertEqual(result.start, 1)
        self.assertEqual(result.stop, 10)
        self.assertEqual(result.step, 2)
        result = P.new()
        self.assertEqual(result.start, 1)
        self.assertEqual(result.stop, 10)
        self.assertEqual(result.step, 2)

    def test_Property_iterable(self):
        P = properties.Property(iter([1, 2, 3]))

        self.assertEqual(P(), 1)
        self.assertEqual(P.new(), 2)
        self.assertEqual(P.new(), 3)
        self.assertEqual(P.new(), 3)  # Last value repeats indefinitely

        # Edge case with empty iterable.
        P = properties.Property(iter([]))
        self.assertIsNone(P())
        self.assertIsNone(P.new())
        self.assertIsNone(P.new())

        # Iterator nested in a list.
        P = properties.Property([iter([1, 2]), iter([3])])
        self.assertEqual(P(), [1, 3])
        self.assertEqual(P.new(), [2, 3])
        self.assertEqual(P.new(), [2, 3])

        # Iterator nested in a dict.
        P = properties.Property({"a": iter([1, 2]), "b": iter([3])})
        self.assertEqual(P(), {"a": 1, "b": 3})
        self.assertEqual(P.new(), {"a": 2, "b": 3})
        self.assertEqual(P.new(), {"a": 2, "b": 3})

        # Iterator nested in a tuple.
        P = properties.Property((iter([1, 2]), iter([3]), 0))
        self.assertEqual(P(), (1, 3, 0))
        self.assertEqual(P.new(), (2, 3, 0))
        self.assertEqual(P.new(), (2, 3, 0))

    def test_Property_list(self):
        P = properties.Property([1, lambda: 2, properties.Property(3)])
        self.assertEqual(P(), [1, 2, 3])
        self.assertEqual(P.new(), [1, 2, 3])

        P = properties.Property(
            [
                lambda _ID=(): 1 * np.random.rand(),
                lambda: 2 * np.random.rand(),
                properties.Property(lambda _ID=(): 3 * np.random.rand()),
            ]
        )
        for _ in range(10):
            self.assertEqual(P.new(), P())
            self.assertTrue(P()[0] >= 0 and P()[0] <= 1)
            self.assertTrue(P()[1] >= 0 and P()[1] <= 2)
            self.assertTrue(P()[2] >= 0 and P()[2] <= 3)

    def test_Property_dict(self):
        P = properties.Property(
            {
                "a": 1, 
                "b": lambda: 2, 
                "c": properties.Property(3),
            }
        )
        self.assertEqual(P(), {"a": 1, "b": 2, "c": 3})
        self.assertEqual(P.new(), {"a": 1, "b": 2, "c": 3})

        P = properties.Property(
            {
                "a": lambda _ID=(): 1 * np.random.rand(),
                "b": lambda: 2 * np.random.rand(),
                "c": properties.Property(lambda _ID=(): 3 * np.random.rand()),
            }
        )
        for _ in range(10):
            self.assertEqual(P.new(), P())
            self.assertTrue(P()["a"] >= 0 and P()["a"] <= 1)
            self.assertTrue(P()["b"] >= 0 and P()["b"] <= 2)
            self.assertTrue(P()["c"] >= 0 and P()["c"] <= 3)

    def test_Property_tuple(self):
        P = properties.Property((1, lambda: 2, properties.Property(3)))
        self.assertEqual(P(), (1, 2, 3))
        self.assertEqual(P.new(), (1, 2, 3))

        P = properties.Property(
            (
                lambda _ID=(): 1 * np.random.rand(),
                lambda: 2 * np.random.rand(),
                properties.Property(lambda _ID=(): 3 * np.random.rand()),
            )
        )
        for _ in range(10):
            self.assertEqual(P.new(), P())
            self.assertTrue(P()[0] >= 0 and P()[0] <= 1)
            self.assertTrue(P()[1] >= 0 and P()[1] <= 2)
            self.assertTrue(P()[2] >= 0 and P()[2] <= 3)

    def test_Property_DeepTrackNode(self):
        node = DeepTrackNode(100)
        P = properties.Property(node)
        self.assertEqual(P(), 100)
        self.assertEqual(P.new(), 100)

        node = DeepTrackNode(lambda _ID=(): np.random.rand())
        P = properties.Property(node)
        for _ in range(10):
            self.assertEqual(P.new(), P())
            self.assertTrue(P() >= 0 and P() <= 1)

    def test_Property_ID(self):
        P = properties.Property(lambda _ID: _ID)
        self.assertEqual(P(), ())

        P = properties.Property(lambda _ID: _ID)
        self.assertEqual(P((1,)), (1,))

        P = properties.Property(lambda _ID: _ID)
        self.assertEqual(P((1, 2, 3)), (1, 2, 3))

        # _ID propagation in list containers.
        P = properties.Property([lambda _ID: _ID, 0])
        self.assertEqual(P((1, 2)), [(1, 2), 0])

        # _ID propagation in dict containers.
        P = properties.Property({"a": lambda _ID: _ID, "b": 0})
        self.assertEqual(P((3,)), {"a": (3,), "b": 0})

        # _ID propagation in tuple containers.
        P = properties.Property((lambda _ID: _ID, 0))
        self.assertEqual(P((4, 5)), ((4, 5), 0))

    def test_Property_combined(self):
        P = properties.Property(
            {
                "constant": 42,
                "list": [1, lambda: 2, properties.Property(3)],
                "dict": {"a": properties.Property(1), "b": lambda: 2},
                "function": lambda x, y: x * y,
                "slice": slice(1, lambda: 10, properties.Property(2)),
            },
            x=properties.Property(5),
            y=properties.Property(3),
        )

        result = P()
        self.assertEqual(result["constant"], 42)
        self.assertEqual(result["list"], [1, 2, 3])
        self.assertEqual(result["dict"], {"a": 1, "b": 2})
        self.assertEqual(result["function"], 15)
        self.assertEqual(result["slice"].start, 1)
        self.assertEqual(result["slice"].stop, 10)
        self.assertEqual(result["slice"].step, 2)

    def test_Property_dependency_callable(self):
        # Callable with named dependency is tracked.
        d1 = properties.Property(0.5)
        P = properties.Property(lambda d1: d1 + 1, d1=d1)
        _ = P()  # Trigger evaluation to ensure child edges exist.
        self.assertIn(P, d1.recurse_children())

        # Closure dependency is NOT tracked (expected behavior).
        d1 = properties.Property(0.5)
        P = properties.Property(lambda: d1() + 1)
        _ = P()
        self.assertNotIn(P, d1.recurse_children())

        # Kwarg filtering: unused dependencies are ignored.
        x = properties.Property(1)
        y = properties.Property(2)
        P = properties.Property(lambda x: x + 1, x=x, y=y)
        self.assertEqual(P(), 2)
        self.assertNotIn(P, y.recurse_children())
        self.assertIn(P, x.recurse_children())


    def test_PropertyDict_basics(self):

        PD = properties.PropertyDict(
            constant=42,
            random=lambda: np.random.rand(),
            dependent=lambda constant: constant + 1,
        )

        self.assertIn("constant", PD)
        self.assertIn("constant", PD())
        self.assertIn("random", PD)
        self.assertIn("random", PD())
        self.assertIn("dependent", PD)
        self.assertIn("dependent", PD())

        self.assertIsInstance(PD["constant"], properties.Property)
        self.assertEqual(PD["constant"](), 42)
        self.assertEqual(PD()["constant"], 42)

        self.assertIsInstance(PD["random"], properties.Property)
        self.assertTrue(0 <= PD["random"]() <= 1)
        self.assertTrue(0 <= PD()["random"] <= 1)

        self.assertIsInstance(PD["dependent"], properties.Property)
        self.assertEqual(PD["dependent"](), 43)
        self.assertEqual(PD()["dependent"], 43)

        # Basic dict behavior checks
        PD = properties.PropertyDict(a=1, b=2)
        self.assertEqual(len(PD), 2)
        self.assertEqual(set(PD.keys()), {"a", "b"})
        self.assertEqual(set(PD().keys()), {"a", "b"})        

        # Test that dependency resolution works regardless of kwarg order
        PD = properties.PropertyDict(
            dependent=lambda constant: constant + 1,
            random=lambda: np.random.rand(),
            constant=42,
        )
        self.assertEqual(PD["constant"](), 42)
        self.assertEqual(PD["dependent"](), 43)

        # Test that values are cached until .new() / .update()
        PD = properties.PropertyDict(
            random=lambda: np.random.rand(),
        )

        for _ in range(10):
            self.assertEqual(PD.new()["random"], PD()["random"])
            self.assertTrue(0 <= PD()["random"] <= 1)

    def test_PropertyDict_missing_dependency_raises_on_call(self):
        PD = properties.PropertyDict(dependent=lambda missing: missing + 1)
        with self.assertRaises(TypeError):
            _ = PD()["dependent"]

    def test_PropertyDict_ID_propagation(self):
        # Case len(_ID) == 2
        PD = properties.PropertyDict(
            id_val=lambda _ID: _ID,
            first=lambda _ID: _ID[0] if _ID else None,
            second=lambda _ID: _ID[1] if _ID and len(_ID) >= 2 else None,
            constant=1,
        )

        self.assertEqual(PD((1, 2))["id_val"], (1, 2))
        self.assertEqual(PD((1, 2))["first"], 1)
        self.assertEqual(PD((1, 2))["second"], 2)
        self.assertEqual(PD((1, 2))["constant"], 1)

        # Case len(_ID) == 1
        PD = properties.PropertyDict(
            id_val=lambda _ID: _ID,
            first=lambda _ID: _ID[0] if _ID else None,
            second=lambda _ID: _ID[1] if _ID and len(_ID) >= 2 else None,
            constant=1,
        )

        self.assertEqual(PD((1,))["id_val"], (1,))
        self.assertEqual(PD((1,))["first"], 1)
        self.assertEqual(PD((1,))["second"], None)
        self.assertEqual(PD((1,))["constant"], 1)

        # Case len(_ID) == 0
        PD = properties.PropertyDict(
            id_val=lambda _ID: _ID,
            first=lambda _ID: _ID[0] if _ID else None,
            second=lambda _ID: _ID[1] if _ID and len(_ID) >= 2 else None,
            constant=1,
        )

        self.assertEqual(PD()["id_val"], ())
        self.assertEqual(PD()["first"], None)
        self.assertEqual(PD()["second"], None)
        self.assertEqual(PD()["constant"], 1)


    def test_SequentialProperty_init(self):
        # Test basic initialization and children/dependencies
        sp = properties.SequentialProperty()

        self.assertEqual(sp.sequence_length(), 0)
        self.assertEqual(sp.sequence_index(), 0)
        self.assertEqual(sp.sequence(), [])
        self.assertEqual(sp.previous_values(), [])
        self.assertEqual(sp.previous_value(), None)
        self.assertEqual(sp.initial_sampling_rule, None)
        self.assertEqual(sp.sample(), None)

        self.assertEqual(sp(), None)

        self.assertEqual(len(sp.recurse_children()), 1)
        self.assertEqual(len(sp.recurse_dependencies()), 5)

        self.assertEqual(len(sp.sequence_length.recurse_children()), 2)
        self.assertEqual(len(sp.sequence_length.recurse_dependencies()), 1)

        self.assertEqual(len(sp.sequence_index.recurse_children()), 4)
        self.assertEqual(len(sp.sequence_index.recurse_dependencies()), 1)

        self.assertEqual(len(sp.previous_value.recurse_children()), 2)
        self.assertEqual(len(sp.previous_value.recurse_dependencies()), 2)

        self.assertEqual(len(sp.previous_values.recurse_children()), 2)
        self.assertEqual(len(sp.previous_values.recurse_dependencies()), 2)

        # Test basic initialization and children/dependencies with parameters
        sp = properties.SequentialProperty(
            initial_sampling_rule=1,
            sampling_rule=lambda sequence_index: sequence_index * 10,
            sequence_length=5,
        )

        self.assertEqual(sp.sequence_length(), 5)
        self.assertEqual(sp.sequence_index(), 0)
        self.assertEqual(sp.sequence(), [])
        self.assertEqual(sp.previous_values(), [])
        self.assertEqual(sp.previous_value(), None)
        self.assertEqual(sp.initial_sampling_rule(), 1)
        self.assertEqual(sp.sample(), 0)

        self.assertEqual(sp(), 1)
        self.assertEqual(sp(), 1)
        self.assertTrue(sp.next_step())
        self.assertEqual(sp(), 10)
        self.assertEqual(sp(), 10)
        self.assertTrue(sp.next_step())
        self.assertEqual(sp(), 20)
        self.assertEqual(sp(), 20)

        self.assertEqual(len(sp.recurse_children()), 1)
        self.assertEqual(len(sp.recurse_dependencies()), 5)

        self.assertEqual(len(sp.sequence_length.recurse_children()), 2)
        self.assertEqual(len(sp.sequence_length.recurse_dependencies()), 1)

        self.assertEqual(len(sp.sequence_index.recurse_children()), 4)
        self.assertEqual(len(sp.sequence_index.recurse_dependencies()), 1)

        self.assertEqual(len(sp.previous_value.recurse_children()), 2)
        self.assertEqual(len(sp.previous_value.recurse_dependencies()), 2)

        self.assertEqual(len(sp.previous_values.recurse_children()), 2)
        self.assertEqual(len(sp.previous_values.recurse_dependencies()), 2)

    def test_SequentialProperty_full_run(self):
        # Test full run: generate a complete sequence and verify history.
        sp = properties.SequentialProperty(
            initial_sampling_rule=1,
            sampling_rule=lambda previous_value: previous_value + 1,
            sequence_length=10,
        )

        expected = list(range(1, 11))

        for step in range(sp.sequence_length()):
            self.assertEqual(sp(), expected[step])

            advanced = sp.next_step()

            if step < sp.sequence_length() - 1:
                self.assertTrue(advanced)
                self.assertEqual(sp.sequence_index(), step + 1)
                self.assertEqual(len(sp.sequence()), step + 1)
            else:
                # Final step: cannot advance further.
                self.assertFalse(advanced)
                self.assertEqual(sp.sequence_index(), step)

        self.assertEqual(len(sp.sequence()), sp.sequence_length())
        self.assertEqual(sp.sequence(), expected)
        self.assertEqual(sp.previous_value(), expected[-2])
        self.assertEqual(sp.previous_values(), expected[:-2])
        self.assertEqual(sp.sequence_index(), sp.sequence_length() - 1)

        # Test no sampling_rule but initial_sampling_rule exists.
        sp = properties.SequentialProperty(
            initial_sampling_rule=7,
            sampling_rule=None,
            sequence_length=3,
        )

        self.assertEqual(sp(), 7)
        self.assertTrue(sp.next_step())
        self.assertIsNone(sp())
        self.assertTrue(sp.next_step())
        self.assertIsNone(sp())
        self.assertFalse(sp.next_step())

    def test_SequentialProperty_error_in_current_value(self):
        # Test error path in current_value()
        sp = properties.SequentialProperty(
            initial_sampling_rule=1,
            sampling_rule=lambda previous_value: previous_value + 1,
            sequence_length=3,
        )

        # No calls yet, so history is empty, but index is 0.
        with self.assertRaises(IndexError):
            sp.current_value()

        # Then after one evaluation:
        sp()
        self.assertEqual(sp.current_value(), 1)

    # Test _ID
    # TODO add test using _ID


if __name__ == "__main__":
    unittest.main()
