# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

from deeptrack.backend._config import TORCH_AVAILABLE
from deeptrack.backend.core import DeepTrackNode
from deeptrack.utils import get_kwarg_names

from numpy import array, testing_assert_array_equal
from numpy.random import rand

from deeptrack import properties


class TestProperties(unittest.TestCase):

    def test_Property_constant_list_nparray_tensor(self):
        P = properties.Property(42)
        self.assertEqual(P(), 42)
        P.update()
        self.assertEqual(P(), 42)

        P = properties.Property((1, 2, 3))
        self.assertEqual(P(), (1, 2, 3))
        P.update()
        self.assertEqual(P(), (1, 2, 3))

        P = properties.Property(array([1, 2, 3]))
        testing.assert_array_equal(P(), array([1, 2, 3]))
        P.update()
        testing.assert_array_equal(P(), array([1, 2, 3]))

        if TORCH_AVAILABLE:
            import torch

            P = properties.Property(torch.Tensor([1, 2, 3]))
            self.assertTrue(torch.equal(P(), torch.tensor([1, 2, 3])))
            P.update()
            self.assertTrue(torch.equal(P(), torch.tensor([1, 2, 3])))

    def test_Property_function(self):

        # Lambda function.
        P = properties.Property(lambda x: x * 2, x=properties.Property(10))
        self.assertEqual(P(), 20)
        P.update()
        self.assertEqual(P(), 20)

        # Function.
        def func1(x):
            return 2 * x

        P = properties.Property(func1, x=properties.Property(10))
        self.assertEqual(P(), 20)
        P.update()
        self.assertEqual(P(), 20)

        # Lambda function with randomness.
        P = properties.Property(lambda: rand())
        for _ in range(10):
            P.update()
            self.assertEqual(P(), P())
            self.assertTrue(P() >= 0 and P() <= 1)

        # Function with randomness.
        def func2(x):
            return 2 * x

        P = properties.Property(
            func2,
            x=properties.Property(lambda: rand()),
        )
        for _ in range(10):
            P.update()
            self.assertEqual(P(), P())
            self.assertTrue(P() >= 0 and P() <= 2)

    def test_Property_slice(self):
        P = properties.Property(slice(1, lambda: 10, properties.Property(2)))
        result = P()
        self.assertEqual(result.start, 1)
        self.assertEqual(result.stop, 10)
        self.assertEqual(result.step, 2)
        P.update()
        self.assertEqual(result.start, 1)
        self.assertEqual(result.stop, 10)
        self.assertEqual(result.step, 2)

    def test_Property_iterable(self):
        P = properties.Property(iter([1, 2, 3]))

        self.assertEqual(P(), 1)
        P.update()
        self.assertEqual(P(), 2)
        P.update()
        self.assertEqual(P(), 3)
        P.update()
        self.assertEqual(P(), 3)  # Last value repeats indefinitely

    def test_Property_list(self):
        P = properties.Property([1, lambda: 2, properties.Property(3)])
        self.assertEqual(P(), [1, 2, 3])
        P.update()
        self.assertEqual(P(), [1, 2, 3])

        P = properties.Property(
            [
                lambda _ID=(): 1 * rand(),
                lambda: 2 * rand(),
                properties.Property(lambda _ID=(): 3 * rand()),
            ]
        )
        for _ in range(10):
            P.update()
            self.assertEqual(P(), P())
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
        P.update()
        self.assertEqual(P(), {"a": 1, "b": 2, "c": 3})

        P = properties.Property(
            {
                "a": lambda _ID=(): 1 * rand(),
                "b": lambda: 2 * rand(),
                "c": properties.Property(lambda _ID=(): 3 * rand()),
            }
        )
        for _ in range(10):
            P.update()
            self.assertEqual(P(), P())
            self.assertTrue(P()["a"] >= 0 and P()["a"] <= 1)
            self.assertTrue(P()["b"] >= 0 and P()["b"] <= 2)
            self.assertTrue(P()["c"] >= 0 and P()["c"] <= 3)

    def test_Property_DeepTrackNode(self):
        node = DeepTrackNode(100)
        P = properties.Property(node)
        self.assertEqual(P(), 100)
        P.update()
        self.assertEqual(P(), 100)

        node = DeepTrackNode(lambda _ID=(): rand())
        P = properties.Property(node)
        for _ in range(10):
            P.update()
            self.assertEqual(P(), P())
            self.assertTrue(P() >= 0 and P() <= 1)

    def test_Property_ID(self):
        P = properties.Property(lambda _ID: _ID)
        self.assertEqual(P(), ())

        P = properties.Property(lambda _ID: _ID)
        self.assertEqual(P((1,)), (1,))

        P = properties.Property(lambda _ID: _ID)
        self.assertEqual(P((1, 2, 3)), (1, 2, 3))

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

    def test_PropertyDict(self):

        PD = properties.PropertyDict(
            constant=42,
            random=lambda: rand(),
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

    def test_SequentialProperty(self):
        SP = properties.SequentialProperty()
        SP.sequence_length.store(5)
        SP.sample = lambda _ID=(): SP.sequence_index() + 1

        for step in range(SP.sequence_length()):
            SP.sequence_index.store(step)
            current_value = SP.sample()
            SP.store(current_value)

            self.assertEqual(SP.data[()].current_value(),
                             list(range(1, step + 2)))

            SP.previous_value.invalidate()
            # print(SP.previous_value())

            SP.previous_values.invalidate()
            # print(SP.previous_values())

        self.assertEqual(SP.previous_value(), 4)
        self.assertEqual(SP.previous_values(),
                         list(range(1, SP.sequence_length() - 1)))


if __name__ == "__main__":
    unittest.main()
