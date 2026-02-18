# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import numpy as np

from deeptrack import TORCH_AVAILABLE, utils


if TORCH_AVAILABLE:
    import torch


class DummyClass:
    def method(self):
        pass

    def __len__(self):
        return 42


class TestUtils(unittest.TestCase):

    def test_hasmethod(self):
        self.assertTrue(utils.hasmethod(utils, "hasmethod"))
        self.assertFalse(utils.hasmethod(utils, "not_a_method"))
        self.assertTrue(utils.hasmethod(DummyClass, "method"))
        self.assertFalse(utils.hasmethod(DummyClass, "not_real"))
        self.assertTrue(utils.hasmethod(DummyClass(), "method"))
        self.assertTrue(utils.hasmethod(DummyClass(), "__len__"))
        self.assertFalse(utils.hasmethod(123, "foo"))  # int has no foo

        # Built-in edge cases
        self.assertTrue(utils.hasmethod([], "append"))
        self.assertFalse(utils.hasmethod([], "not_a_real_method"))

    def test_as_list(self):
        # Scalars
        self.assertEqual(utils.as_list(1), [1])
        self.assertEqual(utils.as_list(None), [None])
        self.assertEqual(utils.as_list(3.14), [3.14])

        # Containers
        self.assertEqual(utils.as_list([1, 2]), [1, 2])
        self.assertEqual(utils.as_list((1, 2)), [1, 2])
        self.assertEqual(sorted(utils.as_list({1, 2})), [1, 2])

        # Generator
        gen = (i for i in range(2))
        self.assertEqual(utils.as_list(gen), [0, 1])

        # Strings and bytes
        self.assertEqual(utils.as_list("abc"), ["abc"])
        self.assertEqual(utils.as_list(b"123"), [b"123"])

        # Numpy array
        arr = np.array([1, 2, 3])
        result = utils.as_list(arr)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(all(isinstance(x, (int, np.generic)) for x in result))

        if TORCH_AVAILABLE:
            tensor = torch.tensor([[1, 2], [3, 4]])
            result = utils.as_list(tensor)

            # By default, this will be [tensor([1, 2]), tensor([3, 4])]
            self.assertEqual(len(result), 2)
            self.assertTrue(all(isinstance(x, torch.Tensor) for x in result))

    def test_get_kwarg_names(self):
        def f1():
            pass

        self.assertEqual(utils.get_kwarg_names(f1), [])

        def f2(a):
            pass

        self.assertEqual(utils.get_kwarg_names(f2), ["a"])

        def f3(a, b=1):
            pass

        self.assertEqual(utils.get_kwarg_names(f3), ["a", "b"])

        def f4(a, *args, b=2):
            pass

        self.assertEqual(utils.get_kwarg_names(f4), ["b"])

        def f5(*args, b, c=2):
            pass

        self.assertEqual(utils.get_kwarg_names(f5), ["b", "c"])

        def f6(a, b, *args):
            pass

        self.assertEqual(utils.get_kwarg_names(f6), [])

        def f7(a, b=1, c=3, **kwargs):
            pass

        self.assertEqual(utils.get_kwarg_names(f7), ["a", "b", "c"])

        # Built-in function (should not raise)
        self.assertIsInstance(utils.get_kwarg_names(len), list)

        # Lambda
        l = lambda a, b=2: a + b
        self.assertEqual(utils.get_kwarg_names(l), ["a", "b"])

        # Method
        self.assertIn("self", utils.get_kwarg_names(DummyClass.method))

    def test_kwarg_has_default(self):
        def f1(a, b=2):
            pass

        self.assertFalse(utils.kwarg_has_default(f1, "a"))
        self.assertTrue(utils.kwarg_has_default(f1, "b"))

        # Not in function
        self.assertFalse(utils.kwarg_has_default(f1, "c"))

    def test_safe_call(self):
        def f(a, b=2, c=3):
            return a + b + c

        # All args present
        self.assertEqual(utils.safe_call(f, positional_args=[1], b=2, c=3), 6)
        # Only some kwargs present
        self.assertEqual(utils.safe_call(f, positional_args=[1], b=4), 8)
        # No kwargs
        self.assertEqual(utils.safe_call(f, positional_args=[1]), 6)
        # Extra kwargs are ignored
        self.assertEqual(utils.safe_call(f, positional_args=[1], b=5, x=10), 9)
        # Only kwargs
        self.assertEqual(utils.safe_call(f, a=1, b=2, c=3), 6)

        # Should ignore kwargs not in function signature
        def g(a):
            return a

        self.assertEqual(utils.safe_call(g, a=42, extrakw=1), 42)

        # Missing required arg should raise error
        def h(a):
            return a

        with self.assertRaises(TypeError):
            utils.safe_call(h)

        def k(a, *, b):
            return a + b

        with self.assertRaises(TypeError):
            utils.safe_call(k, a=1)  # Missing b


if __name__ == "__main__":
    unittest.main()
