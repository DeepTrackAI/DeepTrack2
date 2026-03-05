# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

from deeptrack import config, TORCH_AVAILABLE, xp
from deeptrack import wrappers


class TestWrappers(unittest.TestCase):

    def test___all__(self):
        from deeptrack import Wrapper

    def test_Wrapper_arithmetic_and_structure(self):

        backends = ["numpy", "torch"] if TORCH_AVAILABLE else ["numpy"]

        for backend in backends:
            old_backend = config.get_backend()
            try:
                with self.subTest(backend=backend):
                    config.set_backend(backend)

                    arr1 = xp.arange(16, dtype=xp.float32).reshape(4, 4)
                    arr2 = xp.ones((4, 4), dtype=xp.float32)

                    props = {"position": (1, 2)}

                    w1 = wrappers.Wrapper(array=arr1, properties=props)
                    w2 = wrappers.Wrapper(array=arr2)

                    # Scalar arithmetic
                    r1 = w1 + 2
                    r2 = 2 + w1
                    r3 = w1 - 1
                    r4 = w1 * 3
                    r5 = w1 / 2
                    r6 = w1 // 2
                    r7 = w1**2

                    for r in [r1, r2, r3, r4, r5, r6, r7]:
                        self.assertIsInstance(r, wrappers.Wrapper)
                        self.assertEqual(r.shape, w1.shape)
                        self.assertEqual(r.ndim, w1.ndim)
                        self.assertEqual(r.properties, w1.properties)

                    self.assertIsNot(r1, w1)  # New object
                    self.assertIsNot(
                        r1.properties, w1.properties
                    )  # Properties copied
                    self.assertEqual(
                        type(r1.array), type(arr1)
                    )  # Backend preserved

                    # Wrapper-wrapper arithmetic
                    r8 = w1 + w2
                    r9 = w1 * w2

                    for r in [r8, r9]:
                        self.assertIsInstance(r, wrappers.Wrapper)
                        self.assertEqual(r.shape, w1.shape)
                        self.assertEqual(r.ndim, w1.ndim)
                        self.assertEqual(r.properties, w1.properties)

                    # Comparisons
                    r10 = w1 > 5
                    r11 = w1 < 10
                    r12 = w1 >= 3
                    r13 = w1 <= 8

                    for r in [r10, r11, r12, r13]:
                        self.assertIsInstance(r, wrappers.Wrapper)
                        self.assertEqual(r.shape, w1.shape)

                    # Bitwise
                    mask1 = wrappers.Wrapper(array=(arr1 > 5))
                    mask2 = wrappers.Wrapper(array=(arr1 > 10))

                    r14 = mask1 & mask2
                    r15 = mask1 ^ mask2

                    for r in [r14, r15]:
                        self.assertIsInstance(r, wrappers.Wrapper)
                        self.assertEqual(r.shape, mask1.shape)

                    # Original wrapper should not change
                    self.assertTrue(bool(xp.all(w1.array == arr1)))

                    # Numerical correctness
                    self.assertTrue(bool(xp.all(r1.array == (arr1 + 2))))
                    self.assertTrue(bool(xp.all(r8.array == (arr1 + arr2))))

            finally:
                config.set_backend(old_backend)


if __name__ == "__main__":
    unittest.main()
