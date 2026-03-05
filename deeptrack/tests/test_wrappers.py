import unittest
import numpy as np

from deeptrack import config, TORCH_AVAILABLE
from deeptrack import wrappers

if TORCH_AVAILABLE:
    import torch


class TestWrappers(unittest.TestCase):

    def test_Wrapper_arithmetic_and_structure(self):

        backends = ["numpy"]
        if TORCH_AVAILABLE:
            backends.append("torch")

        for backend in backends:

            config.set_backend(backend)

            arr = np.arange(16).reshape(4, 4).astype(np.float32)
            arr2 = np.ones((4, 4)).astype(np.float32)

            if backend == "torch":
                arr = torch.tensor(arr)
                arr2 = torch.tensor(arr2)

            props = {"position": (1, 2)}

            w = wrappers.Wrapper(array=arr, properties=props)
            w2 = wrappers.Wrapper(array=arr2)

            # scalar arithmetic
            r1 = w + 2
            r2 = 2 + w
            r3 = w - 1
            r4 = w * 3
            r5 = w / 2
            r6 = w // 2
            r7 = w ** 2

            for r in [r1, r2, r3, r4, r5, r6, r7]:
                self.assertIsInstance(r, wrappers.Wrapper)
                self.assertEqual(r.shape, w.shape)
                self.assertEqual(r.ndim, w.ndim)
                self.assertEqual(r.properties, w.properties)
                
            self.assertIsNot(r1, w)  # new object
            self.assertIsNot(r1.properties, w.properties)  # properties copied
            self.assertEqual(type(r1.array), type(arr))  # backend preserved

            # wrapper-wrapper arithmetic
            r8 = w + w2
            r9 = w * w2

            for r in [r8, r9]:
                self.assertIsInstance(r, wrappers.Wrapper)
                self.assertEqual(r.shape, w.shape)
                self.assertEqual(r.ndim, w.ndim)
                self.assertEqual(r.properties, w.properties)

            # comparisons
            r10 = w > 5
            r11 = w < 10
            r12 = w >= 3
            r13 = w <= 8

            for r in [r10, r11, r12, r13]:
                self.assertIsInstance(r, wrappers.Wrapper)
                self.assertEqual(r.shape, w.shape)

            # bitwise
            mask1 = wrappers.Wrapper(array=(arr > 5))
            mask2 = wrappers.Wrapper(array=(arr > 10))

            r14 = mask1 & mask2
            r15 = mask1 ^ mask2

            for r in [r14, r15]:
                self.assertIsInstance(r, wrappers.Wrapper)
                self.assertEqual(r.shape, mask1.shape)

            # original wrapper should not change
            if backend == "torch":
                self.assertTrue(torch.equal(w.array, arr))
            else:
                self.assertTrue(np.array_equal(w.array, arr))

            # numerical correctness
            if backend == "torch":
                self.assertTrue(torch.equal(r1.array, arr + 2))
                self.assertTrue(torch.equal(r8.array, arr + arr2))
            else:
                self.assertTrue(np.array_equal(r1.array, arr + 2))
                self.assertTrue(np.array_equal(r8.array, arr + arr2))

if __name__ == "__main__":
    unittest.main()