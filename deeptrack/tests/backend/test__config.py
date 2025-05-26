# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

from deeptrack.backend import _config


class TestCore(unittest.TestCase):

    def test__Proxy_set_backend(self):

        from array_api_compat import numpy as apc_np
        import numpy as np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)
        array = xp.arange(5)
        self.assertIsInstance(array, np.ndarray)

        if _config.TORCH_AVAILABLE:
            from array_api_compat import torch as apc_torch
            import torch

            xp.set_backend(apc_torch)
            array = xp.arange(5)
            self.assertIsInstance(array, torch.Tensor)

            # Switch bact to NumPy.
            xp.set_backend(apc_np)
            array = xp.arange(5)
            self.assertIsInstance(array, np.ndarray)

    def test__Proxy_get_float_dtype(self):

        from array_api_compat import numpy as apc_np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)

        # Test default float dtype (NumPy)
        dtype_default = xp.get_float_dtype()
        self.assertIn(
            dtype_default,
            ("float64", "numpy.float64"),  # API compat may return either
        )

        # Test explicit float32
        dtype_32 = xp.get_float_dtype("float32")
        self.assertIn(dtype_32, ("float32", "numpy.float32"))

        # Test explicit float64
        dtype_32 = xp.get_float_dtype("float64")
        self.assertIn(dtype_32, ("float64", "numpy.float64"))

        if _config.TORCH_AVAILABLE:
            from array_api_compat import torch as apc_torch

            xp.set_backend(apc_torch)

            # Test default float dtype (PyTorch)
            dtype_default = xp.get_float_dtype()
            self.assertIn(
                str(dtype_default),
                ("float32", "torch.float32"),
            )

            # Test explicit float32
            dtype_32 = xp.get_float_dtype("float32")
            self.assertIn(
                str(dtype_32),
                ("float32", "torch.float32"),
            )

            # Switch bact to NumPy.
            xp.set_backend(apc_np)

            dtype_default = xp.get_float_dtype()
            self.assertIn(dtype_default, ("float64", "numpy.float64"))


if __name__ == "__main__":
    unittest.main()
