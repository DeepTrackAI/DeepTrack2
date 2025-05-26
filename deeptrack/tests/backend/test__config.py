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

    def test__Proxy_get_int_dtype(self):

        from array_api_compat import numpy as apc_np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)

        # Test default int dtype (NumPy)
        dtype_default = xp.get_int_dtype()
        self.assertIn(dtype_default, ("int64", "numpy.int64"))

        # Test explicit int32
        dtype_32 = xp.get_int_dtype("int32")
        self.assertIn(dtype_32, ("int32", "numpy.int32"))

        # Test explicit int64
        dtype_64 = xp.get_int_dtype("int64")
        self.assertIn(dtype_64, ("int64", "numpy.int64"))

        if _config.TORCH_AVAILABLE:
            from array_api_compat import torch as apc_torch

            xp.set_backend(apc_torch)

            # Test default int dtype (PyTorch)
            dtype_default = xp.get_int_dtype()
            self.assertIn(
                str(dtype_default),
                ("int64", "torch.int64"),
            )

            # Test explicit int32
            dtype_32 = xp.get_int_dtype("int32")
            self.assertIn(
                str(dtype_32),
                ("int32", "torch.int32"),
            )

            # Switch back to NumPy
            xp.set_backend(apc_np)
            dtype_default = xp.get_int_dtype()
            self.assertIn(dtype_default, ("int64", "numpy.int64"))

    def test__Proxy_get_complex_dtype(self):

        from array_api_compat import numpy as apc_np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)

        # Test default complex dtype (NumPy)
        dtype_default = xp.get_complex_dtype()
        self.assertIn(dtype_default, ("complex128", "numpy.complex128"))

        # Test explicit complex64
        dtype_64 = xp.get_complex_dtype("complex64")
        self.assertIn(dtype_64, ("complex64", "numpy.complex64"))

        # Test explicit complex128
        dtype_128 = xp.get_complex_dtype("complex128")
        self.assertIn(dtype_128, ("complex128", "numpy.complex128"))

        if _config.TORCH_AVAILABLE:
            from array_api_compat import torch as apc_torch

            xp.set_backend(apc_torch)

            # Test default complex dtype (PyTorch)
            dtype_default = xp.get_complex_dtype()
            self.assertIn(
                str(dtype_default),
                ("complex64", "torch.complex64"),
            )

            # Test explicit complex64
            dtype_64 = xp.get_complex_dtype("complex64")
            self.assertIn(
                str(dtype_64),
                ("complex64", "torch.complex64"),
            )

            # Switch back to NumPy
            xp.set_backend(apc_np)
            dtype_default = xp.get_complex_dtype()
            self.assertIn(dtype_default, ("complex128", "numpy.complex128"))

    def test__Proxy_get_bool_dtype(self):
        from array_api_compat import numpy as apc_np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)

        # Test default bool dtype (NumPy)
        dtype_default = xp.get_bool_dtype()
        self.assertIn(dtype_default, ("bool", "numpy.bool_"))

        if _config.TORCH_AVAILABLE:
            from array_api_compat import torch as apc_torch

            xp.set_backend(apc_torch)

            # Test default bool dtype (PyTorch)
            dtype_default = xp.get_bool_dtype()
            self.assertIn(
                str(dtype_default),
                ("bool", "torch.bool"),
            )

            # Switch back to NumPy
            xp.set_backend(apc_np)
            dtype_default = xp.get_bool_dtype()
            self.assertIn(dtype_default, ("bool", "numpy.bool_"))


if __name__ == "__main__":
    unittest.main()
