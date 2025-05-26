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

        import sys

        from array_api_compat import numpy as apc_np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)

        # Test default bool dtype (NumPy)
        dtype_default = xp.get_bool_dtype()
        if sys.version_info >= (3, 10):
            self.assertIn(
                dtype_default,
                ("bool", "numpy.bool", "numpy.bool_"),
            )

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
            if sys.version_info >= (3, 10):
                self.assertIn(
                    dtype_default,
                    ("bool", "numpy.bool", "numpy.bool_"),
                )

    def test__Proxy___getattr__(self):

        from array_api_compat import numpy as apc_np
        import numpy as np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)

        # The proxy should forward .arange to NumPy's arange
        arange = xp.arange(3)
        self.assertIsInstance(arange, np.ndarray)
        self.assertEqual(arange.shape, (3,))

        # The proxy should forward .ones to NumPy's ones
        ones = xp.ones((2, 2))
        self.assertIsInstance(ones, np.ndarray)
        self.assertEqual(ones.shape, (2, 2))

        if _config.TORCH_AVAILABLE:
            from array_api_compat import torch as apc_torch
            import torch

            xp.set_backend(apc_torch)
            arr = xp.arange(3)

            # The proxy should forward .arange to PyTorch's arange
            arange = xp.arange(3)
            self.assertIsInstance(arange, torch.Tensor)
            self.assertEqual(arange.shape, (3,))

            # The proxy should forward .ones to PyTorch's ones
            ones = xp.ones((2, 2))
            self.assertIsInstance(ones, torch.Tensor)
            self.assertEqual(ones.shape, (2, 2))

            # Switch back to NumPy and test again
            xp.set_backend(apc_np)
            arange = xp.arange(3)
            self.assertIsInstance(arange, np.ndarray)
            self.assertEqual(arange.shape, (3,))

    def test__Proxy___dir__(self):

        from array_api_compat import numpy as apc_np

        xp = _config._Proxy("numpy")
        xp.set_backend(apc_np)

        attrs_numpy = dir(xp)
        self.assertIsInstance(attrs_numpy, list)
        # These attributes should be present for NumPy backend
        self.assertIn("arange", attrs_numpy)
        self.assertIn("ones", attrs_numpy)

        if _config.TORCH_AVAILABLE:
            from array_api_compat import torch as apc_torch

            xp.set_backend(apc_torch)

            attrs_torch = dir(xp)
            self.assertIsInstance(attrs_torch, list)
            # These attributes should be present for NumPy backend
            self.assertIn("arange", attrs_torch)
            self.assertIn("ones", attrs_torch)


if __name__ == "__main__":
    unittest.main()
