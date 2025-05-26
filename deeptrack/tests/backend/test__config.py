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


if __name__ == "__main__":
    unittest.main()
