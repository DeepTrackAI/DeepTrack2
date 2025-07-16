# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

from deeptrack import units
from deeptrack import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class TestUnits(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
