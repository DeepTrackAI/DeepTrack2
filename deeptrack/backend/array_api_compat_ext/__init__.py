from deeptrack import TORCH_AVAILABLE
if TORCH_AVAILABLE:
  from array_api_compat import torch as apctorch
  from deeptrack.backend.array_api_compat_ext.torch import random

# NumPy and PyTorch random functions are incompatible with each other.
# The current array_api_compat module does not fix this incompatibility.
# So we implement our own patch, which implements a numpy-compatible interface
# for the torch random functions.
  apctorch.random = random
