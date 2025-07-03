from deeptrack import TORCH_AVAILABLE

if TORCH_AVAILABLE:
  from deeptrack.backend.array_api_compat_ext.torch import random


__all__ = ["random"]
