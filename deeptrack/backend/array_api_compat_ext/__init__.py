# TODO: would be better to delay the import of torch until it is actually used
from array_api_compat import torch as apctorch
from .torch import random

# numpy and torch random functions are incompatible with each other.
# The current array_api_compat module does to fix this incompatibility.
# So we implement our own patch, which implements a numpy-compatible interface
# for the torch random functions.
apctorch.random = random
