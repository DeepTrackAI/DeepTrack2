# TODO: would be better to delay the import of torch until it is actually used
from array_api_compat import torch as apctorch
from .torch import random

apctorch.random = random
