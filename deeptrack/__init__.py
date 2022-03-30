# flake8: noqa
from pint import UnitRegistry, Context
from .backend.pint_definition import pint_definitions

units = UnitRegistry(pint_definitions.split("\n"))
units.enable_contexts("dt")

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


from .features import *
from .aberrations import *
from .augmentations import *

from .math import *
from .noises import *
from .optics import *
from .scatterers import *
from .sequences import *
from .elementwise import *
from .statistics import *
from .holography import *

from .image import strip

from . import (
    image,
    losses,
    generators,
    models,
    utils,
    layers,
    backend,
    test,
    visualization,
)
