# flake8: noqa
from pint import UnitRegistry, Context

units = UnitRegistry()
_pixel_context = """
@context(pixel_size = 1) deeptrack = dt
    [printing_unit] -> [length]: value * pixel_size * (meter / pixel)
    [length] -> [printing_unit]: value / pixel_size / (meter / pixel)
@end
"""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

units.load_definitions(_pixel_context.split("\n"))
units.enable_contexts("dt")

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

from .image import array, strip

from . import image, losses, generators, models, utils, layers, backend, test
