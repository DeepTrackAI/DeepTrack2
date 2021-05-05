from pint import UnitRegistry, Context

units = UnitRegistry()
_pixel_context = """
@context(pixel_size = 1) deeptrack = dt
    [printing_unit] -> [length]: value * pixel_size * (meter / pixel)
    [length] -> [printing_unit]: value / pixel_size / (meter / pixel)
@end
"""


units.load_definitions(_pixel_context.split("\n"))
units.enable_contexts("dt")

from .aberrations import *
from .augmentations import *
from .features import *
from .math import *
from .noises import *
from .optics import *
from .scatterers import *
from .sequences import *
from .elementwise import *
from .statistics import *


from . import image, losses, generators, models, utils, layers, backend, test