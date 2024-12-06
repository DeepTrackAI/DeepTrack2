# flake8: noqa
import lazy_import
from pint import UnitRegistry

'''units = UnitRegistry(pint_definitions.split("\n"))'''#TBE

# Create a UnitRegistry and add custom units.
units = UnitRegistry()
custom_units = [
    "pixel = 1 micrometer = px",  ### Can this be erased?
    "xpixel = 1 micrometer = xpx",  ### why these are defined as 1 um?
    "ypixel = 1 micrometer = ypx",
    "zpixel = 1 micrometer = zpx",
    "simulation_xpixel = 1 micrometer = sxpx",
    "simulation_ypixel = 1 micrometer = sypx",
    "simulation_zpixel = 1 micrometer = szpx"
]
for unit in custom_units:
    units.define(unit)

'''# Check if tensorflow is installed without importing it
import pkg_resources

installed = [pkg.key for pkg in pkg_resources.working_set]

if "tensorflow" in installed:
    HAS_TENSORFLOW = True
else:
    HAS_TENSORFLOW = False

if "torch" in installed:
    HAS_TORCH = True
else:
    HAS_TORCH = False

if HAS_TENSORFLOW and HAS_TORCH:
    import torch # torch must be imported before tensorflow'''#TBE

from deeptrack.features import *
from deeptrack.aberrations import *
from deeptrack.augmentations import *

from deeptrack.math import *
from deeptrack.noises import *
from deeptrack.optics import *
from deeptrack.scatterers import *
from deeptrack.sequences import *
from deeptrack.elementwise import *
from deeptrack.statistics import *
from deeptrack.holography import *

from deeptrack.image import strip

# if not HAS_TENSORFLOW:
    # Lazy imports to avoid overhead of importing tensorflow

generators = lazy_import.lazy_module("deeptrack.generators")
models = lazy_import.lazy_module("deeptrack.models")
datasets = lazy_import.lazy_module("deeptrack.datasets")
losses = lazy_import.lazy_module("deeptrack.losses")
layers = lazy_import.lazy_module("deeptrack.layers")
visualization = lazy_import.lazy_module("deeptrack.visualization")

# if not HAS_TORCH:
pytorch = lazy_import.lazy_module("deeptrack.pytorch")
deeplay = lazy_import.lazy_module("deeptrack.deeplay")

should_import = False
if should_import:
    from . import generators
    from . import models
    from . import datasets
    from . import losses
    from . import layers
    from . import visualization
    from . import pytorch
    from . import deeplay

from deeptrack import (
    image,
    utils,
    backend,
    # Fake imports for IDE autocomplete
    # Does not actually import anything
)
