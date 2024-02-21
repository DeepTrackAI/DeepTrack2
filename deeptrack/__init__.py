# flake8: noqa
from pint import UnitRegistry, Context
from .backend.pint_definition import pint_definitions
import lazy_import
import importlib


units = UnitRegistry(pint_definitions.split("\n"))

# Check if tensorflow is installed without importing it
import pkg_resources

installed = [pkg.key for pkg in pkg_resources.working_set]

if "tensorflow" in installed:
    HAS_TENSORFLOW = True
else:
    HAS_TENSORFLOW = False

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

# if HAS_TENSORFLOW:
    # Lazy imports to avoid overhead of importing tensorflow
generators = lazy_import.lazy_module("deeptrack.generators")
models = lazy_import.lazy_module("deeptrack.models")
datasets = lazy_import.lazy_module("deeptrack.datasets")
losses = lazy_import.lazy_module("deeptrack.losses")
layers = lazy_import.lazy_module("deeptrack.layers")
pytorch = lazy_import.lazy_module("deeptrack.pytorch")
deeplay = lazy_import.lazy_module("deeptrack.deeplay")
visualization = lazy_import.lazy_module("deeptrack.visualization")

from deeptrack import (
    image,
    utils,
    backend,
    test,
    # Fake imports for IDE autocomplete
    # Does not actually import anything
    generators,
    models,
    datasets,
    losses,
    layers,
    pytorch,
    sources,
    visualization,
    deeplay
)
