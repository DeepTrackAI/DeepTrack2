# flake8: noqa
import lazy_import

from pint import UnitRegistry
from .backend.pint_definition import pint_definitions

import warnings
import importlib.util

# Checks if TensorFlow is installed and issues a compatibility warning.
# Check if TensorFlow is installed
tensorflow_installed = importlib.util.find_spec("tensorflow") is not None
if tensorflow_installed:
    warnings.warn(
        (
            "TensorFlow is detected in your environment. "
            "DeepTrack2 version 2.0++ no longer supports TensorFlow. "
            "If you need TensorFlow support, "
            "please install the legacy version 1.7 of DeepTrack2:\n\n"
            "    pip install deeptrack==1.7\n\n"
            "For more details, refer to the DeepTrack documentation."
        ),
        UserWarning
    )

# Create a unit registry with custom pixel-related units.
units = UnitRegistry(pint_definitions.split("\n"))

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

# if not HAS_TORCH:
pytorch = lazy_import.lazy_module("deeptrack.pytorch")
deeplay = lazy_import.lazy_module("deeptrack.deeplay")

should_import = False
if should_import:
    from . import pytorch
    from . import deeplay

from deeptrack import (
    image,
    utils,
    backend,
    # Fake imports for IDE autocomplete
    # Does not actually import anything
)
