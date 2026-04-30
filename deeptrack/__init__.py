# flake8: noqa
from typing import TYPE_CHECKING

from pint import UnitRegistry
from deeptrack.backend.pint_definition import pint_definitions

import warnings
import importlib.util

# Check if TensorFlow is installed and issues a compatibility warning.
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
        UserWarning,
    )

# Create a unit registry with custom pixel-related units.
units_registry = UnitRegistry(pint_definitions.split("\n"))
units = units_registry  # Alias for backward compatibility


from deeptrack.backend import *

from deeptrack.properties import *
from deeptrack.features import *
from deeptrack.sequences import *
from deeptrack.wrappers import *
from deeptrack.elementwise import *

from deeptrack.aberrations import *
from deeptrack.augmentations import *
from deeptrack.math import *
from deeptrack.noises import *
from deeptrack.optics import *
from deeptrack.scatterers import *
from deeptrack.statistics import *
from deeptrack.holography import *

if TORCH_AVAILABLE:
    import deeptrack.pytorch

if DEEPLAY_AVAILABLE:
    import deeptrack.deeplay


if TYPE_CHECKING:
    from deeptrack import pytorch
    from deeptrack import deeplay


from deeptrack import (
    utils,
    backend,
    # Fake imports for IDE autocomplete
    # Does not actually import anything
)

from deeptrack import tests  # TODO: Eliminate once tests is moved out.
