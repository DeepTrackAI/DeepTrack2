"""Core features for building and processing pipelines in DeepTrack2.

This module defines the core classes and utilities used to create and 
manipulate features in DeepTrack2, enabling users to build sophisticated data 
processing pipelines with modular, reusable, and composable components.

Key Features
-------------
- **Features**

    A `Feature` is a building block of a data processing pipeline. 
    It represents a transformation applied to data, such as image manipulation,
    data augmentation, or computational operations. Features are highly 
    customizable and can be combined into pipelines for complex workflows.

- **Structural Features**

    Structural features extend the basic `Feature` class by adding hierarchical
    or logical structures, such as chains, branches, or probabilistic choices. 
    They enable the construction of pipelines with advanced data flow 
    requirements.

- **Feature Properties**

    Features in DeepTrack2 can have dynamically sampled properties, enabling 
    parameterization of transformations. These properties are defined at 
    initialization and can be updated during pipeline execution.

- **Pipeline Composition**

    Features can be composed into flexible pipelines using intuitive operators 
    (`>>`, `&`, etc.), making it easy to define complex data processing 
    workflows.

- **Lazy Evaluation**

    DeepTrack2 supports lazy evaluation of features, ensuring that data is 
    processed only when needed, which improves performance and scalability.

Module Structure
----------------
Key Classes: 

- `Feature`: Base class for all features in DeepTrack2.

    It represents a modular data transformation with properties and methods for
    customization.

- `StructuralFeature`: Provide structure without input transformations.

    A specialized feature for organizing and managing hierarchical or logical 
    structures in the pipeline.

- `ArithmeticOperationFeature`: Apply arithmetic operation element-wise.

    A parent class for features performing arithmetic operations like addition,
    subtraction, multiplication, and division.

Structural Feature Classes:
- `Chain`: Sequentially apply multiple features to the input data (>>).
- `Branch`: Alias of `Chain`.
- `Probability`: Resolve a feature with a certain probability.
- `Repeat`: Apply a feature multiple times in sequence (^).
- `Combine`: Combine multiple features into a single feature.
- `Bind`: Bind a feature with property arguments.
- `BindResolve`: Alias of `Bind`.
- `BindUpdate`: DEPRECATED Bind a feature with certain arguments.
- `ConditionalSetProperty`: DEPRECATED Conditionally override child properties.
- `ConditionalSetFeature`: DEPRECATED Conditionally resolve features.

Other Feature Classes:
- `DummyFeature`: A no-op feature that simply returns the input unchanged.
- `Value`: Store a constant value as a feature.
- `Stack`: Stack the input and the value.
- `Arguments`: A convenience container for pipeline arguments.
- `Slice`: Dynamically applies array indexing to inputs.
- `Lambda`: Apply a user-defined function to the input.
- `Merge`: Apply a custom function to a list of inputs.
- `OneOf`: Resolve one feature from a given collection.
- `OneOfDict`: Resolve one feature from a dictionary and apply it to an input.
- `LoadImage`: Load an image from disk and preprocess it.
- `SampleToMasks`: Create a mask from a list of images.
- `AsType`: Convert the data type of images.
- `ChannelFirst2d`: DEPRECATED Convert an image to a channel-first format.
- `Upscale`: Simulate a pipeline at a higher resolution.
- `NonOverlapping`: Ensure volumes are placed non-overlapping in a 3D space.
- `Store`: Store the output of a feature for reuse.
- `Squeeze`: Squeeze the input image to the smallest possible dimension.
- `Unsqueeze`: Unsqueeze the input image to the smallest possible dimension.
- `ExpandDims`: Alias of `Unsqueeze`.
- `MoveAxis`: Moves the axis of the input image.
- `Transpose`: Transpose the input image.
- `Permute`: Alias of `Transpose`.
- `OneHot`: Convert the input to a one-hot encoded array.
- `TakeProperties`: Extract all instances of properties from a pipeline.

Arithmetic Feature Classes:
- `Add`: Add a value to the input.
- `Subtract`: Subtract a value from the input.
- `Multiply`: Multiply the input by a value.
- `Divide`: Divide the input with a value.
- `FloorDivide`: Divide the input with a value.
- `Power`: Raise the input to a power.
- `LessThan`: Determine if input is less than value.
- `LessThanOrEquals`: Determine if input is less than or equal to value.
- `LessThanOrEqual`: Alias for `LessThanOrEquals`.
- `GreaterThan`: Determine if input is greater than value.
- `GreaterThanOrEquals`: Determine if input is greater than or equal to value.
- `GreaterThanOrEqual`: Alias for `GreaterThanOrEquals`.
- `Equals`: Determine if input is equal to value.
- `Equal`: Alias for `Equals`.

Functions:

- `propagate_data_to_dependencies`:

    def propagate_data_to_dependencies(
        feature: Feature,
        **kwargs: Any
    ) -> None

    Propagates data to all dependencies of a feature, updating their properties
    with the provided values.

Examples
--------
Define a simple pipeline with features:
>>> import deeptrack as dt
>>> import numpy as np

Create a basic addition feature:
>>> class BasicAdd(dt.Feature):
...     def get(self, image, value, **kwargs):
...         return image + value

Create two features:
>>> add_five = BasicAdd(value=5)
>>> add_ten = BasicAdd(value=10)

Chain features together:
>>> pipeline = dt.Chain(add_five, add_ten)

Or equivalently:
>>> pipeline = add_five >> add_ten

Process an input image:
>>> input_image = np.array([[1, 2, 3], [4, 5, 6]])
>>> output_image = pipeline(input_image)
>>> print(output_image)
[[16 17 18]
 [19 20 21]]

"""

from __future__ import annotations

import itertools
import operator
import random
from typing import Any, Callable, Iterable, Literal, TYPE_CHECKING

import array_api_compat as apc
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import animation
from pint import Quantity
from scipy.spatial.distance import cdist

from deeptrack import units
from deeptrack.backend import config, TORCH_AVAILABLE, xp
from deeptrack.backend.core import DeepTrackNode
from deeptrack.backend.units import ConversionTable, create_context
from deeptrack.image import Image
from deeptrack.properties import PropertyDict, SequentialProperty
from deeptrack.sources import SourceItem
from deeptrack.types import ArrayLike, PropertyLike

if TORCH_AVAILABLE:
    import torch

__all__ = [
    "Feature",
    "StructuralFeature",
    "Chain",
    "Branch",
    "DummyFeature",
    "Value",
    "ArithmeticOperationFeature",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "FloorDivide",
    "Power",
    "LessThan",
    "LessThanOrEquals",
    "LessThanOrEqual",
    "GreaterThan",
    "GreaterThanOrEquals",
    "GreaterThanOrEqual",
    "Equals",
    "Equal",
    "Stack",
    "Arguments",
    "Probability",
    "Repeat",
    "Combine",
    "Slice",
    "Bind",
    "BindResolve",
    "BindUpdate",
    "ConditionalSetProperty",
    "ConditionalSetFeature",
    "Lambda",
    "Merge",
    "OneOf",
    "OneOfDict",
    "LoadImage",  #TODO ***MG***
    "SampleToMasks",  #TODO ***MG***
    "AsType",  #TODO ***MG***
    "Upscale",  #TODO ***AL***
    "ChannelFirst2d",  #TODO ***AL***
    "NonOverlapping",  #TODO ***AL***
    "Store",  #TODO ***JH***
    "Squeeze",
    "Unsqueeze",
    "ExpandDims",
    "MoveAxis",
    "Transpose",
    "Permute",
    "OneHot",
    "TakeProperties",  #TODO ***JH***
]


if TYPE_CHECKING:
    import torch


MERGE_STRATEGY_OVERRIDE: int = 0
MERGE_STRATEGY_APPEND: int = 1


class Feature(DeepTrackNode):
    """Base feature class.

    Features define the image generation process.
    
    All features operate on lists of images. Most features, such as noise,
    apply a tranformation to all images in the list. This transformation can be
    additive, such as adding some Gaussian noise or a background illumination,
    or non-additive, such as introducing Poisson noise or performing a low-pass
    filter. This transformation is defined by the `get(image, **kwargs)`
    method, which all implementations of the class `Feature` need to define.
    This method operates on a single image at a time.

    Whenever a Feature is initialized, it wraps all keyword arguments passed to
    the constructor as `Property` objects, and stored in the `properties` 
    attribute as a `PropertyDict`.
    
    When a Feature is resolved, the current value of each property is sent as
    input to the get method.

    **Computational Backends and Data Types**
    
    This class also provides mechanisms for managing numerical types and 
    computational backends.

    Supported backends include NumPy and PyTorch. The active backend is 
    determined at initialization and stored in the `_backend` attribute, which 
    is used internally to control how computations are executed. The backend
    can be switched using the `.numpy()` and `.torch()` methods.

    Numerical types used in computation (float, int, complex, and bool) can be 
    configured using the `.dtype()` method. The chosen types are retrieved 
    via the properties `float_dtype`, `int_dtype`, `complex_dtype`, and 
    `bool_dtype`. These are resolved dynamically using the backend's internal 
    type resolution system and are used in downstream computations.

    The computational device (e.g., "cpu" or a specific GPU) is managed through 
    the `.to()` method and accessed via the `device` property. This is 
    especially relevant for PyTorch backends, which support GPU acceleration.

    Parameters
    ----------
    _input: Any, optional.
        The input data for the feature. If left empty, no initial input is set.
        It is most commonly a NumPy array, PyTorch tensor, or Image object, or
        a list of NumPy arrays, PyTorch tensors, or Image objects; however, it
        can be anything.
    **kwargs: Any
        Keyword arguments to configure the feature. Each keyword argument is 
        wrapped as a `Property` and added to the `properties` attribute, 
        allowing dynamic sampling and parameterization during the feature's 
        execution. These properties are passed to the `get()` method when a
        feature is resolved.

    Attributes
    ----------
    properties: PropertyDict
        A dictionary containing all keyword arguments passed to the 
        constructor, wrapped as instances of `Property`. The properties can 
        dynamically sample values during pipeline execution. A sampled copy of
        this dictionary is passed to the `get` function and appended to the 
        properties of the output image.
    _input: DeepTrackNode
        A node representing the input data for the feature. It is most commonly
        a NumPy array, PyTorch tensor, or Image object, or a list of NumPy
        arrays, PyTorch tensors, or Image objects; however, it can be anything.
        It supports lazy evaluation and graph traversal.
    _random_seed: DeepTrackNode
        A node representing the feature’s random seed. This allows for 
        deterministic behavior when generating random elements, and ensures 
        reproducibility during evaluation.
    arguments: Feature | None
        An optional `Feature` whose properties are bound to this feature. This 
        allows dynamic property sharing and centralized parameter management 
        in complex pipelines.
    __list_merge_strategy__: int
        Specifies how the output of `.get(image, **kwargs)` is merged with the 
        current `_input`. Options include:
        - `MERGE_STRATEGY_OVERRIDE` (0, default): `_input` is replaced by the
        new output.
        - `MERGE_STRATEGY_APPEND` (1): The output is appended to the end of 
        `_input`.
    __distributed__: bool
        Determines whether `.get(image, **kwargs)` is applied to each element 
        of the input list independently (`__distributed__ = True`) or to the 
        list as a whole (`__distributed__ = False`).
    __conversion_table__: ConversionTable
        Defines the unit conversions used by the feature to convert its 
        properties into the desired units.
    _wrap_array_with_image: bool
        Internal flag that determines whether arrays are wrapped as `Image` 
        instances during evaluation. When `True`, image metadata and properties 
        are preserved and propagated. It defaults to `False`.
    float_dtype: np.dtype
        The data type of the float numbers.
    int_dtype: np.dtype
        The data type of the integer numbers.
    complex_dtype: np.dtype
        The data type of the complex numbers.
    bool_dtype: np.dtype
        The data type of the boolean numbers.
    device: str or torch.device
        The device on which the feature is executed.
    _backend: Literal["numpy", "torch"]
        The computational backend.

    Methods
    -------
    `get(image: Any, **kwargs: Any) -> Any`
        Abstract method that defines how the feature transforms the input. The
        input is most commonly a NumPy array, PyTorch tensor, or Image object,
        but it can be anything.
    `__call__(image_list: Any, _ID: tuple[int, ...], **kwargs: Any) -> Any`
        It executes the feature or pipeline on the input and applies property 
        overrides from `kwargs`.
    `resolve(image_list: Any, _ID: tuple[int, ...], **kwargs: Any) -> Any`
        Alias of `__call__()`.
    `to_sequential(**kwargs: Any) -> Feature`
        It convert a feature to be resolved as a sequence.
    `store_properties(toggle: bool, recursive: bool) -> Feature`
        It controls whether the properties are stored in the output `Image`
        object.
    `torch(device: torch.device or None, recursive: bool) -> 'Feature'`
        It sets the backend to torch.
    `numpy(recursice: bool) -> Feature`
        It set the backend to numpy.
    `get_backend() -> Literal["numpy", "torch"]`
        It returns the current backend of the feature.
    `dtype(float: Literal["float32", "float64", "default"] or None, int: Literal["int16", "int32", "int64", "default"] or None, complex: Literal["complex64", "complex128", "default"] or None, bool: Literal["bool", "default"] or None) -> Feature`
        It set the dtype to be used during evaluation.
    `to(device: str or torch.device) -> Feature`
        It set the device to be used during evaluation.
    `batch(batch_size: int) -> tuple`
        It batches the feature for repeated execution.
    `action(_ID: tuple[int, ...]) -> Any | list[Any]`
        It implements the core logic to create or transform the input(s).
    `update(**global_arguments: Any) -> Feature`
        It refreshes the feature to create a new image.
    `add_feature(feature: Feature) -> Feature`
        It adds a feature to the dependency graph of this one.
    `seed(updated_seed: int, _ID: tuple[int, ...]) -> int`
        It sets the random seed for the feature, ensuring deterministic 
        behavior.
    `bind_arguments(arguments: Feature) -> Feature`
        It binds another feature’s properties as arguments to this feature.
    `plot(input_image: np.ndarray | list[np.ndarray] | Image | list[Image] | None = None, resolve_kwargs: dict | None = None, interval: float | None = None, **kwargs: Any) -> Any`
        It visualizes the output of the feature.

    **Private and internal methods.**
    `_normalize(**properties: Any) -> dict[str, Any]`
        It normalizes the properties of the feature.
    `_process_properties(propertydict: dict[str, Any]) -> dict[str, Any]`
        It preprocesses the input properties before calling the `get` method.
    `_activate_sources(x: Any) -> None`
        It activates sources in the input data.
    `__getattr__(key: str) -> Any`
        It provides custom attribute access for the Feature class.
    `__iter__() -> Feature`
        It returns an iterator for the feature.
    `__next__() -> Any`
        It return the next element iterating over the feature.
    `__rshift__(other: Any) -> Feature`
        It allows chaining of features.
    `__rrshift__(other: Any) -> Feature`
        It allows right chaining of features.
    `__add__(other: Any) -> Feature`
        It overrides add operator.
    `__radd__(other: Any) -> Feature`
        It overrides right add operator.
    `__sub__(other: Any) -> Feature`
        It overrides subtraction operator.
    `__rsub__(other: Any) -> Feature`
        It overrides right subtraction operator.
    `__mul__(other: Any) -> Feature`
        It overrides multiplication operator.
    `__rmul__(other: Any) -> Feature`
        It overrides right multiplication operator.
    `__truediv__(other: Any) -> Feature`
        It overrides division operator.
    `__rtruediv__(other: Any) -> Feature`
        It overrides right division operator.
    `__floordiv__(other: Any) -> Feature`
        It overrides floor division operator.
    `__rfloordiv__(other: Any) -> Feature`
        It overrides right floor division operator.
    `__pow__(other: Any) -> Feature`
        It overrides power operator.
    `__rpow__(other: Any) -> Feature`
        It overrides right power operator.
    `__gt__(other: Any) -> Feature`
        It overrides greater than operator.
    `__rgt__(other: Any) -> Feature`
        It overrides right greater than operator.
    `__lt__(other: Any) -> Feature`
        It overrides less than operator.
    `__rlt__(other: Any) -> Feature`
        It overrides right less than operator.
    `__le__(other: Any) -> Feature`
        It overrides less than or equal to operator.
    `__rle__(other: Any) -> Feature`
        It overrides right less than or equal to operator.
    `__ge__(other: Any) -> Feature`
        It overrides greater than or equal to operator.
    `__rge__(other: Any) -> Feature`
        It overrides right greater than or equal to operator.
    `__xor__(other: Any) -> Feature`
        It overrides XOR operator.
    `__and__(other: Feature) -> Feature`
        It overrides AND operator.
    `__rand__(other: Feature) -> Feature`
        It overrides right AND operator.
    `__getitem__(key: Any) -> Feature`
        It allows direct slicing of the data.
    `_format_input(image_list: Any, **kwargs: Any) -> list[Any or Image]`
        It formats the input data for the feature.
    `_process_and_get(image_list: Any, **kwargs: Any) -> list[Any or Image]`
        It calls the `get` method according to the `__distributed__` attribute.
    `_process_output(image_list: Any, **kwargs: Any) -> None`
        It processes the output of the feature.
    `_image_wrapped_format_input(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        It ensures the input is a list of Image.
    `_no_wrap_format_input(image_list: Any, **kwargs: Any) -> list[Any]`
        It ensures the input is a list of Image.
    `_image_wrapped_process_and_get(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        It calls the `get()` method according to the `__distributed__`
        attribute.
    `_no_wrap_process_and_get(image_list: Any | list[Any], **kwargs: Any) -> list[Any]`
        It calls the `get()` method according to the `__distributed__`
        attribute.
    `_image_wrapped_process_output(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> None`
        It processes the output of the feature.
    `_no_wrap_process_output(image_list: Any | list[Any], **kwargs: Any) -> None`
        It processes the output of the feature.

    Examples
    --------
    >>> import deeptrack as dt

    **Define and evaluate a simple feature**

    >>> import numpy as np
    >>>
    >>> feature = dt.Value(value=np.array([1, 2, 3]))
    >>> result = feature()
    >>> result
    array([1, 2, 3])

    **Chain features using '>>'**

    >>> pipeline = dt.Value(value=np.array([1, 2, 3])) >> dt.Add(value=2)
    >>> pipeline()
    array([3, 4, 5])

    **Use arithmetic operators for syntactic sugar**

    >>> feature = dt.Value(value=np.array([1, 2, 3]))
    >>> result = (feature + 1) * 2 - 1
    >>> result()
    array([3, 5, 7])

    This is equivalent to chaining with `Add`, `Multiply`, and `Subtract`.

    **Evaluate a dynamic feature using `.update()`**

    >>> feature = dt.Value(value=lambda: np.random.rand())
    >>> output1 = feature()
    >>> output1
    0.9938966963707441

    >>> output2 = feature()  # Cached result
    >>> output2
    0.9938966963707441

    >>> feature.update()
    >>> output3 = feature()  # New sample
    >>> output3
    0.3874078815170007

    **Generate a batch of outputs**

    >>> feature = dt.Value(lambda: np.random.rand()) + 1
    >>> batch = feature.batch(batch_size=3)
    >>> batch
    (array([1.6888222 , 1.88422131, 1.90027316]),)

    **Store and retrieve properties from outputs**

    >>> feature = dt.Value(value=3).store_properties(True)
    >>> output = feature(np.array([1, 2]))
    >>> output.get_property("value")
    3

    **Switch computational backend to torch**

    >>> import torch
    >>>
    >>> feature = dt.Add(value=5).torch()
    >>> input_tensor = torch.tensor([1.0, 2.0])
    >>> feature(input_tensor)
    tensor([6., 7.])

    **Use `.seed()` for reproducibility**

    >>> feature = dt.Value(lambda: np.random.randint(0, 100))
    >>> seed = feature.seed()
    >>> v1 = feature.update()()
    >>> v1
    76

    >>> feature.seed(seed)
    >>> v2 = feature.update()()
    >>> v2
    76

    **Sequential feature with evolving property**

    >>> def rotate(sequence_length, previous_value):
    ...     return previous_value + 2 * np.pi / sequence_length

    >>> rotating = dt.Ellipse(
    ...     position=(16, 16),
    ...     radius=(1.5, 1),
    ...     rotation=0,
    ... ).to_sequential(rotation=rotate)

    >>> frames = dt.Sequence(rotating, sequence_length=5).update()
    >>> images = frames()
    >>> len(images)
    5

    **Bind dynamic arguments across multiple features**

    >>> arguments = dt.Arguments(frequency=1, amplitude=2)
    >>> wave = (
    ...     dt.Value(
    ...         value=lambda frequency: np.linspace(0, 2 * np.pi * frequency, 100),
    ...         frequency=arguments.frequency,
    ...     )
    ...     >> np.sin
    ...     >> dt.Multiply(
    ...         value=lambda amplitude: amplitude,
    ...         amplitude=arguments.amplitude,
    ...     )
    ... )
    >>> wave.bind_arguments(arguments)

    >>> from matplotlib import pyplot as plt
    >>>
    >>> plt.plot(wave())
    >>> plt.show()

    >>> plt.plot(wave(frequency=2, amplitude=1))  # Raw image with no noise
    >>> plt.show()

    """

    properties: PropertyDict
    _input: DeepTrackNode
    _random_seed: DeepTrackNode
    arguments: Feature | None

    __list_merge_strategy__ = MERGE_STRATEGY_OVERRIDE
    __distributed__ = True
    __conversion_table__ = ConversionTable()

    _wrap_array_with_image: bool = False

    _float_dtype: str
    _int_dtype: str
    _complex_dtype: str
    _device: str | torch.device
    _backend: Literal["numpy", "torch"]

    @property
    def float_dtype(self) -> np.dtype | torch.dtype:
        """The dtype of the float numbers."""
        return xp.get_float_dtype(self._float_dtype)

    @property
    def int_dtype(self) -> np.dtype | torch.dtype:
        """The dtype of the integer numbers."""
        return xp.get_int_dtype(self._int_dtype)

    @property
    def complex_dtype(self) -> np.dtype | torch.dtype:
        """The dtype of the complex numbers."""
        return xp.get_complex_dtype(self._complex_dtype)

    @property
    def bool_dtype(self) -> np.dtype | torch.dtype:
        """The dtype of the boolean numbers."""
        return xp.get_bool_dtype(self._bool_dtype)

    @property
    def device(self) -> str | torch.device:
        """The device to be used during evaluation."""
        return self._device

    def __init__(
        self: Feature,
        _input: Any = [],
        **kwargs: Any,
    ):
        """Initialize a new Feature instance.

        Parameters
        ----------
        _input: Any, optional
            The initial input(s) for the feature. It is most commonly a NumPy
            array, PyTorch tensor, or Image object, or a list of NumPy arrays,
            PyTorch tensors, or Image objects; however, it can be anything. If
            not provided, defaults to an empty list.
        **kwargs: Any
            Keyword arguments that are wrapped into `Property` instances and 
            stored in `self.properties`, allowing for dynamic or parameterized
            behavior.

        """

        # Store backend on initialization.
        self._backend = config.get_backend()

        # Store the dtype and device on initialization.
        self._float_dtype = "default"
        self._int_dtype = "default"
        self._complex_dtype = "default"
        self._bool_dtype = "default"
        self._device = config.get_device()

        super().__init__()

        # Ensure the feature has a 'name' property; default = class name.
        kwargs.setdefault("name", type(self).__name__)

        # 1) Create a PropertyDict to hold the feature’s properties.
        self.properties = PropertyDict(**kwargs)
        self.properties.add_child(self)
        # self.add_dependency(self.properties)  # Executed by add_child.

        # 2) Initialize the input as a DeepTrackNode.
        self._input = DeepTrackNode(_input)
        self._input.add_child(self)
        # self.add_dependency(self._input)  # Executed by add_child.

        # 3) Random seed node (for deterministic behavior if desired).
        self._random_seed = DeepTrackNode(
            lambda: random.randint(0, 2147483648)
        )
        self._random_seed.add_child(self)
        # self.add_dependency(self._random_seed)  # Executed by add_child.

        # Initialize arguments to None.
        self.arguments = None

    def get(
        self: Feature,
        image: Any,
        **kwargs: Any,
    ) -> Any:
        """Transform an input (abstract method).

        Abstract method that defines how the feature transforms the input. The 
        current value of all properties will be passed as keyword arguments.

        Parameters
        ----------
        image: Any
            The input to transform. It is most commonly a NumPy array, PyTorch
            tensor, or Image object, but it can be anything.
        **kwargs: Any
            The current value of all properties in `properties`, as well as any 
            global arguments passed to the feature.

        Returns
        -------
        Any
            The transformed image or list of images.

        Raises
        ------
        NotImplementedError
            Raised if this method is not overridden by subclasses.

        """

        raise NotImplementedError

    def __call__(
        self: Feature,
        image_list: Any = None,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Execute the feature or pipeline.

        This method executes the feature or pipeline on the provided input and 
        updates the computation graph if necessary. It handles overriding 
        properties using additional keyword arguments.

        The actual computation is performed by calling the parent `__call__` 
        method in the `DeepTrackNode` class, which manages lazy evaluation and 
        caching.

        Parameters
        ----------
        image_list: Any, optional
            The input to the feature or pipeline. It is most commonly a NumPy
            array, PyTorch tensor, or Image object, or a list of NumPy arrays,
            PyTorch tensors, or Image objects; however, it can be anything. It
            defaults to `None`, in which case the feature uses the previous set
            input values or propagates properties.
        **kwargs: Any
            Additional parameters passed to the pipeline. These override 
            properties with matching names. For example, calling 
            `feature(x, value=4)` executes `feature` on the input `x` while 
            setting the property `value` to `4`. All features in a pipeline are 
            affected by these overrides.

        Returns
        -------
        Any
            The output of the feature or pipeline after execution. This is
            typically a NumPy array, PyTorch tensor, or Image object, or a list
            of NumPy arrays, PyTorch tensors, or Image objects.

        Examples
        --------
        >>> import deeptrack as dt

        Deafine a feature:
        >>> feature = dt.Add(value=2)

        Call this feature with an input:
        >>> import numpy as np
        >>>
        >>> feature(np.array([1, 2, 3]))
        array([3, 4, 5])

        Execute the feature with previously set input:
        >>> feature()  # Uses stored input
        array([3, 4, 5])

        Override a property:
        >>> feature(np.array([1, 2, 3]), value=10)
        array([11, 12, 13])

        """

        with config.with_backend(self._backend):
            # If image_list is as Source, activate it.
            self._activate_sources(image_list)

            # Potentially fragile.
            # Maybe a special variable dt._last_input instead?
            # If the input is not empty, set the value of the input.
            if (
                image_list is not None
                and not (isinstance(image_list, list) and len(image_list) == 0)
                and not (isinstance(image_list, tuple)
                        and any(isinstance(x, SourceItem) for x in image_list))
            ):
                self._input.set_value(image_list, _ID=_ID)

            # A dict to store values of self.arguments before updating them.
            original_values = {}

            # If there are no self.arguments, instead propagate the values of
            # the kwargs to all properties in the computation graph.
            if kwargs and self.arguments is None:
                propagate_data_to_dependencies(self, **kwargs)

            # If there are self.arguments, update the values of self.arguments
            # to match kwargs.
            if isinstance(self.arguments, Feature):
                for key, value in kwargs.items():
                    if key in self.arguments.properties:
                        original_values[key] = \
                            self.arguments.properties[key](_ID=_ID)
                        self.arguments.properties[key]\
                            .set_value(value, _ID=_ID)

            # This executes the feature. DeepTrackNode will determine if it
            # needs to be recalculated. If it does, it will call the `action`
            # method.
            output = super().__call__(_ID=_ID)

            # If there are self.arguments, reset the values of self.arguments
            # to their original values.
            for key, value in original_values.items():
                self.arguments.properties[key].set_value(value, _ID=_ID)

        return output

    resolve = __call__

    def to_sequential(
        self: Feature,
        **kwargs: Any,
    ) -> Feature:
        """Convert a feature to be resolved as a sequence.

        Should be called on individual features, not combinations of features.
        All keyword arguments will be treated as sequential properties and will
        be passed to the parent feature.

        If a property from the keyword argument already exists in the feature,
        the existing property will be used to initialize the passed property
        (that is, it will be used for the first timestep).

        Parameters
        ----------
        self: Feature
            Feature to make sequential.
        kwargs: Any
            Keyword arguments to pass on as sequential properties of `feature`.

        Returns
        -------
        Feature
            The input feature evolved as a sequence

        Examples
        --------
        >>> import deeptrack as dt

        Sequentially evaluate a rotating ellipse.

        Create the optics:
        >>> optics = dt.Fluorescence(
        ...     NA=0.6,
        ...     magnification=10,
        ...     resolution=1e-6,
        ...     wavelength=633e-9,
        ...     output_region=(0, 0, 32, 32),
        ... )

        Create the scatterer:
        >>> ellipse = Ellipse(
        ...     position_unit="pixel",
        ...     position=(16, 16),
        ...     intensity=1,
        ...     radius=(1.5e-6, 1e-6),
        ...     rotation=0,  # Initial rotation at time step 0
        ... )

        Implement a function to increment the rotation:
        >>> from numpy import pi
        >>>
        >>> def get_rotation(sequence_length, previous_value):
        ...     delta = 2 * pi / sequence_length
        ...     return previous_value + delta

        Call `to_sequential()` to resolve the feature sequentially:
        >>> rotating_ellipse = ellipse.to_sequential(rotation=get_rotation)

        Image the scatterer with the optics:
        >>> imaged_rotating_ellipse = optics(rotating_ellipse)

        Encapsulate as a `Sequence` object and specify the sequence length:
        >>> imaged_rotating_ellipse_sequence = Sequence(
        ...     imaged_rotating_ellipse,
        ...     sequence_length=10
        ... )

        Finally observe the scatterer rotate:
        >>> imaged_rotating_ellipse_sequence.update().plot();

        """

        for property_name in kwargs.keys():
            if property_name in self.properties:
                # Insert sequential property with initialized value taken from
                # the already available property.
                self.properties[property_name] = SequentialProperty(
                    self.properties[property_name], **self.properties
                )
            else:
                # Insert empty sequential property.
                self.properties[property_name] = SequentialProperty()

            self.properties.add_dependency(self.properties[property_name])
            # self.properties[property_name].add_child(self.properties)

        for property_name, sampling_rule in kwargs.items():
            prop = self.properties[property_name]

            all_kwargs = dict(
                previous_value=prop.previous_value,
                previous_values=prop.previous_values,
                sequence_length=prop.sequence_length,
                sequence_index=prop.sequence_index,
            )

            for key, value in self.properties.items():
                if key == property_name:
                    continue

                if isinstance(value, SequentialProperty):
                    all_kwargs[key] = value
                    all_kwargs["previous_" + key] = value.previous_values
                else:
                    all_kwargs[key] = value

            if not prop.initial_sampling_rule:
                prop.initial_sampling_rule = prop.create_action(
                    sampling_rule,
                    **{k:all_kwargs[k] for k in all_kwargs
                       if k != "previous_value"},
                )

            prop.sample = prop.create_action(sampling_rule, **all_kwargs)

        return self

    def store_properties(
        self: Feature,
        toggle: bool = True,
        recursive: bool = True,
    ) -> Feature:
        """Control whether to return an Image object.
        
        If selected `True`, the output of the evaluation of the feature is an 
        Image object that also contains the properties.

        Parameters
        ----------
        toggle: bool
            If `True` (default), store properties. If `False`, do not store.
        recursive: bool
            If `True` (default), also set the same behavior for all dependent
            features. If `False`, it does not.

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt

        Create a feature and enable property storage:
        >>> feature = dt.Add(value=2)
        >>> feature.store_properties(True)

        Evaluate the feature and inspect the stored properties:
        >>> import numpy as np
        >>>
        >>> output = feature(np.array([1, 2, 3]))
        >>> isinstance(output, dt.Image)
        True
        >>> output.get_property("value")
        2

        Disable property storage:
        >>> feature.store_properties(False)
        >>> output = feature(np.array([1, 2, 3]))
        >>> isinstance(output, dt.Image)
        False

        Apply recursively to a pipeline:
        >>> feature1 = dt.Add(value=1)
        >>> feature2 = dt.Multiply(value=2)
        >>> pipeline = feature1 >> feature2
        >>> pipeline.store_properties(True, recursive=True)
        >>> output = pipeline(np.array([1, 2]))
        >>> output.get_property("value")
        1
        >>> output.get_property("value", get_one=False)
        [1, 2]

        """

        self._wrap_array_with_image = toggle

        if recursive:
            for dependency in self.recurse_dependencies():
                if isinstance(dependency, Feature):
                    dependency.store_properties(toggle, recursive=False)

        return self

    def torch(
        self: Feature,
        device: torch.device | None = None,
        recursive: bool = True,
    ) -> Feature:
        """Set the backend to torch.

        Parameters
        ----------
        device: torch.device, optional
            The target device of the output (e.g., cpu or cuda). It defaults to
            `None`.
        recursive: bool, optional
            If `True` (default), it also convert all dependent features. If
            `False`, it does not.

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt
        >>> import torch

        Create a feature and switch to the PyTorch backend:
        >>> feature = dt.Multiply(value=2)
        >>> feature.torch()

        Call the feature on a torch tensor:
        >>> input_tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> output = feature(input_tensor)
        >>> output
        tensor([2., 4., 6.])

        Switch to GPU if available (CUDA):
        >>> if torch.cuda.is_available():
        ...     device = torch.device("cuda")
        ...     feature.torch(device=device)
        ...     output = feature(torch.tensor([1.0, 2.0, 3.0], device=device))
        ...     output.device.type
        'cuda'

        Switch to GPU if available (MPS):
        >>> if (torch.backends.mps.is_available()
        ...     and torch.backends.mps.is_built()):
        ...     device = torch.device("mps")
        ...     feature.torch(device=device)
        ...     output = feature(torch.tensor([1.0, 2.0, 3.0], device=device))
        ...     output.device.type
        'mps'

        Apply recursively in a pipeline:
        >>> f1 = dt.Add(value=1)
        >>> f2 = dt.Multiply(value=2)
        >>> pipeline = f1 >> f2
        >>> pipeline.torch()
        >>> output = pipeline(torch.tensor([1.0, 2.0]))
        >>> output
        tensor([4., 6.])

        """

        self._backend = "torch"
        if recursive:
            for dependency in self.recurse_dependencies():
                if isinstance(dependency, Feature):
                    dependency.torch(device, recursive=False)

        self.invalidate()
        return self

    def numpy(
        self: Feature,
        recursive: bool = True,
    ) -> Feature:
        """Set the backend to numpy.

        Parameters
        ----------
        recursive: bool, optional
            If `True` (default), also convert all dependent features.

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt
        >>> import numpy as np

        Create a feature and ensure it uses the NumPy backend:
        >>> feature = dt.Add(value=5)
        >>> feature.numpy()

        Evaluate the feature on a NumPy array:
        >>> output = feature(np.array([1, 2, 3]))
        >>> output
        array([6, 7, 8])

        Apply recursively in a pipeline:
        >>> f1 = dt.Multiply(value=2)
        >>> f2 = dt.Subtract(value=1)
        >>> pipeline = f1 >> f2
        >>> pipeline.numpy()
        >>> output = pipeline(np.array([1, 2, 3]))
        >>> output
        array([1, 3, 5])

        """

        self._backend = "numpy"
        if recursive:
            for dependency in self.recurse_dependencies():
                if isinstance(dependency, Feature):
                    dependency.numpy(recursive=False)
        self.invalidate()
        return self

    def get_backend(
            self: Feature
    ) -> Literal["numpy", "torch"]:
        """Get the current backend of the feature.

        Returns
        -------
        Literal["numpy", "torch"]
            The backend of this feature

        Examples
        --------
        >>> import deeptrack as dt

        Create a feature:
        >>> feature = dt.Add(value=5)

        Set the feature's backend to NumPy and check it:
        >>> feature.numpy()
        >>> feature.get_backend()
        'numpy'

        Set the feature's backend to PyTorch and check it:
        >>> feature.torch()
        >>> feature.get_backend()
        'torch'

        """
        return self._backend

    def dtype(
        self: Feature,
        float: Literal["float32", "float64", "default"] | None = None,
        int: Literal["int16", "int32", "int64", "default"] | None = None,
        complex: Literal["complex64", "complex128", "default"] | None = None,
        bool: Literal["bool", "default"] | None = None,
    ) -> Feature:
        """Set the dtype to be used during evaluation.

        It alters the dtype used for array creation, but does not automatically
        cast the type.

        Parameters
        ----------
        float: str, optional
            The float dtype to set. It can be `"float32"`, `"float64"`,
            `"default"`, or `None`. It defaults to `None`.
        int: str, optional
            The int dtype to set. It can be `"int16"`, `"int32"`, `"int64"`,
            `"default"`, or `None`. It defaults to `None`.
        complex: str, optional
            The complex dtype to set. It can be `"complex64"`, `"complex128"`,
            `"default"`, or `None`. It defaults to `None`.
        bool: str, optional
            The bool dtype to set. It can be `"bool"`, `"default"`, or `None`.
            It defaults to `None`.

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt

        Set float and int data types for a feature:
        >>> feature = dt.Multiply(value=2)
        >>> feature.dtype(float="float32", int="int16")
        >>> feature.float_dtype
        dtype('float32')
        >>> feature.int_dtype
        dtype('int16')

        Use complex numbers in the feature:
        >>> feature.dtype(complex="complex128")
        >>> feature.complex_dtype
        dtype('complex128')

        Reset float dtype to default:
        >>> feature.dtype(float="default")
        >>> feature.float_dtype  # resolved from config
        dtype('float64')  # depending on backend config

        """

        if float is not None:
            self._float_dtype = float
        if int is not None:
            self._int_dtype = int
        if complex is not None:
            self._complex_dtype = complex
        if bool is not None:
            self._bool_dtype = bool

        return self

    def to(
        self: Feature,
        device: str | torch.device,
    ) -> Feature:
        """Set the device to be used during evaluation.

        Parameters
        ----------
        device: str or torch.device
            The device to use. If the backend is numpy, this can only be "cpu".

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt
        >>> import torch

        Create a feature and assign a device (for torch backend):
        >>> feature = dt.Add(value=1)
        >>> feature.torch()
        >>> feature.to(torch.device("cpu"))
        >>> feature.device
        device(type='cpu')

        Move the feature to GPU (if available):
        >>> if torch.cuda.is_available():
        ...     feature.to(torch.device("cuda"))
        ...     feature.device
        device(type='cuda')

        Use Apple MPS device on Apple Silicon (if supported):
        >>> if (torch.backends.mps.is_available()
        ...     and torch.backends.mps.is_built()):
        ...     feature.to(torch.device("mps"))
        ...     feature.device
        device(type='mps')

        """

        self._device = device

        return self

    def batch(
        self: Feature,
        batch_size: int = 32,
    ) -> tuple:
        """Batch the feature.

        This method produces a batch of outputs by repeatedly calling 
        `update()` and `__call__()`.

        Parameters
        ----------
        batch_size: int
            The number of times to sample or generate data. It defaults to 32.

        Returns
        -------
        tuple
            A tuple where each element corresponds to one component of the
            output. If the outputs are NumPy arrays or PyTorch tensors, each
            element is a stacked array.

        Examples
        --------
        >>> import deeptrack as dt

        Define a feature that adds a random value to a fixed array:
        >>> import numpy as np
        >>>
        >>> feature = (
        ...     dt.Value(value=np.array([[-1, 1]]))
        ...     >> dt.Add(value=lambda: np.random.rand())
        ... )

        Evaluate the feature once:
        >>> output = feature()
        >>> output
        array([[-0.77378939,  1.22621061]])

        Generate a batch of outputs:
        >>> batch = feature.batch(batch_size=3)
        >>> batch
        (array([[-0.2375814 ,  1.7624186 ],
                [-0.65764878,  1.34235122],
                [-0.87449525,  1.12550475]]),)

        """

        results = [self.update()() for _ in range(batch_size)]

        try:
            # Attempt to unzip results
            results = [(r,) for r in results]
        except TypeError:
            # If outputs are scalar (not iterable), wrap each in a tuple
            results = [(r,) for r in results]
            results = [(r,) for r in results]

        results = list(zip(*results))

        for idx, r in enumerate(results):
            results[idx] = xp.stack(r)

        return tuple(results)

    def action(
        self: Feature,
        _ID: tuple[int, ...] = (),
    ) -> Any | list[Any]:
        """Core logic to create or transform the input.

        This method is the central point where the feature's transformation is
        actually executed. It retrieves the input data, evaluates the current
        values of all properties, formats the input into a list of `Image`
        objects, and applies the `get()` method to perform the desired
        transformation.

        Depending on the configuration, the transformation can be applied to
        each element of the input independently or to the full list at once.

        The outputs are optionally post-processed, and then merged back into
        the input according to the configured merge strategy.
        Parameters

        The behavior of this method is influenced by several class attributes:

        - `__distributed__`: If `True` (default), the `get()` method is applied
          independently to each input in the input list. If `False`, the
          `get()` method is applied to the entire list at once.

        - `__list_merge_strategy__`: Determines how the outputs returned by
          `get()` are combined with the original inputs:
            * `MERGE_STRATEGY_OVERRIDE` (default): The output replaces the
              input.
            * `MERGE_STRATEGY_APPEND`: The output is appended to the input
              list.

        - `_wrap_array_with_image`: If `True`, input arrays are wrapped as
          `Image` instances and their properties are preserved. Otherwise,
          they are treated as raw arrays.

        - `_process_properties()`: This hook can be overridden to pre-process
          properties before they are passed to `get()` (e.g., for unit
          normalization).

        - `_process_output()`: Handles post-processing of the output images,
          including appending feature properties and binding argument features.

        ----------
        _ID: tuple[int], optional
            The unique identifier for the current execution. It defaults to ().

        Returns
        -------
        Any or list[Any]
            The resolved output or list of resolved outputs. If only a single
            output is generated, the result is unwrapped for convenience.

        Examples
        --------
        >>> import deeptrack as dt

        Define a feature that adds a sampled value:
        >>> import numpy as np
        >>>
        >>> feature = (
        ...     dt.Value(value=np.array([1, 2, 3]))
        ...     >> dt.Add(value=0.5)
        ... )

        Execute core logic manually:
        >>> output = feature.action()
        >>> output
        array([1.5, 2.5, 3.5])

        Use a list of inputs:
        >>> feature = (
        ...     dt.Value(value=[
        ...         np.array([1, 2, 3]),
        ...         np.array([4, 5, 6]),
        ...     ])
        ...     >> dt.Add(value=0.5)
        ... )
        >>> output = feature.action()
        >>> output
        [array([1.5, 2.5, 3.5]), array([4.5, 5.5, 6.5])]

        """

        # Retrieve the input images.
        image_list = self._input(_ID=_ID)

        # Get the current property values.
        feature_input = self.properties(_ID=_ID).copy()

        # Call the _process_properties hook, default does nothing.
        # For example, it can be used to ensure properties are formatted
        # correctly or to rescale properties.
        feature_input = self._process_properties(feature_input)
        if _ID != ():
            feature_input["_ID"] = _ID

        # Ensure that input is a list.
        image_list = self._format_input(image_list, **feature_input)

        # Set the seed from the hash_key. Ensures equal results.
        # self.seed(_ID=_ID)

        # _process_and_get calls the get function correctly according
        # to the __distributed__ attribute.
        new_list = self._process_and_get(image_list, **feature_input)

        self._process_output(new_list, feature_input)

        # Merge input and new_list.
        if self.__list_merge_strategy__ == MERGE_STRATEGY_OVERRIDE:
            image_list = new_list
        elif self.__list_merge_strategy__ == MERGE_STRATEGY_APPEND:
            image_list = image_list + new_list

        # For convencience, list images of length one are unwrapped.
        if len(image_list) == 1:
            return image_list[0]
        else:
            return image_list

    def update(
        self: Feature,
        **global_arguments: Any,
    ) -> Feature:
        """Refresh the feature to generate a new output.

        By default, when a feature is called multiple times, it returns the 
        same value.

        Calling `update()` forces the feature to recompute and 
        return a new value the next time it is evaluated.

        Parameters
        ----------
        **global_arguments: Any
            Deprecated. Has no effect. Previously used to inject values 
            during update. Use `Arguments` or call-time overrides instead.

        Returns
        -------
        Feature
            The updated feature instance, ensuring the next evaluation produces 
            a fresh result.

        Examples
        -------
        >>> import deeptrack as dt

        >>> import numpy as np
        >>>
        >>> feature = dt.Value(value=lambda: np.random.rand())
        >>> output1 = feature()
        >>> output1
        0.9173610765203623

        >>> output2 = feature()
        >>> output2  # Same as before
        0.9173610765203623

        >>> feature.update()  # Feature updated
        >>> output3 = feature()
        >>> output3
        0.13917950359184617

        """

        if global_arguments:
            import warnings

            # Deprecated, but not necessary to raise hard error.
            warnings.warn(
                "Passing information through .update is no longer supported. "
                "A quick fix is to pass the information when resolving the feature. "
                "The prefered solution is to use dt.Arguments",
                DeprecationWarning,
            )

        super().update()

        return self

    def add_feature(
        self: Feature,
        feature: Feature,
    ) -> Feature:
        """Add a feature to the dependecy graph of this one.

        This method establishes a dependency relationship by registering the 
        provided `feature` as a child node of the current feature. This ensures
        that its evaluation and property resolution are included in the current
        feature’s computation graph.

        Internally, it calls `feature.add_child(self)`, which automatically 
        handles graph integration and triggers recomputation if necessary.

        This is often used to define explicit data dependencies or to ensure 
        side-effect features are computed when this feature is resolved.

        Parameters
        ----------
        feature: Feature
            The feature to add as a dependency.

        Returns
        -------
        Feature
            The newly added feature (for chaining).

        Examples
        --------
        >>> import deeptrack as dt

        Define the main feature that adds a constant to the input:
        >>> feature = dt.Add(value=2)

        Define a side-effect feature:
        >>> dependency = dt.Value(value=42)

        Register the dependency so its state becomes part of the graph:
        >>> feature.add_feature(dependency)

        Execute the main feature on an input array:
        >>> import numpy as np
        >>>
        >>> result = feature(np.array([1, 2, 3]))
        >>> result
        array([3, 4, 5])

        Note that the `dependency` does not affect the result directly, but it
        will be tracked and updated as part of the pipeline's evaluation graph.
        This can be useful if the dependency affects any parameters of the main
        feature.

        """

        feature.add_child(self)
        # self.add_dependency(feature)  # Already done by add_child().

        return feature

    def  seed(
        self: Feature,
        updated_seed: int | None = None,
        _ID: tuple[int, ...] = (),
    ) -> int:
        """Seed all random number generators for reproducibility.

        This method sets the global random seed for Python's `random` module, 
        NumPy, and (if available) PyTorch. If `updated_seed` is provided, it 
        replaces the value of the internal `_random_seed` node before
        resolution.

        This method sets the following:
        - `random.seed(seed)` for Python's RNG
        - `np.random.seed(seed)` for NumPy
        - `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`

        The same seed will lead to deterministic behavior within each backend
        (e.g., `random`, NumPy or PyTorch), but not **across** them. NumPy and
        PyTorch use different RNG algorithms, so identical seeds will not
        generate the same random numbers across backends.

        Parameters
        ----------
        updated_seed: int or None, optional
            If provided, sets a fixed value for the internal `_random_seed`.
        _ID: tuple[int, ...], optional
            Unique identifier used to resolve the seed value. It defaults to
            `()`.

        Returns
        -------
        int
            The resolved seed value used for all RNGs.

        Examples
        --------
        >>> import deeptrack as dt

        **Using `random`**
        Define a feature that samples a random integer from 0 to 10 using the
        Python standard library's `random` module:
        >>> import random
        >>>
        >>> feature = dt.Value(lambda: random.randint(0, 10))
        >>> 
        >>> for _ in range(3):
        ...     print(f"output={feature.update()()} seed={feature.seed()}")
        output=3 seed=355549663
        output=5 seed=119234165
        output=9 seed=1956541335

        Each time `.update()` is called, the internal `_random_seed` is
        re-sampled and used to reseed the Python `random` module. This 
        produces a new deterministic seed, but different output values.

        Fix the seed to reuse it later for reproducibility:
        >>> seed = feature.seed()
        >>> seed
        1956541335

        Now reseed the feature with the same value before each update,
        to make the output deterministic and repeatable.
        >>> for _ in range(3):
        ...    feature.seed(seed)
        ...    print(f"output={feature.update()()} seed={feature.seed()}")
        output=5 seed=1933964715
        output=5 seed=1933964715
        output=5 seed=1933964715

        Since the random seed is fixed before each sample, the output is
        the same every time. Note: the seed reported after sampling may
        differ if it's re-sampled internally, but the output remains stable.

        **Using NumPy**
        Similar observations can be made with NumPy:
        >>> import numpy as np
        >>>
        >>> feature = dt.Value(lambda: np.random.randint(0, 10))

        **Using PyTorch**        
        Similar observations can be made with PyTorch:
        >>> import torch
        >>>
        >>> feature = dt.Value(lambda: torch.randint(0, 10, (1,)).item())

        """

        if updated_seed:
            self._random_seed.set_value(updated_seed)

        seed = self._random_seed(_ID=_ID)

        random.seed(seed)
        np.random.seed(seed)

        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        return seed

    def bind_arguments(
        self: Feature,
        arguments: Arguments | Feature,
    ) -> Feature:
        """Bind another feature’s properties as arguments to this feature.

        This method allows properties of `arguments` to be dynamically linked 
        to this feature, enabling shared configurations across multiple
        features. It is commonly used in advanced feature pipelines.

        This method is often used in combination with the `Arguments` feature,
        which provides a utility that helps manage and propagate feature
        arguments efficiently.

        The values from `arguments` override the corresponding feature’s own
        properties at call-time, but do not modify them permanently.

        Parameters
        ----------
        arguments: Arguments or Feature
            The feature whose properties will be bound as arguments to this
            feature.

        Returns
        -------
        Feature
            The current feature instance with bound arguments.

        Examples
        --------
        >>> import deeptrack as dt

        Create an `Arguments` feature:
        >>> arguments = dt.Arguments(scale=2.0)

        Bind it with a pipeline:
        >>> pipeline = dt.Value(value=3) >> dt.Add(value=1 * arguments.scale)
        >>> pipeline.bind_arguments(arguments)
        >>> result = pipeline()
        >>> result
        5.0

        Override the argument dynamically:
        >>> result = pipeline(scale=1.0)
        >>> result
        4.0

        Without binding, the result would be still 5.0 as `scale` would still
        be the original one.

        """

        self.arguments = arguments

        return self

    #TODO ***MG***
    def plot(
        self: Feature,
        input_image: np.ndarray | list[np.ndarray] | Image | list[Image] = None,
        resolve_kwargs: dict = None,
        interval: float = None,
        **kwargs: Any,
    ) -> Any:
        """Visualizes the output of the feature.

        This method resolves the feature and visualizes the result. If the output is 
        an `Image`, it displays it using `pyplot.imshow`. If the output is a list, it 
        creates an animation. In Jupyter notebooks, the animation is played inline 
        using `to_jshtml()`. In scripts, the animation is displayed using the 
        matplotlib backend.

        Any parameters in `kwargs` are passed to `pyplot.imshow`.

        Parameters
        ----------
        input_image: np.ndarray or Image or list[np.ndarray or Image], optional
            The input image or list of images passed as an argument to the `resolve` 
            call. If `None`, uses previously set input values or propagates properties.
        resolve_kwargs: dict, optional
            Additional keyword arguments passed to the `resolve` call.
        interval: float, optional
            The time between frames in the animation, in milliseconds. The default 
            value is 33 ms.
        **kwargs: dict, optional
            Additional keyword arguments passed to `pyplot.imshow`.
       
        Returns
        -------
        Any
            The output of the feature or pipeline after execution.

        """

        from IPython.display import HTML, display

        # if input_image is not None:
        #     input_image = [Image(input_image)]

        output_image = self.resolve(input_image, **(resolve_kwargs or {}))

        # If a list, assume video
        if not isinstance(output_image, list):
            # Single image
            plt.imshow(output_image, **kwargs)
            return plt.gca()

        # Assume video
        fig = plt.figure()
        images = []
        plt.axis("off")
        for image in output_image:
            images.append([plt.imshow(image, **kwargs)])

        if not interval:
            if isinstance(output_image[0], Image):
                interval = output_image[0].get_property("interval") or (1 / 30 * 1000)
            else:
                interval = 1 / 30 * 1000

        anim = animation.ArtistAnimation(
            fig, images, interval=interval, blit=True, repeat_delay=0
        )

        try:
            get_ipython  # Throws NameError if not in Notebook
            display(HTML(anim.to_jshtml()))
            return anim

        except NameError:
            # Not in an notebook
            plt.show()

        except RuntimeError:
            # In notebook, but animation failed
            import ipywidgets as widgets

            def plotter(frame=0):
                plt.imshow(output_image[frame][:, :, 0], **kwargs)
                plt.show()

            return widgets.interact(
                plotter,
                frame=widgets.IntSlider(
                    value=0, min=0, max=len(images) - 1, step=1
                ),
            )

    #TODO ***AL***
    def _normalize(
        self: Feature,
        **properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize the properties.

        This method handles all unit normalizations and conversions. For each
        class in the method resolution order (MRO), it checks if the class has
        a `__conversion_table__` attribute. If found, it calls the `convert`
        method of the conversion table using the properties as arguments.

        Parameters
        ----------
        **properties: dict[str, Any]
            The properties to be normalized and converted.

        Returns
        -------
        dict[str, Any]
            The normalized and converted properties.

        Examples
        --------
        TODO

        """

        for cl in type(self).mro():
            if hasattr(cl, "__conversion_table__"):
                properties = cl.__conversion_table__.convert(**properties)

        for key, val in properties.items():
            if isinstance(val, Quantity):
                properties[key] = val.magnitude

        return properties

    def _process_properties(
        self: Feature,
        propertydict: dict[str, Any],
    ) -> dict[str, Any]:
        """Preprocess the input properties before calling `.get()`.

        This method acts as a preprocessing hook for subclasses, allowing them 
        to modify or normalize input properties before the feature's main 
        computation.

        Notes:
        - Calls `_normalize()` internally to standardize input properties.
        - Subclasses may override this method to implement additional 
          preprocessing steps.

        Parameters
        ----------
        propertydict: dict[str, Any]
            The dictionary of properties to be processed before being passed 
            to the `.get()` method.

        Returns
        -------
        dict[str, Any]
            The processed property dictionary after normalization.

        Examples
        --------
        TODO

        """

        propertydict = self._normalize(**propertydict)

        return propertydict

    def _activate_sources(
        self: Feature,
        x: SourceItem | list[SourceItem] | Any,
    ) -> None:
        """Activates source items within the given input.

        This method checks whether the input `x` or its elements (if `x` is a 
        list) are instances of `SourceItem`. If so, the source is called to 
        trigger its behavior—typically to update or emit a new value. This is 
        necessary to ensure source-driven features (e.g., time-dependent or 
        externally updated values) are evaluated when the pipeline is run.

        Non-`SourceItem` elements in `x` are ignored.

        This method is typically invoked at the beginning of `__call__()` to 
        activate all relevant sources before resolving a feature.

        Parameters
        ----------
        x: SourceItem or list[SourceItem] or Any
            The input to process. If `x` is a `SourceItem`, it is activated.
            If `x` is a list, each `SourceItem` within the list is activated.
            If `x` is `None` or contains no sources, the method has no effect.

        Examples
        --------
        >>> import deeptrack as dt

        Create a dummy source that prints when called:
        >>> class MySource(dt.sources.SourceItem):
        ...     def __call__(self):
        ...         print("Source activated")

        Instantiate a feature and manually activate a source:
        >>> feature = dt.Value(value=1)
        >>> source = MySource(callbacks=[])
        >>> feature._activate_sources(source)
        Source activated

        Use a list of sources:
        >>> feature._activate_sources([source, 42, "text"])
        Source activated

        """

        if isinstance(x, SourceItem):
            x()
        elif isinstance(x, list):
            for source in x:
                if isinstance(source, SourceItem):
                    source()

    def __getattr__(
        self: Feature,
        key: str,
    ) -> Any:
        """Access properties of the feature as if they were attributes.

        This method allows dynamic access to the feature's properties via 
        standard attribute syntax. For example, `feature.my_property` is 
        equivalent to:

        >>> feature.properties["my_property"]`()

        This is only called if the attribute is not found via the normal lookup
        process (i.e., it's not a real attribute or method). It checks whether
        `key` exists in the `properties` dictionary, and if so, returns the
        corresponding `Property` instance.

        Parameters
        ----------
        key: str
            The name of the attribute being accessed.

        Returns
        -------
        Any
            The corresponding property if it exists in `self.properties`.

        Raises
        ------
        AttributeError
            If `properties` is not set, or if `key` does not exist in it.

        Examples
        --------
        >>> import deeptrack as dt

        Create a feature with a property:
        >>> feature = dt.DummyFeature(value=42)

        Access the property as an attribute:
        >>> feature.value()
        42

        Attempting to access a non-existent property raises an `AttributeError`:
        >>> feature.nonexistent()
        ...
        AttributeError: 'DummyFeature' object has no attribute 'nonexistent'

        """

        if "properties" in self.__dict__:
            properties = self.__dict__["properties"]
            if key in properties:
                return properties[key]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def __iter__(
        self: Feature,
    ) -> Feature:
        """Return self as an iterator over feature values.

        This makes the `Feature` object compatible with Python's iterator 
        protocol. Each call to `next(feature)` generates a new output by 
        resampling its properties and resolving the pipeline.

        Returns
        -------
        Feature
            Returns self, which defines `__next__()` to yield outputs.

        Examples
        --------
        >>> import deeptrack as dt

        Create feature:
        >>> import numpy as np
        >>>
        >>> feature = dt.Value(value=lambda: np.random.rand())

        Use the feature in a loop:
        >>> for sample in feature:
        ...     print(sample)
        ...     if sample > 0.5:
        ...         break
        0.43126475134786546
        0.3270413736199965
        0.6734339603677173

        """

        return self

        #TODO ***BM*** TBE? Previous implementation, not standard in Python
        # while True:
        #     yield from next(self)

    def __next__(
        self: Feature,
    ) -> Any:
        """Return the next resolved feature in the sequence.

        This method allows a `Feature` to be used as an iterator that yields
        a new result at each step. It is called automatically by `next(feature)`
        or when used in iteration.

        Each call to `__next__()` triggers a resampling of all properties and
        evaluation of the pipeline using `self.update().resolve()`.

        Returns
        -------
        Any
            A newly generated output from the feature.

        Examples
        --------
        >>> import deeptrack as dt

        Create a feature:
        >>> import numpy as np
        >>>
        >>> feature = dt.Value(value=lambda: np.random.rand())

        Get a single sample:
        >>> next(feature)
        0.41251758103924216

        """

        return self.update().resolve()

        #TODO ***BM*** TBE? Previous implementation, not standard in Python
        # yield self.update().resolve()

    def __rshift__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Chains this feature with another feature or function using '>>'.

        This operator enables pipeline-style chaining. The expression:

        >>> feature >> other

        creates a new pipeline where the output of `feature` is passed as 
        input to `other`.

        If `other` is a `Feature` or `DeepTrackNode`, this returns a 
        `Chain(feature, other)`. If `other` is a callable (e.g., a function),
        it is wrapped using `dt.Lambda(lambda: other)` and chained 
        similarly. The lambda returns the function itself, which is then 
        automatically called with the upstream feature’s output during 
        evaluation.

        If `other` is neither a `DeepTrackNode` nor a callable, the operator 
        is not implemented and returns `NotImplemented`, which may lead to a 
        `TypeError` if no matching reverse operator is defined.

        Parameters
        ----------
        other: Any
            The feature, node, or callable to chain after `self`.

        Returns
        -------
        Feature
            A new chained feature combining `self` and `other`.

        Raises
        ------
        TypeError
            If `other` is not a `DeepTrackNode` or callable, the operator 
            returns `NotImplemented`, which may raise a `TypeError` if no 
            matching reverse operator is defined.

        Examples
        --------
        >>> import deeptrack as dt

        Chain two features:
        >>> feature1 = dt.Value(value=[1, 2, 3])
        >>> feature2 = dt.Add(value=1)
        >>> pipeline = feature1 >> feature2
        >>> result = pipeline()
        >>> result
        [2, 3, 4]

        Chain with a callable (e.g., NumPy function):
        >>> import numpy as np
        >>>
        >>> feature = dt.Value(value=np.array([1, 2, 3]))
        >>> function = np.mean
        >>> pipeline = feature >> function
        >>> result = pipeline()
        >>> result
        2.0

        This is equivalent to:
        >>> pipeline = feature >> dt.Lambda(lambda: function)

        The lambda returns the function object. During evaluation, DeepTrack 
        internally calls that function with the resolved output of `feature`.

        Attempting to chain with an unsupported object raises a TypeError:
        >>> feature >> "invalid"
            ...
        TypeError: unsupported operand type(s) for >>: 'Value' and 'str'

        """

        if isinstance(other, DeepTrackNode):
            return Chain(self, other)

        # If other is a function, call it on the output of the feature.
        # For example, feature >> some_function
        if callable(other):
            return self >> Lambda(lambda: other)

        # The operator is not implemented for other inputs.
        return NotImplemented

    def __rrshift__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Chains another feature or value with this feature using '>>'.

        This operator supports chaining when the `Feature` appears on the 
        right-hand side of a pipeline. The expression:

        >>> other >> feature

        triggers `feature.__rrshift__(other)` if `other` does not implement 
        `__rshift__`, or if its implementation returns `NotImplemented`.

        If `other` is a `Feature`, this is equivalent to:

        >>> dt.Chain(other, feature)

        If `other` is a raw value (e.g., a list or array), it is wrapped using
        `dt.Value(value=other)` before chaining:

        >>> dt.Chain(dt.Value(value=other), feature)

        Parameters
        ----------
        other: Any
            The value or feature to be evaluated before this feature.

        Returns
        -------
        Feature
            A new chained feature where `other` is evaluated first.

        Raises
        ------
        TypeError
            If `other` is not a supported type, this method returns 
            `NotImplemented`, which may raise a `TypeError` if no matching 
            forward operator is defined.

        Notes
        -----
        This method enables chaining where a `Feature` appears on the
        right-hand side of the `>>` operator. It is triggered when the
        left-hand operand does not implement `__rshift__`, or when its
        implementation returns `NotImplemented`.

        This is particularly useful when chaining two `Feature` instances or
        when the left-hand operand is a custom class designed to delegate
        chaining behavior. For example:

        >>> pipeline = dt.Value(value=[1, 2, 3]) >> dt.Add(value=1)

        In this case, if `dt.Value` does not handle `__rshift__`, Python will
        fall back to calling `Add.__rrshift__(...)`, which constructs the
        chain.

        However, this mechanism does **not** apply to built-in types like
        `int`, `float`, or `list`. Due to limitations in Python's operator
        overloading, expressions like:

        >>> 1 >> dt.Add(value=1)
        >>> [1, 2, 3] >> dt.Add(value=1)

        will raise `TypeError`, because Python does not delegate to the
        right-hand operand’s `__rrshift__` method for built-in types.

        To chain a raw value into a feature, wrap it explicitly using
        `dt.Value`:

        >>> dt.Value(1) >> dt.Add(value=1)

        This is functionally equivalent and avoids the need for fallback
        behavior.

        """

        if isinstance(other, Feature):
            return Chain(other, self)

        if isinstance(other, DeepTrackNode):
            return Chain(Value(other), self)

        return NotImplemented

    def __add__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Adds another value or feature using '+'.

        This operator is shorthand for chaining with `dt.Add`. The expression:

        >>> feature + other

        is equivalent to:

        >>> feature >> dt.Add(value=other)

        Internally, this method constructs a new `Add` feature and uses the 
        right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to be added. It is passed to `Add` as the
            `value` argument.

        Returns
        -------
        Feature
            A new feature that adds `other` to the output of `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Add a constant value to a static input:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature + 5
        >>> result = pipeline()
        >>> result
        [6, 7, 8]

        This is equivalent to:
        >>> pipeline = feature >> dt.Add(value=5)

        Add a dynamic feature that samples values at each call:
        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = feature + noise
        >>> result = pipeline.update()()
        >>> result
        [1.325563919290048, 2.325563919290048, 3.325563919290048]

        This is equivalent to:
        >>> pipeline = feature >> dt.Add(value=noise)

        """

        return self >> Add(other)

    def __radd__(
        self: Feature,
        other: Any
    ) -> Feature:
        """Adds this feature to another value using right '+'.

        This operator is the right-hand version of `+`, enabling expressions 
        where the `Feature` appears on the right-hand side. The expression:

        >>> other + feature

        is equivalent to:

        >>> dt.Value(value=other) >> dt.Add(value=feature)

        Internally, this method constructs a `Value` feature from `other` and 
        chains it into an `Add` feature that adds the current feature as a 
        dynamic value.

        Parameters
        ----------
        other: Any
            A constant or `Feature` to which `self` will be added. It is 
            passed as the input to `Value`.

        Returns
        -------
        Feature
            A new feature that adds `self` to `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Add a feature to a constant:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 5 + feature
        >>> result = pipeline()
        >>> result
        [6, 7, 8]

        This is equivalent to:
        >>> pipeline = dt.Value(value=5) >> dt.Add(value=feature)

        Add a feature to a dynamic value:
        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = noise + feature
        >>> result = pipeline.update()()
        >>> result
        [1.5254613210875014, 2.5254613210875014, 3.5254613210875014]

        This is equivalent to:
        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.rand())
        ...     >> dt.Add(value=feature)
        ... )

        """

        return Value(other) >> Add(self)

    def __sub__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Subtract another value or feature using '-'.

        This operator is shorthand for chaining with `Subtract`.
        The expression:

        >>> feature - other

        is equivalent to:

        >>> feature >> dt.Subtract(value=other)

        Internally, this method constructs a new `Subtract` feature and uses
        the right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to be subtracted. It is passed to
            `Subtract` as the `value` argument.

        Returns
        -------
        Feature
            A new feature that subtracts `other` from the output of `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Subtract a constant value from a static input:
        >>> feature = dt.Value(value=[5, 6, 7])
        >>> pipeline = feature - 2
        >>> result = pipeline()
        >>> result
        [3, 4, 5]

        This is equivalent to:
        >>> pipeline = feature >> dt.Subtract(value=2)

        Subtract a dynamic feature that samples a value at each call:
        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = feature - noise
        >>> result = pipeline.update()()
        >>> result
        [4.524072925059197, 5.524072925059197, 6.524072925059197]

        This is equivalent to:
        >>> pipeline = feature >> dt.Subtract(value=noise)
        
        """

        return self >> Subtract(other)

    def __rsub__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Subtract this feature from another value using right '-'.

        This operator is the right-hand version of `-`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression:

        >>> other - feature

        is equivalent to:

        >>> dt.Value(value=other) >> dt.Subtract(value=feature)

        Internally, this method constructs a `Value` feature from `other` and
        chains it into a `Subtract` feature that subtracts the current feature
        as a dynamic value.

        Parameters
        ----------
        other: Any
            A constant or `Feature` to which `self` will be subtracted. It is
            passed as the input to `Value`.

        Returns
        -------
        Feature
            A new feature that subtracts `self` from `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Subtract a feature from a constant:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 5 - feature
        >>> result = pipeline()
        >>> result
        [4, 3, 2]

        This is equivalent to:
        >>> pipeline = dt.Value(value=5) >> dt.Subtract(value=feature)

        Subtract a feature from a dynamic value:
        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = noise - feature
        >>> result = pipeline.update()()
        >>> result
        [-0.18761746914784516, -1.1876174691478452, -2.1876174691478454]

        This is equivalent to:
        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.rand())
        ...     >> dt.Subtract(value=feature)
        ... )

        """

        return Value(other) >> Subtract(self)

    def __mul__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Multiply this feature with another value using '*'.

        This operator is shorthand for chaining with `Multiply`.
        The expression:

        >>> feature * other

        is equivalent to:

        >>> feature >> dt.Multiply(value=other)

        Internally, this method constructs a new `Multiply` feature and uses
        the right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to be multiplied. It is passed to
            `dt.Multiply` as the `value` argument.

        Returns
        -------
        Feature
            A new feature that multiplies `other` to the output of `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Multiply a constant value to a static input:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature * 2
        >>> result = pipeline()
        >>> result
        [2, 4, 6]

        This is equivalent to:
        >>> pipeline = feature >> dt.Multiply(value=2)

        Multiply with a dynamic feature that samples a value at each call:
        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = feature * noise
        >>> result = pipeline.update()()
        >>> result
        [0.2809370704818722, 0.5618741409637444, 0.8428112114456167]

        This is equivalent to:
        >>> pipeline = feature >> dt.Multiply(value=noise)

        """

        return self >> Multiply(other)

    def __rmul__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Multiply another value with this feature using right '*'.

        This operator is the right-hand version of `*`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression:

        >>> other * feature

        is equivalent to:

        >>> dt.Value(value=other) >> dt.Multiply(value=feature)

        Internally, this method constructs a `Value` feature from `other` and
        chains it into a `Multiply` feature that multiplies the current feature
        as a dynamic value.

        Parameters
        ----------
        other: Any
            A constant or `Feature` that will be multiplied by `self`. It is
            passed as the input to `Value`.

        Returns
        -------
        Feature
            A new feature that muliplies `self` by `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Multiply a feature to a constant:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 2 * feature
        >>> result = pipeline()
        >>> result
        [2, 4, 6]

        This is equivalent to:
        >>> pipeline = dt.Value(value=2) >> dt.Multiply(value=feature)

        Multiply a feature to a dynamic value:
        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = noise * feature
        >>> result = pipeline.update()()
        >>> result
        [0.8784860790329121, 1.7569721580658242, 2.635458237098736]

        This is equivalent to:
        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.rand())
        ...     >> dt.Multiply(value=feature)
        ... )

        """

        return Value(other) >> Multiply(self)

    #TODO ***AL***
    def __truediv__(
        self: Feature, 
        other: Any
        ) -> Feature:
        """Divides this feature by another value using '/'.

        """

        return self >> Divide(other)

    #TODO ***AL***
    def __rtruediv__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Divides another value by this feature using right '/'.

        """

        return Value(other) >> Divide(self)

    #TODO ***AL***
    def __floordiv__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Performs floor division using '//'.

        """

        return self >> FloorDivide(other)

    #TODO ***AL***
    def __rfloordiv__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Performs right floor division using '//'.

        """

        return Value(other) >> FloorDivide(self)

    def __pow__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Raise this feature (base) to a power (exponent) using '**'.

        This operator is shorthand for chaining with `Power`. The expression:

        >>> feature ** other

        is equivalent to:

        >>> feature >> dt.Power(value=other)

        Internally, this method constructs a new `Power` feature and uses the
        right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` representing the exponent. It is passed to
            `Power` as the `value` argument.

        Returns
        -------
        Feature
            A new feature representing `self` to the power of `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Raise a static base to a constant exponent:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature ** 3
        >>> result = pipeline()
        >>> result
        [1, 8, 27]

        This is equivalent to:
        >>> pipeline = feature >> dt.Power(value=3)

        Raise to a dynamic exponent that samples values at each call:
        >>> import numpy as np
        >>>
        >>> random_exponent = dt.Value(value=lambda: np.random.randint(10))
        >>> pipeline = feature ** random_exponent
        >>> result = pipeline.update()()
        >>> result
        [1, 64, 729]

        This is equivalent to:
        >>> pipeline = feature >> dt.Power(value=random_exponent)
 
        """

        return self >> Power(other)

    #TODO ***JH***
    def __rpow__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Raises another value to this feature as a power using right '**'.

        """

        return Value(other) >> Power(self)

    #TODO ***JH***
    def __gt__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Checks if this feature is greater than another using '>'.

        """

        return self >> GreaterThan(other)

    #TODO ***JH***
    def __rgt__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if another value is greater than this feature using 
        right '>'.

        """

        return Value(other) >> GreaterThan(self)

    #TODO ***JH***
    def __lt__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if this feature is less than another using '<'.

        """

        return self >> LessThan(other)

    #TODO ***JH***
    def __rlt__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if another value is less than this feature using right '<'.

        """
        
        return Value(other) >> LessThan(self)

    #TODO ***JH***
    def __le__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if this feature is less than or equal to another using '<='.

        """

        return self >> LessThanOrEquals(other)

    #TODO ***JH***
    def __rle__(
        self: Feature,
        other: Any
    ) -> Feature:
        """Checks if another value is less than or equal to this feature using 
        right '<='.

        """

        return Value(other) >> LessThanOrEquals(self)

    #TODO ***JH***
    def __ge__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if this feature is greater than or equal to another 
        using '>='.

        """

        return self >> GreaterThanOrEquals(other)

    #TODO ***JH***
    def __rge__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if another value is greater than or equal to this feature 
        using right '>='.

        """

        return Value(other) >> GreaterThanOrEquals(self)

    #TODO ***JH***
    def __xor__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Repeats the feature a given number of times using '^'.
        
        """

        return Repeat(self, other)

    #TODO ***JH***
    def __and__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Stacks this feature with another using '&'.

        """

        return self >> Stack(other)

    #TODO ***JH***
    def __rand__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Stacks another value with this feature using right '&'.
        
        """
        
        return Value(other) >> Stack(self)

    def __getitem__(
        self: Feature,
        slices: Any,
    ) -> Feature:
        """Allows direct slicing of the feature's output.

        This operator enables syntax like:

        >>> feature[:, 0]

        to extract a slice from the output of the feature, just as one would 
        with a NumPy array or PyTorch tensor.

        Internally, this is equivalent to chaining with `dt.Slice`, and the 
        expression:

        >>> feature[slices]

        is equivalent to:

        >>> feature >> dt.Slice(slices)

        If the slice is not already a tuple (i.e., a single index or slice),
        it is wrapped in one. The resulting tuple is converted to a list to 
        allow sampling of dynamic slices at runtime.

        Parameters
        ----------
        slices: Any
            The slice or index to apply to the feature output. Can be an int, 
            slice object, or a tuple of them.

        Returns
        -------
        Feature
            A new feature that applies slicing to the output of the current 
            feature.

        Examples
        --------
        >>> import deeptrack as dt

        Create a feature:
        >>> import numpy as np
        >>>
        >>> feature = dt.Value(value=np.arange(9).reshape(3, 3))
        >>> feature()
        array([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])

        Slice a row:
        >>> sliced = feature[1]
        >>> sliced()
        array([3, 4, 5])

        This is equivalent to:
        >>> sliced = feature >> dt.Slice([1])

        Slice with multiple axes:
        >>> sliced = feature[1:, 1:]
        >>> sliced()
        array([[4, 5],
               [7, 8]])

        This is equivalent to:
        >>> sliced = feature >> dt.Slice([slice(1, None), slice(1, None)])

        """

        if not isinstance(slices, tuple):
            slices = (slices,)

        # Make it a list to ensure that each element is sampled independently.
        slices = list(slices)

        return self >> Slice(slices)

    # Private properties to dispatch based on config.
    @property
    def _format_input(self: Feature) -> Callable[[Any], list[Any or Image]]:
        """Select the appropriate input formatting function for configuration.

        Returns either `_image_wrapped_format_input` or
        `_no_wrap_format_input`, depending on whether image metadata
        (properties) should be preserved and processed downstream.

        This selection is controlled by the `_wrap_array_with_image` flag.

        Returns
        -------
        Callable
            A function that formats the input into a list of Image objects or
            raw arrays, depending on the configuration.

        """

        if self._wrap_array_with_image:
            return self._image_wrapped_format_input

        return self._no_wrap_format_input

    @property
    def _process_and_get(self: Feature) -> Callable[[Any], list[Any or Image]]:
        """Select the appropriate processing function based on configuration.

        Returns a method that applies the feature’s transformation (`get`) to
        the input data, either with or without wrapping and preserving `Image`
        metadata.

        The decision is based on the `_wrap_array_with_image` flag:
        - If `True`, returns `_image_wrapped_process_and_get`
        - If `False`, returns `_no_wrap_process_and_get`

        Returns
        -------
        Callable
            A function that applies `.get()` to the input, either preserving
            or ignoring metadata depending on configuration.

        """

        if self._wrap_array_with_image:
            return self._image_wrapped_process_and_get

        return self._no_wrap_process_and_get

    @property
    def _process_output(self: Feature) -> Callable[[Any], None]:
        """Select the appropriate output processing function for configuration.

        Returns a method that post-processes the outputs of the feature,
        typically after the `get()` method has been called. The selected method
        depends on whether the feature is configured to wrap outputs in `Image`
        objects (`_wrap_array_with_image = True`).

        - If `True`, returns `_image_wrapped_process_output`, which appends
          feature properties to each `Image`.
        - If `False`, returns `_no_wrap_process_output`, which extracts raw
          array values from any `Image` instances.

        Returns
        -------
        Callable
            A post-processing function for the feature output.

        """

        if self._wrap_array_with_image:
            return self._image_wrapped_process_output

        return self._no_wrap_process_output

    def _image_wrapped_format_input(
        self: Feature,
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image] | None,
        **kwargs: Any,
    ) -> list[Image]:
        """Wrap input data as Image instances before processing.

        This method ensures that all elements in the input are `Image`
        objects. If any raw arrays are provided, they are wrapped in `Image`.
        This allows features to propagate metadata and store properties in the
        output.

        Parameters
        ----------
        image_list: np.ndarray or list[np.ndarray] or Image or list[Image] or None
            The input to the feature. If not a list, it is converted into a
            single-element list. If `None`, it returns an empty list.

        Returns
        -------
        list[Image]
            A list where all items are instances of `Image`.

        """

        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]

        return [(Image(image)) for image in image_list]

    def _no_wrap_format_input(
        self: Feature,
        image_list: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Process input data without wrapping it as Image instances.

        This method returns the input list as-is (after ensuring it is a list).
        It is used when metadata is not needed or performance is a concern.

        Parameters
        ----------
        image_list: Any
            The input to the feature. If not already a list, it is wrapped in
            one. If `None`, it returns an empty list.

        Returns
        -------
        list[Any]
            A list of raw input elements, without any transformation.

        """

        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]

        return image_list

    def _image_wrapped_process_and_get(
        self: Feature,
        image_list: Image | list[Image] | Any | list[Any],
        **feature_input: dict[str, Any],
    ) -> list[Image]:
        """Processes input data while maintaining Image properties.

        This method applies the `get()` method to the input while ensuring that
        output values are wrapped as `Image` instances and preserve the 
        properties of the corresponding input images.

        If `__distributed__ = True`, `get()` is called separately for each 
        input image. If `False`, the full list is passed to `get()` at once.

        Parameters
        ----------
        image_list: Image or list[Image] or Any or list[Any]
            The input data to be processed.
        **feature_input: dict[str, Any]
            The keyword arguments containing the sampled properties to pass 
            to the `get()` method.

        Returns
        -------
        list[Image]
            The list of processed images, with properties preserved.

        """

        if self.__distributed__:
            # Call get on each image in list, and merge properties from
            # corresponding image.

            results = []

            for image in image_list:
                output = self.get(image, **feature_input)
                if not isinstance(output, Image):
                    output = Image(output)

                output.merge_properties_from(image)
                results.append(output)

            return results

        # ELse, call get on entire list.
        new_list = self.get(image_list, **feature_input)

        if not isinstance(new_list, list):
            new_list = [new_list]

        for idx, image in enumerate(new_list):
            if not isinstance(image, Image):
                new_list[idx] = Image(image)
        return new_list

    def _no_wrap_process_and_get(
        self: Feature,
        image_list: Any | list[Any],
        **feature_input: dict[str, Any],
    ) -> list[Any]:
        """Process input data without additional wrapping and retrieve results.

        This method applies the `get()` method to the input without wrapping 
        results in `Image` objects, and without propagating or merging metadata.

        If `__distributed__ = True`, `get()` is called separately for each 
        element in the input list. If `False`, the full list is passed to 
        `get()` at once.

        Parameters
        ----------
        image_list: Any or list[Any]
            The input data to be processed.
        **feature_input: dict
            The keyword arguments containing the sampled properties to pass 
            to the `get()` method.

        Returns
        -------
        list[Any]
            The list of processed outputs (raw arrays, tensors, etc.).

        """

        if self.__distributed__:
            # Call get on each image in list, and merge properties from
            # corresponding image

            return [self.get(x, **feature_input) for x in image_list]

        # Else, call get on entire list.
        new_list = self.get(image_list, **feature_input)

        if not isinstance(new_list, list):
            new_list = [new_list]

        return new_list

    def _image_wrapped_process_output(
        self: Feature,
        image_list: Image | list[Image] | Any | list[Any],
        feature_input: dict[str, Any],
    ) -> None:
        """Append feature properties and input data to each Image.

        This method is called after `get()` when the feature is set to wrap
        its outputs in `Image` instances. It appends the sampled properties
        (from `feature_input`) to the metadata of each `Image`. If the feature
        is bound to an `arguments` object, those properties are also appended.

        Parameters
        ----------
        image_list: list[Image]
            The output images from the feature.
        feature_input: dict[str, Any]
            The resolved property values used during this evaluation.

        """

        for index, image in enumerate(image_list):
            if self.arguments:
                image.append(self.arguments.properties())
            image.append(feature_input)

    def _no_wrap_process_output(
        self: Feature,
        image_list: Any | list[Any],
        feature_input: dict[str, Any],
    ) -> None:
        """Extract and update raw values from Image instances.

        This method is called after `get()` when the feature is not configured
        to wrap outputs as `Image` instances. If any `Image` objects are
        present in the output list, their underlying array values are extracted
        using `.value` (i.e., `image._value`).

        Parameters
        ----------
        image_list: list[Any]
            The list of outputs returned by the feature.
        feature_input: dict[str, Any]
            The resolved property values used during this evaluation (unused).

        """

        for index, image in enumerate(image_list):
            if isinstance(image, Image):
                image_list[index] = image._value


def propagate_data_to_dependencies(feature: Feature, **kwargs: dict[str, Any]) -> None:
    """Updates the properties of dependencies in a feature's dependency tree.

    This function traverses the dependency tree of the given feature and 
    updates the properties of each dependency based on the provided keyword 
    arguments. Only properties that already exist in the `PropertyDict` of a 
    dependency are updated.

    By dynamically updating the properties in the dependency tree, this 
    function ensures that any changes in the feature's context or configuration
    are propagated correctly to its dependencies.

    Parameters
    ----------
    feature: Feature
        The feature whose dependencies are to be updated. The dependencies are 
        recursively traversed to ensure that all relevant nodes in the 
        dependency tree are considered.
    **kwargs: dict of str, Any
        Key-value pairs specifying the property names and their corresponding 
        values to be set in the dependencies. Only properties that exist in the
        `PropertyDict` of a dependency will be updated.

    Examples
    --------
    >>> import deeptrack as dt

    Update the properties of a feature and its dependencies:
    >>> feature = dt.DummyFeature(value=10)
    >>> dt.propagate_data_to_dependencies(feature, value=20)
    >>> feature.value()
    20

    This will update the `value` property of the `feature` and its 
    dependencies, provided they have a property named `value`.

    """

    for dep in feature.recurse_dependencies():
        if isinstance(dep, PropertyDict):
            for key, value in kwargs.items():
                if key in dep:
                    dep[key].set_value(value)


class StructuralFeature(Feature):
    """Provide the structure of a feature set without input transformations.

    A `StructuralFeature` does not modify the input data or introduce new
    properties. Instead, it serves as a logical and organizational tool for
    grouping, chaining, or structuring pipelines.

    This feature is typically used to:
    - group or chain sub-features (e.g., `Chain`)
    - apply conditional or sequential logic (e.g., `Probability`)
    - organize pipelines without affecting data flow (e.g., `Combine`)

    `StructuralFeature` inherits all behavior from `Feature`, without
    overriding `__init__` or `get`.

    Attributes
    ----------
    __property_verbosity__ : int
        Controls whether this feature's properties appear in the output image's
        property list. A value of `2` hides them from output.
    __distributed__ : bool
        If `True`, applies `get` to each element in a list individually.
        If `False`, processes the entire list as a single unit. It defaults to
        `False`.

    """

    __property_verbosity__: int = 2  # Hide properties from logs or output
    __distributed__: bool = False  # Process the entire image list in one call


class Chain(StructuralFeature):
    """Resolve two features sequentially.

    Applies two features sequentially: the output of `feature_1` is passed as
    input to `feature_2`. This allows combining simple operations into complex
    pipelines.

    This is equivalent to using the `>>` operator:

    >>> dt.Chain(A, B) ≡ A >> B

    Parameters
    ----------
    feature_1: Feature
        The first feature in the chain. Its output is passed to `feature_2`.
    feature_2: Feature
        The second feature in the chain, which processes the output from 
        `feature_1`.
    **kwargs: Any, optional
        Additional keyword arguments passed to the parent `StructuralFeature` 
        (and, therefore, `Feature`).

    Methods
    -------
    `get(image: Any, _ID: tuple[int, ...], **kwargs: Any) -> Any`
        Apply the two features in sequence on the given input image.

    Examples
    --------
    >>> import deeptrack as dt

    Create a feature chain where the first feature adds a constant offset, and 
    the second feature multiplies the result by a constant:
    >>> A = dt.Add(value=10)
    >>> M = dt.Multiply(value=0.5)
    >>>
    >>> chain = A >> M

    Equivalent to: 
    >>> chain = dt.Chain(A, M)

    Create a dummy image:
    >>> import numpy as np
    >>>
    >>> dummy_image = np.zeros((2, 4))

    Apply the chained features:
    >>> chain(dummy_image)
    array([[5., 5., 5., 5.],
        [5., 5., 5., 5.]])

    """

    def __init__(
        self: Chain,
        feature_1: Feature,
        feature_2: Feature,
        **kwargs: Any,
    ):
        """Initialize the chain with two sub-features.

        This constructor initializes the feature chain by setting `feature_1` 
        and `feature_2` as dependencies. Updates to these sub-features 
        automatically propagate through the DeepTrack computation graph, 
        ensuring consistent evaluation and execution.

        Parameters
        ----------
        feature_1: Feature
            The first feature to be applied.
        feature_2: Feature
            The second feature, applied to the result of `feature_1`.
        **kwargs: Any
            Additional keyword arguments passed to the parent constructor
            (e.g., name, properties).

        """

        super().__init__(**kwargs)

        self.feature_1 = self.add_feature(feature_1)
        self.feature_2 = self.add_feature(feature_2)

    def get(
        self: Feature,
        image: Any,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Apply the two features sequentially to the given input image(s).

        This method first applies `feature_1` to the input image(s) and then
        passes the output through `feature_2`.

        Parameters
        ----------
        image: Any
            The input data to transform sequentially. Most typically, this is
            a NumPy array, a PyTorch tensor, or an Image.
        _ID: tuple[int, ...], optional
            A unique identifier for caching or parallel execution. It defaults
            to an empty tuple.
        **kwargs: Any
            Additional parameters passed to or sampled by the features. These
            are generally unused here, as each sub-feature fetches its required
            properties internally.

        Returns
        -------
        Any
            The final output after `feature_1` and then `feature_2` have
            processed the input.

        """

        image = self.feature_1(image, _ID=_ID)
        image = self.feature_2(image, _ID=_ID)
        return image


Branch = Chain  # Alias for backwards compatibility.


class DummyFeature(Feature):
    """A no-op feature that simply returns the input unchanged.

    This class can serve as a container for properties that don't directly 
    transform the data but need to be logically grouped. 
    
    Since it inherits from `Feature`, any keyword arguments passed to the
    constructor are stored as `Property` instances in `self.properties`,
    enabling dynamic behavior or parameterization without performing any
    transformations on the input data.

    Parameters
    ----------
    _input: Any, optional
        An optional input (typically an image or list of images) that can be
        set for the feature. It defaults to an empty list [].
    **kwargs: Any
        Additional keyword arguments are wrapped as `Property` instances and 
        stored in `self.properties`.

    Methods
    -------
    `get(image: Any, **kwargs: Any) -> Any`
        It simply returns the input image(s) unchanged.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an image and pass it through a `DummyFeature` to demonstrate 
    no changes to the input data:
    >>> dummy_image = np.ones((60, 80))

    Initialize the DummyFeature:
    >>> dummy_feature = dt.DummyFeature(value=42)

    Pass the image through the DummyFeature:
    >>> output_image = dummy_feature(dummy_image)

    Verify the output is identical to the input:
    >>> np.array_equal(dummy_image, output_image)
    True

    Access the properties stored in DummyFeature:
    >>> dummy_feature.properties["value"]()
    42

    """

    def get(
        self: DummyFeature,
        image: Any,
        **kwargs: Any,
    ) -> Any:
        """Return the input image or list of images unchanged.

        This method simply returns the input without any transformation. 
        It adheres to the `Feature` interface by accepting additional keyword 
        arguments for consistency, although they are not used.

        Parameters
        ----------
        image: Any
            The input (typically an image or list of images) to pass through
            without modification.
        **kwargs: Any
            Additional properties sampled from `self.properties` or passed 
            externally. These are unused here but provided for consistency 
            with the `Feature` interface.

        Returns
        -------
        Any
            The same input that was passed in (typically an image or list of
            images).

        """

        return image


class Value(Feature):
    """Represent a constant (per evaluation) value in a DeepTrack pipeline.

    This feature holds a constant value (e.g., a scalar or array) and supplies 
    it on demand to other parts of the pipeline.
    
    Wen called with an image, it does not transform the input image but instead
    returns the stored value.

    Parameters
    ----------
    value: PropertyLike[float or array], optional
        The numerical value to store. It defaults to 0.
        If an `Image` is provided, a warning is issued recommending conversion
        to a NumPy array or a PyTorch tensor for performance reasons.
    **kwargs: Any
        Additional named properties passed to the `Feature` constructor.

    Attributes
    ----------
    __distributed__: bool
        Set to `False`, indicating that this feature’s `get(...)` method 
        processes the entire list of images (or data) at once, rather than 
        distributing calls for each item.

    Methods
    -------
    `get(image: Any, value: float, **kwargs: Any) -> float or array`
        Returns the stored value, ignoring the input image.

    Examples
    --------
    >>> import deeptrack as dt

    Initialize a constant value and retrieve it:
    >>> value = dt.Value(42)
    >>> value()
    42

    Override the value at call time:
    >>> value(value=100)
    100

    Initialize a constant array value and retrieve it:
    >>> import numpy as np
    >>>
    >>> arr_value = dt.Value(np.arange(4))
    >>> arr_value()
    array([0, 1, 2, 3])

    Override the array value at call time:
    >>> arr_value(value=np.array([10, 20, 30, 40]))
    array([10, 20, 30, 40])

    Initialize a constant PyTorch tensor value and retrieve it:
    >>> import torch
    >>>
    >>> tensor_value = dt.Value(torch.tensor([1., 2., 3.]))
    >>> tensor_value()
    tensor([1., 2., 3.])

    Override the tensor value at call time:
    >>> tensor_value(value=torch.tensor([10., 20., 30.]))
    tensor([10., 20., 30.])

    """

    __distributed__: bool = False  # Process as a single batch.

    def __init__(
        self: Value,
        value: PropertyLike[float | ArrayLike] = 0,
        **kwargs: Any,
    ):
        """Initialize the `Value` feature to store a constant value.

        This feature holds a constant numerical value and provides it to the 
        pipeline as needed.
        
        If an `Image` object is supplied, a warning is issued to encourage
        converting it to a NumPy array or a PyTorch tensor for performance
        optimization.

        Parameters
        ----------
        value: PropertyLike[float or array], optional
            The initial value to store. If an `Image` is provided, a warning is
            raised. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the `Feature` constructor, 
            such as custom properties or the feature name.

        """

        if isinstance(value, Image):
            import warnings

            warnings.warn(
                "Passing an Image object as the value to dt.Value may lead to "
                "performance deterioration. Consider converting the Image to "
                "a NumPy array with np.array(image), or to a PyTorch tensor "
                "with torch.tensor(np.array(image)).",
                DeprecationWarning,
            )

        super().__init__(value=value, **kwargs)

    def get(
        self: Value,
        image: Any,
        value: float | ArrayLike[Any],
        **kwargs: Any,
    ) -> float | ArrayLike[Any]:
        """Return the stored value, ignoring the input image.

        The `get` method simply returns the stored numerical value, allowing 
        for dynamic overrides when the feature is called.

        Parameters
        ----------
        image: Any
            Input data typically processed by features. For `Value`, this is 
            ignored and does not affect the output.
        value: float or array
            The current value to return. This may be the initial value or an 
            overridden value supplied during the method call.
        **kwargs: Any
            Additional keyword arguments, which are ignored but included for 
            consistency with the feature interface.

        Returns
        -------
        float or array
            The stored or overridden `value`, returned unchanged.

        """

        return value


class ArithmeticOperationFeature(Feature):
    """Apply an arithmetic operation element-wise to inputs.

    This feature performs an arithmetic operation (e.g., addition, subtraction,
    multiplication) on the input data. The inputs can be single values or lists
    of values.

    If a list is passed, the operation is applied to each element. 

    If both inputs are lists of different lengths, the shorter list is cycled.

    Parameters
    ----------
    op: Callable[[Any, Any], Any]
        The arithmetic operation to apply, such as a built-in operator 
        (`operator.add`, `operator.mul`) or a custom callable.
    value: float or int or list[float or int], optional
        The second operand for the operation. It defaults to 0. If a list is 
        provided, the operation will apply element-wise.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature`.

    Attributes
    ----------
    __distributed__: bool
        Indicates that this feature’s `get(...)` method processes the input as 
        a whole (`False`) rather than distributing calls for individual items.

    Methods
    -------
    `get(image: Any, value: float or int or list[float or int], **kwargs: Any) -> list[Any]`
        Apply the arithmetic operation element-wise to the input data.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import operator

    Define a simple addition operation:
    >>> addition = dt.ArithmeticOperationFeature(operator.add, value=10)

    Create a list of input values:
    >>> input_values = [1, 2, 3, 4]

    Apply the operation:
    >>> output_values = addition(input_values)
    >>> print(output_values)
    [11, 12, 13, 14]

    """

    __distributed__: bool = False

    def __init__(
        self: ArithmeticOperationFeature,
        op: Callable[[Any, Any], Any],
        value: PropertyLike[
            float
            | int
            | ArrayLike
            | list[float | int | ArrayLike]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the ArithmeticOperationFeature.

        Parameters
        ----------
        op: Callable[[Any, Any], Any]
            The arithmetic operation to apply, such as `operator.add`,
            `operator.mul`, or any custom callable that takes two arguments and
            returns a single output value.
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The second operand(s) for the operation. If a list is provided, the 
            operation is applied element-wise. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature`
            constructor.

        """

        super().__init__(value=value, **kwargs)

        self.op = op

    def get(
        self: ArithmeticOperationFeature,
        image: Any,
        value: float | int | ArrayLike | list[float | int | ArrayLike],
        **kwargs: Any,
    ) -> list[Any]:
        """Apply the operation element-wise to the input data.

        Parameters
        ----------
        image: Any or list[Any]
            The input data, either a single value or a list of values, to be 
            transformed by the arithmetic operation.
        value: float or int or array or list[float or int or array]
            The second operand(s) for the operation. If a single value is 
            provided, it is broadcast to match the input size. If a list is 
            provided, it will be cycled to match the length of the input list.
        **kwargs: Any
            Additional parameters or property overrides. These are generally 
            unused in this context but provided for compatibility with the 
            `Feature` interface.

        Returns
        -------
        list[Any]
            A list containing the results of applying the operation to the 
            input data element-wise.
            
        """

        # If value is a scalar, wrap it in a list for uniform processing.
        if not isinstance(value, (list, tuple)):
            value = [value]

        # Cycle the shorter list to match the length of the longer list.
        if len(image) < len(value):
            image = itertools.cycle(image)
        elif len(value) < len(image):
            value = itertools.cycle(value)

        # Apply the operation element-wise.
        return [self.op(a, b) for a, b in zip(image, value)]


class Add(ArithmeticOperationFeature):
    """Add a value to the input.
    
    This feature performs element-wise addition (+) to the input.

    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to add to the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Create a pipeline using `Add`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Add(value=5)
    >>> pipeline.resolve()
    [6, 7, 8]
    
    Alternatively, the pipeline can be created using operator overloading:
    >>> pipeline = dt.Value([1, 2, 3]) + 5
    >>> pipeline.resolve()
    [6, 7, 8]    
    
    Or:
    >>> pipeline = 5 + dt.Value([1, 2, 3])
    >>> pipeline.resolve()
    [6, 7, 8]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> sum_feature = dt.Add(value=5)
    >>> pipeline = sum_feature(input_value)
    >>> pipeline.resolve()
    [6, 7, 8]

    """

    def __init__(
        self: Add,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the Add feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to add to the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature`.

        """

        super().__init__(operator.add, value=value, **kwargs)


class Subtract(ArithmeticOperationFeature):
    """Subtract a value from the input.

    This feature performs element-wise subtraction (-) from the input.
    
    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to subtract from the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Create a pipeline using `Subtract`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Subtract(value=2)
    >>> pipeline.resolve()
    [-1, 0, 1]
    
    Alternatively, the pipeline can be created using operator overloading:
    >>> pipeline = dt.Value([1, 2, 3]) - 2
    >>> pipeline.resolve()
    [-1, 0, 1]
    
    Or:
    >>> pipeline = -2 + dt.Value([1, 2, 3])
    >>> pipeline.resolve()
    [-1, 0, 1]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> sub_feature = dt.Subtract(value=2)
    >>> pipeline = sub_feature(input_value)
    >>> pipeline.resolve()
    [-1, 0, 1]

    """

    def __init__(
        self: Subtract,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the Subtract feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to subtract from the input. it defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature`.
       
        """

        super().__init__(operator.sub, value=value, **kwargs)


class Multiply(ArithmeticOperationFeature):
    """Multiply the input by a value.

    This feature performs element-wise multiplication (*) of the input.
    
    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to multiply the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Multiply`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Multiply(value=5)
    >>> pipeline.resolve()
    [5, 10, 15]
    
    Alternatively, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) * 5
    >>> pipeline.resolve()
    [5, 10, 15]

    Or:
    >>> pipeline = 5 * dt.Value([1, 2, 3])
    >>> pipeline.resolve()
    [5, 10, 15]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> mul_feature = dt.Multiply(value=5)
    >>> pipeline = mul_feature(input_value)
    >>> pipeline.resolve()
    [5, 10, 15]

    """

    def __init__(
        self: Multiply,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the Multiply feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to multiply the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.mul, value=value, **kwargs)


class Divide(ArithmeticOperationFeature):
    """Divide the input with a value.

    This feature performs element-wise division (/) of the input.
    
    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to divide the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Divide`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Divide(value=5)
    >>> pipeline.resolve()
    [0.2 0.4 0.6]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) / 5
    >>> pipeline.resolve()
    [0.2 0.4 0.6]
    
    Which is not equivalent to:
    >>> pipeline = 5 / dt.Value([1, 2, 3])  # Different result
    >>> pipeline.resolve()
    [5.0, 2.5, 1.6666666666666667]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> truediv_feature = dt.Divide(value=5)
    >>> pipeline = truediv_feature(input_value)
    >>> pipeline.resolve()
    [0.2 0.4 0.6]

    """

    def __init__(
        self: Divide,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the Divide feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to divide the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.truediv, value=value, **kwargs)


class FloorDivide(ArithmeticOperationFeature):
    """Divide the input with a value.

    This feature performs element-wise floor division (//) of the input.
    
    Floor division produces an integer result when both operands are integers, 
    but truncates towards negative infinity when operands are floating-point 
    numbers.
    
    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to floor-divide the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `FloorDivide`:
    >>> pipeline = dt.Value([-3, 3, 6]) >> dt.FloorDivide(value=5)
    >>> pipeline.resolve()
    [-1, 0, 1]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([-3, 3, 6]) // 5
    >>> pipeline.resolve()
    [-1, 0, 1]
    
    Which is not equivalent to:
    >>> pipeline = 5 // dt.Value([-3, 3, 6])  # Different result
    >>> pipeline.resolve()
    [-2, 1, 0]
    
    Or, more explicitly:
    >>> input_value = dt.Value([-3, 3, 6])
    >>> floordiv_feature = dt.FloorDivide(value=5)
    >>> pipeline = floordiv_feature(input_value)
    >>> pipeline.resolve()
    [-1, 0, 1]

    """

    def __init__(
        self: FloorDivide,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the FloorDivide feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to fllor-divide the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.floordiv, value=value, **kwargs)


class Power(ArithmeticOperationFeature):
    """Raise the input to a power.

    This feature performs element-wise power (**) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to take the power of the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Power`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Power(value=3)
    >>> pipeline.resolve()
    [1, 8, 27]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) ** 3
    >>> pipeline.resolve()
    [1, 8, 27]
    
    Which is not equivalent to:
    >>> pipeline = 3 ** dt.Value([1, 2, 3])  # Different result
    >>> pipeline.resolve()
    [3, 9, 27]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> pow_feature = dt.Power(value=3)
    >>> pipeline = pow_feature(input_value)
    >>> pipeline.resolve()
    [1, 8, 27]

    """

    def __init__(
        self: Power,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the Power feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to take the power of the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.pow, value=value, **kwargs)


class LessThan(ArithmeticOperationFeature):
    """Determine whether input is less than value.

    This feature performs element-wise comparison (<) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to compare (<) with the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `LessThan`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.LessThan(value=2)
    >>> pipeline.resolve()
    [True, False, False]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) < 2
    >>> pipeline.resolve()
    [True, False, False]
    
    Which is not equivalent to:
    >>> pipeline = 2 < dt.Value([1, 2, 3])  # Different result
    >>> pipeline.resolve()
    [False, False, True]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> lt_feature = dt.LessThan(value=2)
    >>> pipeline = lt_feature(input_value)
    >>> pipeline.resolve()
    [True, False, False]

    """

    def __init__(
        self: LessThan,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the LessThan feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to compare (<) with the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.lt, value=value, **kwargs)


class LessThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is less than or equal to value.

    This feature performs element-wise comparison (<=) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to compare (<=) with the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `LessThanOrEquals`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.LessThanOrEquals(value=2)
    >>> pipeline.resolve()
    [True, True, False]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) <= 2
    >>> pipeline.resolve()
    [True, True, False]
    
    Which is not equivalent to:
    >>> pipeline = 2 <= dt.Value([1, 2, 3])  # Different result
    >>> pipeline.resolve()
    [False, True, True]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> le_feature = dt.LessThanOrEquals(value=2)
    >>> pipeline = le_feature(input_value)
    >>> pipeline.resolve()
    [True, True, False]

    """

    def __init__(
        self: LessThanOrEquals,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the LessThanOrEquals feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to compare (<=) with the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.le, value=value, **kwargs)


LessThanOrEqual = LessThanOrEquals


class GreaterThan(ArithmeticOperationFeature):
    """Determine whether input is greater than value.

    This feature performs element-wise comparison (>) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to compare (>) with the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `GreaterThan`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.GreaterThan(value=2)
    >>> pipeline.resolve()
    [False, False, True]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) > 2
    >>> pipeline.resolve()
    [False, False, True]

    Which is not equivalent to:
    >>> pipeline = 2 > dt.Value([1, 2, 3])  # Different result
    >>> pipeline.resolve()
    [True, False, False]
    
    Or, most explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> gt_feature = dt.GreaterThan(value=2)
    >>> pipeline = gt_feature(input_value)
    >>> pipeline.resolve()
    [False, False, True]

    """

    def __init__(
        self: GreaterThan,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the GreaterThan feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to compare (>) with the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.gt, value=value, **kwargs)


class GreaterThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is greater than or equal to value.

    This feature performs element-wise comparison (>=) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to compare (<=) with the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `GreaterThanOrEquals`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.GreaterThanOrEquals(value=2)
    >>> pipeline.resolve()
    [False, True, True]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) >= 2
    >>> pipeline.resolve()
    [False, True, True]

    Which is not equivalent to:
    >>> pipeline = 2 >= dt.Value([1, 2, 3])  # Different result
    >>> pipeline.resolve()
    [True, True, False]
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> ge_feature = dt.GreaterThanOrEquals(value=2)
    >>> pipeline = ge_feature(input_value)
    >>> pipeline.resolve()
    [False, True, True]

    """

    def __init__(
        self: GreaterThanOrEquals,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the GreaterThanOrEquals feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to compare (>=) with the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.ge, value=value, **kwargs)


GreaterThanOrEqual = GreaterThanOrEquals


class Equals(ArithmeticOperationFeature):
    """Determine whether input is equal to a given value.

    This feature performs element-wise comparison between the input and a
    specified value.

    Notes
    -----
    - Unlike other arithmetic operators, `Equals` does not define `__eq__` 
      (`==`) and `__req__` (`==`) in `DeepTrackNode` and `Feature`, as this 
      would affect Python’s built-in identity comparison.
    - This means that the standard `==` operator is overloaded only for 
      expressions involving `Feature` instances but not for comparisons 
      involving regular Python objects.
    - Always use `>>` to apply `Equals` correctly in a feature chain.

    Parameters
    ----------
    value: PropertyLike[int or float or array or list[int or floar or array]], optional
        The value to compare (==) with the input. It defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.
    
    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Equals`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Equals(value=2)
    >>> pipeline.resolve()
    [False, True, False]
    
    Or:
    >>> input_values = [1, 2, 3]
    >>> eq_feature = dt.Equals(value=2)
    >>> output_values = eq_feature(input_values)
    >>> print(output_values)
    [False, True, False]    
    
    These are the **only correct ways** to apply `Equals` in a pipeline.
    
    The following approaches are **incorrect**:
    
    Using `==` directly on a `Feature` instance **does not work** because 
    `Feature` does not override `__eq__`:
    >>> pipeline = dt.Value([1, 2, 3]) == 2  # Incorrect
    >>> pipeline.resolve()  
    AttributeError: 'bool' object has no attribute 'resolve'

    Similarly, directly calling `Equals` on an input feature **immediately 
    evaluates the comparison**, returning a boolean instead of a `Feature`:
    >>> pipeline = dt.Equals(value=2)(dt.Value([1, 2, 3]))  # Incorrect
    >>> pipeline.resolve()
    AttributeError: 'bool' object has no attribute 'resolve'

    """

    def __init__(
        self: Equals,
        value: PropertyLike[
            float
            | int
            | ArrayLike[Any]
            | list[float | int | ArrayLike[Any]]
        ] = 0,
        **kwargs: Any,
    ):
        """Initialize the Equals feature.

        Parameters
        ----------
        value: PropertyLike[float or int or array or list[float or int or array]], optional
            The value to compare with the input. It defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.eq, value=value, **kwargs)


Equal = Equals


class Stack(Feature):
    """Stack the input and the value.
    
    This feature combines the output of the input data (`image`) and the 
    value produced by the specified feature (`value`). The resulting output 
    is a list where the elements of the `image` and `value` are concatenated.

    If either the input (`image`) or the `value` is a single `Image` object, 
    it is automatically converted into a list to maintain consistency in the 
    output format.

    If B is a feature, `Stack` can be visualized as:

    >>>   A >> Stack(B) = [*A(), *B()]

    Parameters
    ----------
    value: PropertyLike[Any]
        The feature or data to stack with the input.
    **kwargs: Any
        Additional arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs. 
        Always `False` for `Stack`, as it processes all inputs at once.

    Methods
    -------
    `get(image: Any, value: Any, **kwargs: Any) -> list[Any]`
        Concatenate the input with the value.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Stack`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Stack(value=[4, 5])
    >>> pipeline.resolve()
    [1, 2, 3, 4, 5]

    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) & [4, 5]
    >>> pipeline.resolve()
    [1, 2, 3, 4, 5]

    Or:
    >>> pipeline = [4, 5] & dt.Value([1, 2, 3])  # Different result
    >>> pipeline.resolve()
    [4, 5, 1, 2, 3]

    Note
    ----
    If a feature is called directly, its result is cached internally. This can
    affect how it behaves when reused in chained pipelines. For exmaple:
    >>> stack_feature = dt.Stack(value=2)
    >>> _ = stack_feature(1)  # Evaluate the feature and cache the output
    >>> (1 & stack_feature)()
    [1, 1, 2]

    To ensure consistent behavior when reusing a feature after calling it,
    reset its state using instead:
    >>> stack_feature = dt.Stack(value=2)
    >>> _ = stack_feature(1)
    >>> stack_feature.update()  # clear cached state
    >>> (1 & stack_feature)()
    [1, 2]

    """

    __distributed__: bool = False

    def __init__(
        self: Stack,
        value: PropertyLike[Any],
        **kwargs: Any,
    ):
        """Initialize the Stack feature.

        Parameters
        ----------
        value: PropertyLike[Any]
            The feature or data to stack with the input.
        **kwargs: Any
            Additional arguments passed to the parent `Feature` class.
        
        """

        super().__init__(value=value, **kwargs)

    def get(
        self: Stack,
        image: Any | list[Any],
        value: Any | list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Concatenate the input with the value.

        It ensures that both the input (`image`) and the value (`value`) are 
        treated as lists before concatenation.

        Parameters
        ----------
        image: Any or list[Any]
            The input data to stack. Can be a single element or a list.
        value: Any or list[Any]
            The feature or data to stack with the input. Can be a single 
            element or a list.
        **kwargs: Any
            Additional keyword arguments (not used here).

        Returns
        -------
        list[Any]
            A list containing all elements from `image` and `value`.

        """

        # Ensure the input is treated as a list.
        if not isinstance(image, list):
            image = [image]

        # Ensure the value is treated as a list.
        if not isinstance(value, list):
            value = [value]

        # Concatenate and return the lists.
        return [*image, *value]


class Arguments(Feature):
    """A convenience container for pipeline arguments.

    The `Arguments` feature allows dynamic control of pipeline behavior by
    providing a container for arguments that can be modified or overridden at
    runtime. This is particularly useful when working with parametrized
    pipelines, such as toggling behaviors based on whether an image is a label
    or a raw input.

    Methods
    -------
    `get(image: Any, **kwargs: Any) -> Any`
        It passes the input image through unchanged, while allowing for
        property overrides.

    Examples
    --------
    >>> import deeptrack as dt

    Create a temporary image file:
    >>> import numpy as np
    >>> import PIL, tempfile
    >>>
    >>> test_image_array = (np.ones((50, 50)) * 128).astype(np.uint8)
    >>> temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    >>> PIL.Image.fromarray(test_image_array).save(temp_png.name)

    A typical use-case is:
    >>> arguments = dt.Arguments(is_label=False)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name)
    ...     >> dt.Gaussian(sigma=arguments.is_label)  # Image with no noise
    ... )
    >>> image_pipeline.bind_arguments(arguments)
    >>>
    >>> image = image_pipeline()
    >>> image.std()
    0.0

    Change the argument:
    >>> image = image_pipeline(is_label=True)  # Image with added noise
    >>> image.std()
    1.0104364326447652

    Remove the temporary image:
    >>> import os
    >>>
    >>> os.remove(temp_png.name)

    For a non-mathematical dependence, create a local link to the property as 
    follows:
    >>> arguments = dt.Arguments(is_label=False)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name)
    ...     >> dt.Gaussian(
    ...         local_is_label=arguments.is_label,
    ...         sigma=lambda local_is_label: 1 if local_is_label else 0,
    ...     )
    ... )
    >>> image_pipeline.bind_arguments(arguments)

    Keep in mind that, if any dependent property is non-deterministic, it may 
    permanently change:
    >>> arguments = dt.Arguments(noise_max=1)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name)
    ...     >> dt.Gaussian(
    ...         noise_max=arguments.noise_max,
    ...         sigma=lambda noise_max: np.random.rand() * noise_max,
    ...     )
    ... )
    >>> image_pipeline.bind_arguments(arguments)
    >>> image_pipeline.store_properties()  # Store image properties
    >>>
    >>> image = image_pipeline()
    >>> image.std(), image.get_property("sigma")
    (0.8464173007136401, 0.8423390304699889)

    >>> image = image_pipeline(noise_max=0)
    >>> image.std(), image.get_property("sigma")
    (0.0, 0.0)

    As with any feature, all arguments can be passed by deconstructing the 
    properties dict:
    >>> arguments = dt.Arguments(is_label=False, noise_sigma=5)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name)
    ...     >> dt.Gaussian(
    ...         sigma=lambda is_label, noise_sigma: (
    ...             0 if is_label else noise_sigma
    ...         ),
    ...         **arguments.properties,
    ...     )
    ... )
    >>> image_pipeline.bind_arguments(arguments)
    >>>
    >>> image = image_pipeline()  # Image with added noise
    >>> image.std()
    5.002151761964336

    >>> image = image_pipeline(is_label=True)  # Raw image with no noise
    >>> image.std()
    0.0

    """

    def get(
        self: Arguments,
        image: Any,
        **kwargs: Any,
    ) -> Any:

        """Return the input image and allow property overrides.

        This method does not modify the input image but provides a mechanism
        for overriding arguments dynamically during pipeline execution.

        Parameters
        ----------
        image: Any
            The input image to be passed through unchanged.
        **kwargs: Any
            Key-value pairs for overriding pipeline properties.

        Returns
        -------
        Any
            The unchanged input image.

        """

        return image


class Probability(StructuralFeature):
    """Resolve a feature with a certain probability.

    This feature conditionally applies a given feature to an input based on a
    sampled uniform random number. If the sampled number is less than the
    specified probability, the feature is resolved; otherwise, the input is
    returned unchanged.

    To resample the decision, call `.update()` before evaluating the feature.

    Parameters
    ----------
    feature: Feature
        The feature to resolve conditionally.
    probability: PropertyLike[float]
        The probability (between 0 and 1) of resolving the feature.
    *args: Any
        Positional arguments passed to the parent `StructuralFeature` class.
    **kwargs: Any
        Additional keyword arguments passed to the parent `StructuralFeature` 
        class.

    Methods
    -------
    `get(image: Any, probability: float, random_number: float, **kwargs: Any) -> Any`
        Resolves the feature if the sampled random number is less than the 
        specified probability.

    Examples
    --------
    >>> import deeptrack as dt
    
    In this example, the `Add` feature is applied to the input image with a 70%
    chance.

    Define a feature and wrap it with `Probability`:
    >>> add_feature = dt.Add(value=2)
    >>> probabilistic_feature = dt.Probability(add_feature, probability=0.7)

    Define an input image:
    >>> import numpy as np
    >>>
    >>> input_image = np.zeros((2, 3))

    Apply the feature:
    >>> probabilistic_feature.update()  # Update the random number
    >>> output_image = probabilistic_feature(input_image)

    With 70% probability, the output is:
    >>> output_image
    array([[2., 2., 2.],
        [2., 2., 2.]])

    With 30% probability, it remains:
    >>> output_image
    array([[0., 0., 0.],
        [0., 0., 0.]])

    """

    def __init__(
        self: Probability,
        feature: Feature,
        probability: PropertyLike[float],
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Probability feature.

        The random number is initialized when this feature is initialized.
        It can be updated using the `update()` method.

        Parameters
        ----------
        feature: Feature
            The feature to resolve conditionally.
        probability: PropertyLike[float]
            The probability (between 0 and 1) of resolving the feature.
        *args: Any
            Positional arguments passed to the parent `StructuralFeature`
            class.
        **kwargs: Any
            Additional keyword arguments passed to the parent
            `StructuralFeature` class.

        """

        super().__init__(
            *args,
            probability=probability,
            random_number=np.random.rand,
            **kwargs,
        )
        self.feature = self.add_feature(feature)

    def get(
        self: Probability,
        image: Any,
        probability: float,
        random_number: float,
        **kwargs: Any,
    ) -> Any:
        """Resolve the feature if random number is less than probability.

        Parameters
        ----------
        image: Any or list[Any]
            The input to process.
        probability: float
            The probability (between 0 and 1) of resolving the feature.
        random_number: float
            A random number sampled to determine whether to resolve the
            feature. It is initialized when this feature is initialized.
            It can be updated using the `update()` method.
        **kwargs: Any
            Additional arguments passed to the feature's `resolve()` method.

        Returns
        -------
        Any
            The processed image. If the feature is resolved, this is the output
            of the feature; otherwise, it is the unchanged input image.

        """

        if random_number < probability:
            image = self.feature.resolve(image, **kwargs)

        return image


class Repeat(StructuralFeature):
    """Apply a feature multiple times in sequence.

    The `Repeat` feature iteratively applies another feature, passing the 
    output of each iteration as the input to the next. This enables chained 
    transformations, where each iteration builds upon the previous one. The 
    number of repetitions is defined by `N`.

    Each iteration operates with its own set of properties, and the index of 
    the current iteration is accessible via `_ID`. `_ID` is extended to include
    the current iteration index, ensuring deterministic behavior when needed.

    This is equivalent to using the `^` operator:

    >>> dt.Repeat(A, 3) ≡ A ^ 3

    Parameters
    ----------
    feature: Feature
        The feature to be repeated.
    N: int
        The number of times to apply the feature in sequence.
    **kwargs: Any

    Methods
    -------
    `get(image: Any, N: int, _ID: tuple[int, ...], **kwargs: Any) -> Any`
        It applies the feature `N` times in sequence, passing the output of
        each iteration as the input to the next.

    Examples
    --------
    >>> import deeptrack as dt
    
    Define an `Add` feature that adds `10` to its input:
    >>> add_ten = dt.Add(value=10)

    Apply this feature 3 times using `Repeat`:
    >>> pipeline = dt.Repeat(add_ten, N=3)

    Process an input list:
    >>> pipeline.resolve([1, 2, 3])
    [31, 32, 33]

    Alternative shorthand using `^` operator:
    >>> pipeline = dt.Add(value=10) ^ 3
    >>> pipeline.resolve([1, 2, 3])
    [31, 32, 33]
    
    """

    def __init__(
        self: Repeat,
        feature: Feature,
        N: int,
        **kwargs: Any,
    ):
        """Initialize the Repeat feature.

        This feature applies `feature` iteratively, passing the output of each 
        iteration as the input to the next. The number of repetitions is 
        controlled by `N`, and each iteration has its own dynamically updated 
        properties.

        Parameters
        ----------
        feature: Feature
            The feature to be applied sequentially `N` times.
        N: int
            The number of times to sequentially apply `feature`, passing the 
            output of each iteration as the input to the next.
        **kwargs: Any
            Keyword arguments that override properties dynamically at each 
            iteration and are also passed to the parent `Feature` class.

        """

        super().__init__(N = N, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Repeat,
        image: Any,
        N: int,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Sequentially apply the feature N times.

        This method applies the feature `N` times, passing the output of each 
        iteration as the input to the next. The `_ID` tuple is updated at 
        each iteration, ensuring dynamic property updates and reproducibility.
  
        Each iteration uses the output of the previous one. This makes `Repeat`
        suitable for building recursive, cumulative, or progressive
        transformations.
  
        Parameters
        ----------
        image: Any
            The input data to be transformed by the repeated feature.
        N: int
            The number of times to sequentially apply the feature, where each 
            iteration builds on the previous output.
        _ID: tuple[int, ...], optional
            A unique identifier for tracking the iteration index, ensuring 
            reproducibility, caching, and dynamic property updates.
        **kwargs: Any
            Additional keyword arguments passed to the feature.

        Returns
        -------
        Any
            The output of the final iteration after `N` sequential applications 
            of the feature.

        """

        if not isinstance(N, int) or N < 0:
            raise ValueError("N must be a non-negative integer.")

        for n in range(N):

            index = _ID + (n,)  # Track iteration index

            image = self.feature(
                image,
                _ID=index,
                replicate_index=index,  # Legacy property
            )

        return image


class Combine(StructuralFeature):
    """Combine multiple features into a single feature.

    This feature applies a list of features to the same input and returns their
    outputs as a list. It is useful for computing multiple parallel outputs
    from the same data (e.g., branches in a feature graph).

    Parameters
    ----------
    features: list[Feature]
        A list of features to combine. Each feature will be applied in order,
        and their outputs collected into a list.
    **kwargs: Any
        Additional keyword arguments passed to the parent `StructuralFeature` 
        class.

    Methods
    -------
    `get(image: Any, **kwargs: Any) -> list[Any]`
        Resolves each feature in the `features` list on the input image and 
        returns their results as a list.

    Examples
    --------
    >>> import deeptrack as dt

    Define a list of features:
    >>> add_1 = dt.Add(value=1)
    >>> add_2 = dt.Add(value=2)
    >>> add_3 = dt.Add(value=3)

    Combine the features:
    >>> combined_feature = dt.Combine([add_1, add_2, add_3])

    Define an input image:
    >>> import numpy as np
    >>>
    >>> input_image = np.zeros((2, 3))

    Apply the combined feature:
    >>> output_list = combined_feature(input_image)
    >>> output_list
    [array([[1., 1., 1.],
            [1., 1., 1.]]),
    array([[2., 2., 2.],
            [2., 2., 2.]]),
    array([[3., 3., 3.],
            [3., 3., 3.]])]

    """

    def __init__(
        self: Combine,
        features: list[Feature],
        **kwargs: Any,
    ):
        """Initialize the Combine feature.

        Parameters
        ----------
        features: list[Feature]
            A list of features to combine. Each feature is added as a 
            dependency to ensure proper execution in the computation graph.
        **kwargs: Any
            Additional keyword arguments passed to the parent 
            `StructuralFeature` class.

        """

        super().__init__(**kwargs)

        self.features = [self.add_feature(f) for f in features]

    def get(
        self: Combine,
        image: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Resolve each feature in the `features` list on the input image.

        Parameters
        ----------
        image: Any
            The input image or list of images to process.
        **kwargs: Any
            Additional arguments passed to each feature's `resolve` method.

        Returns
        -------
        list[Any]
            A list containing the outputs of each feature applied to the input.

        """

        return [f(image, **kwargs) for f in self.features]


class Slice(Feature):
    """Dynamically apply array indexing to inputs.

    This feature allows dynamic slicing of an image using integer indices, 
    slice objects, or ellipses (`...`).

    While normal array indexing is preferred for static cases, `Slice` is
    useful when the slicing parameters must be computed dynamically based on
    other properties.

    Parameters
    ----------
    slices: tuple[int or slice or ellipsis] or list[int or slice or ellipsis]
        The slicing instructions for each dimension. Each element corresponds 
        to a dimension in the input image.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: array or list[array], slices: Iterable[int or slice or ellipsis], **kwargs: Any) -> array or list[array]`
        Applies the specified slices to the input image.

    Examples
    --------
    >>> import deeptrack as dt

    Recommended approach: Use normal indexing for static slicing:
    >>> import numpy as np
    >>>
    >>> feature = dt.DummyFeature()
    >>> static_slicing = feature[0:2, ::2, :]
    >>> result = static_slicing.resolve(np.arange(27).reshape((3, 3, 3)))
    >>> result
    array([[[ 0,  1,  2],
            [ 6,  7,  8]],
           [[ 9, 10, 11],
            [15, 16, 17]]])

    Using `Slice` for dynamic slicing (when necessary when slices depend on
    computed properties):
    >>> feature = dt.DummyFeature()
    >>> dynamic_slicing = feature >> dt.Slice(
    ...     slices=(slice(0, 2), slice(None, None, 2), slice(None))
    ... )
    >>> result = dynamic_slicing.resolve(np.arange(27).reshape((3, 3, 3)))
    >>> result
    array([[[ 0,  1,  2],
            [ 6,  7,  8]],
           [[ 9, 10, 11],
            [15, 16, 17]]])

    In both cases, slices can be defined dynamically based on feature 
    properties.

    """

    def __init__(
        self: Slice,
        slices: PropertyLike[Iterable[int | slice | Ellipsis]],
        **kwargs: Any,
    ):
        """Initialize the Slice feature.

        Parameters
        ----------
        slices: Iterable[int or slice or ellipsis]
            The slicing instructions for each dimension, specified as a 
            list or tuple of integers, slice objects, or ellipses (`...`).
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(slices=slices, **kwargs)

    def get(
        self: Slice,
        image: ArrayLike[Any] | list[ArrayLike[Any]],
        slices: slice | tuple[int | slice | Ellipsis, ...],
        **kwargs: Any,
    ) -> ArrayLike[Any] | list[ArrayLike[Any]]:
        """Apply the specified slices to the input image.

        Parameters
        ----------
        image: array or list[array]
            The input image(s) to be sliced.
        slices: slice ellipsis or tuple[int or slice or ellipsis, ...]
            The slicing instructions for the input image. Typically it is a
            tuple. Each element in the tuple corresponds to a dimension in the
            input image. If a single element is provided, it is converted to a
            tuple.
        **kwargs: Any
            Additional keyword arguments (unused in this implementation).

        Returns
        -------
        array or list[array]
            The sliced image(s).

        """

        try:
            # Convert slices to a tuple if possible
            slices = tuple(slices)
        except ValueError:
            # Leave slices as is if conversion fails
            pass

        return image[slices]


class Bind(StructuralFeature):
    """Bind a feature with property arguments.

    When the feature is resolved, the kwarg arguments are passed to the child 
    feature. Thus, this feature allows passing additional keyword arguments 
    (`kwargs`) to a child feature when it is resolved. These properties can 
    dynamically control the behavior of the child feature.

    Parameters
    ----------
    feature: Feature
        The child feature
    **kwargs: Any
        Properties to send to child

    Methods
    -------
    `get(image: Any, **kwargs: Any) -> Any`
        It resolves the child feature with the provided arguments.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a `Gaussian` feature: 
    >>> gaussian_noise = dt.Gaussian()

    Create a test image:
    >>> import numpy as np
    >>>
    >>> input_image = np.zeros((512, 512))

    Bind fixed values to the parameters:
    >>> bound_feature = dt.Bind(gaussian_noise, mu=-5, sigma=2)

    Resolve the bound feature:
    >>> output_image = bound_feature.resolve(input_image)
    >>> round(np.mean(output_image), 1), round(np.std(output_image), 1)
    (-5.0, 2.0)

    """

    def __init__(
        self: Bind,
        feature: Feature,
        **kwargs: Any,
    ):
        """Initialize the Bind feature.

        Parameters
        ----------
        feature: Feature
            The child feature to bind.
        **kwargs: Any
            Properties or arguments to pass to the child feature.

        """

        super().__init__(**kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self: Bind,
        image: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve the child feature with the dynamically provided arguments.

        Parameters
        ----------
        image: Any
            The input data or image to process.
        **kwargs: Any
            Properties or arguments to pass to the child feature during
            resolution.

        Returns
        -------
        Any
            The result of resolving the child feature with the provided
            arguments.

        """

        return self.feature.resolve(image, **kwargs)


BindResolve = Bind


class BindUpdate(StructuralFeature):  # DEPRECATED
    """Bind a feature with certain arguments.

    .. deprecated:: 2.0
        This feature is deprecated and may be removed in a future release. It
        is recommended to use `Bind` instead for equivalent functionality.
        Further, the current implementation is not guaranteed to be exactly
        equivalent to prior implementations.

    This feature binds a child feature with specific properties (`kwargs`) that 
    are passed to it when it is updated. It is similar to the `Bind` feature 
    but is marked as deprecated in favor of `Bind`.

    Parameters
    ----------
    feature: Feature
        The child feature to bind with specific arguments.
    **kwargs: Any
        Properties to send to the child feature during updates.

    Methods
    -------
    `get(image: Any, **kwargs: Any) -> Any`
        It resolves the child feature with the provided arguments.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a `Gaussian` feature:
    >>> gaussian_noise = dt.Gaussian()

    Dynamically modify the behavior of the feature using `BindUpdate`:
    >>> bound_feature = dt.BindUpdate(gaussian_noise, mu = 5, sigma=3)
    
    >>> import numpy as np
    >>>
    >>> input_image = np.zeros((512, 512))
    >>> output_image = bound_feature.resolve(input_image)
    >>> round(np.mean(output_image), 1), round(np.std(output_image), 1)
    (5.0, 3.0)

    """

    def __init__(
        self: Feature, 
        feature: Feature, 
        **kwargs: Any,
    ):
        """Initialize the BindUpdate feature.

        Parameters
        ----------
        feature: Feature
            The child feature to bind with specific arguments.
        **kwargs: Any
            Properties to send to the child feature during updates.

        Warnings
        --------
        It emits a deprecation warning, encouraging the use of `Bind` instead.

        """

        import warnings

        warnings.warn(
            "BindUpdate is deprecated and may be removed in a future release. "
            "The current implementation is not guaranteed to be exactly "
            "equivalent to prior implementations. "
            "Please use Bind instead.",
            DeprecationWarning,
        )

        super().__init__(**kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self: Feature,
        image: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve the child feature with the provided arguments.

        Parameters
        ----------
        image: Any
            The input data or image to process.
        **kwargs: Any
            Properties or arguments to pass to the child feature during 
            resolution.

        Returns
        -------
        Any
            The result of resolving the child feature with the provided 
            arguments.

        """

        return self.feature.resolve(image, **kwargs)


class ConditionalSetProperty(StructuralFeature):  # DEPRECATED
    """Conditionally override the properties of a child feature.

    .. deprecated:: 2.0
        This feature is deprecated and may be removed in a future release. It
        is recommended to use `Arguments` instead.

    This feature modifies the properties of a child feature only when a 
    specified condition is met. If the condition evaluates to `True`, 
    the given properties are applied; otherwise, the child feature remains 
    unchanged.

    It is advisable to use `Arguments` instead when possible, since this
    feature overwrites properties, which may affect future calls to the
    feature.

    If `condition` is a string, the condition must be explicitly passed when
    resolving.

    The properties applied do not persist unless explicitly stored.

    Parameters
    ----------
    feature: Feature
        The child feature whose properties will be modified conditionally.
    condition: PropertyLike[str or bool] or None
        Either a boolean value (`True`, `False`) or the name of a boolean 
        property in the feature’s property dictionary. If the condition 
        evaluates to `True`, the specified properties are applied.
    **kwargs: Any
        The properties to be applied to the child feature if `condition` is 
        `True`.

    Methods
    -------
    `get(image: Any, condition: str or bool, **kwargs: Any) -> Any`
        Resolves the child feature, conditionally applying the specified 
        properties.

    Examples
    --------
    >>> import deeptrack as dt
    
    Define an image:
    >>> import numpy as np
    >>>
    >>> image = np.ones((512, 512))

    Define a `Gaussian` noise feature:
    >>> gaussian_noise = dt.Gaussian(sigma=0)

    --- Using a boolean condition ---
    Apply `sigma=5` only if `condition=True`:
    >>> conditional_feature = dt.ConditionalSetProperty(
    ...     gaussian_noise, sigma=5,
    ... )

    Resolve with condition met:
    >>> noisy_image = conditional_feature(image, condition=True)
    >>> round(noisy_image.std(), 1)
    5.0

    Resolve without condition:
    >>> conditional_feature.update()  # Essential to reset the property
    >>> clean_image = conditional_feature(image, condition=False)
    >>> round(clean_image.std(), 1)
    0.0

    --- Using a string-based condition ---
    Define condition as a string:
    >>> conditional_feature = dt.ConditionalSetProperty(
    ...     gaussian_noise, sigma=5, condition="is_noisy"
    ... )

    Resolve with condition met:
    >>> noisy_image = conditional_feature(image, is_noisy=True)
    >>> round(noisy_image.std(), 1)
    5.0

    Resolve without condition:
    >>> conditional_feature.update()
    >>> clean_image = conditional_feature(image, is_noisy=False)
    >>> round(clean_image.std(), 1)
    0.0

    """

    def __init__(
        self: ConditionalSetProperty,
        feature: Feature,
        condition: PropertyLike[str | bool] | None = None,
        **kwargs: Any,
    ):
        """Initialize the ConditionalSetProperty feature.

        Parameters
        ----------
        feature: Feature
            The child feature to conditionally modify.
        condition: PropertyLike[str or bool] or None
            A boolean value or the name of a boolean property in the feature's 
            property dictionary. If the condition evaluates to `True`, the 
            specified properties are applied.
        **kwargs: Any
            Properties to apply to the child feature if the condition is 
            `True`.

        """

        import warnings

        warnings.warn(
            "ConditionalSetFeature is deprecated and may be removed in a "
            "future release. Please use Arguments instead when possible.",
            DeprecationWarning,
        )

        if isinstance(condition, str):
            kwargs.setdefault(condition, True)

        super().__init__(condition=condition, **kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self: ConditionalSetProperty,
        image: Any,
        condition: str | bool,
        **kwargs: Any,
    ) -> Any:
        """Resolve the child, conditionally applying specified properties.

        Parameters
        ----------
        image: Any
            The input data or image to process.
        condition: str or  bool
            A boolean value or the name of a boolean property in the feature's 
            property dictionary. If the condition evaluates to `True`, the 
            specified properties are applied.
        **kwargs:: Any
            Additional properties to apply to the child feature if the 
            condition is `True`.

        Returns
        -------
        Any
            The resolved child feature, with properties conditionally modified.

        """

        # Determine the condition value.
        _condition = condition
        if isinstance(condition, str):
            _condition = kwargs.get(condition, False)

        # Apply properties to the child feature if the condition is True.
        if _condition:
            propagate_data_to_dependencies(self.feature, **kwargs)

        return self.feature(image)


class ConditionalSetFeature(StructuralFeature):  # DEPRECATED
    """Conditionally resolve one of two features.

    .. deprecated:: 2.0
        This feature is deprecated and may be removed in a future release. It
        is recommended to use `Arguments` instead.

    This feature allows dynamically selecting and resolving one of two child 
    features depending on whether a specified condition evaluates to `True` or 
    `False`.
    
    The `condition` parameter specifies either:
    - A boolean value (default is `True`).
    - The name of a property to listen to. For example, if
    `condition="is_label"`, the selected feature can be toggled as follows:
    
    >>> feature.resolve(is_label=True)   # Resolves `on_true`
    >>> feature.resolve(is_label=False)  # Resolves `on_false`
    >>> feature.update(is_label=True)    # Updates both features

    Both `on_true` and `on_false` are updated during each call, even if only 
    one is resolved.

    It is advisable to use `Arguments` instead when possible.

    Parameters
    ----------
    on_false: Feature, optional
        The feature to resolve if the condition is `False`. If not provided, 
        the input image remains unchanged.
    on_true: Feature, optional
        The feature to resolve if the condition is `True`. If not provided, 
        the input image remains unchanged.
    condition: str or bool, optional
        The name of the conditional property or a boolean value. If a string 
        is provided, its value is retrieved from `kwargs` or `self.properties`. 
        If not found, the default value is `True`.
    **kwargs: Any
        Additional keyword arguments passed to the parent `StructuralFeature`.

    Methods
    -------
    `get(image: Any, condition: str or bool, **kwargs: Any) -> Any`
        Resolves the appropriate feature based on the condition.

    Examples
    --------
    >>> import deeptrack as dt

    Define an image:
    >>> import numpy as np
    >>>
    >>> image = np.ones((512, 512))

    Define two `Gaussian` noise features:
    >>> true_feature = dt.Gaussian(sigma=0)
    >>> false_feature = dt.Gaussian(sigma=5)
    
    --- Using a boolean condition ---
    Combine the features into a conditional set feature. 
    If not provided explicitely, the condition is assumed to be True:
    >>> conditional_feature = dt.ConditionalSetFeature(
    ...     on_true=true_feature,
    ...     on_false=false_feature,
    ... )

    Resolve based on the condition. If not specified, default is True:
    >>> clean_image = conditional_feature(image)
    >>> round(clean_image.std(), 1)
    0.0
    
    >>> noisy_image = conditional_feature(image, condition=False)
    >>> round(noisy_image.std(), 1)
    5.0

    >>> clean_image = conditional_feature(image, condition=True)
    >>> round(clean_image.std(), 1)
    0.0

    --- Using a string-based condition ---
    Define condition as a string:
    >>> conditional_feature = dt.ConditionalSetFeature(
    ...     on_true=true_feature, 
    ...     on_false=false_feature, 
    ...     condition = "is_noisy",
    ... )

    Resolve based on the conditions:
    >>> noisy_image = conditional_feature(image, is_noisy=False)
    >>> round(noisy_image.std(), 1)
    5.0

    >>> clean_image = conditional_feature(image, is_noisy=True)
    >>> round(clean_image.std(), 1)
    0.0

    """

    def __init__(
        self: ConditionalSetFeature,
        on_false: Feature | None = None,
        on_true: Feature | None = None,
        condition: PropertyLike[str | bool] = True,
        **kwargs: Any,
    ):
        """Initialize the ConditionalSetFeature.

        Parameters
        ----------
        on_false: Feature, optional
            The feature to resolve if the condition evaluates to `False`.
        on_true: Feature, optional
            The feature to resolve if the condition evaluates to `True`.
        condition: str or bool, optional
            The name of the property to listen to, or a boolean value. It
            defaults to `True`.
        **kwargs:: Any
            Additional keyword arguments for the parent `StructuralFeature`.

        """

        import warnings

        warnings.warn(
            "ConditionalSetFeature is deprecated and may be removed in a "
            "future release. Please use Arguments instead when possible.",
            DeprecationWarning,
        )

        if isinstance(condition, str):
            kwargs.setdefault(condition, True)

        super().__init__(condition=condition, **kwargs)

        # Add the child features to the dependency graph if provided.
        if on_true:
            self.add_feature(on_true)
        if on_false:
            self.add_feature(on_false)

        self.on_true = on_true
        self.on_false = on_false

    def get(
        self: ConditionalSetFeature,
        image: Any,
        *,
        condition: str | bool,
        **kwargs: Any,
    ):
        """Resolve the appropriate feature based on the condition.

        Parameters
        ----------
        image: Any
            The input image to process.
        condition: str or bool
            The name of the conditional property or a boolean value. If a 
            string is provided, it is looked up in `kwargs` to get the actual 
            boolean value.
        **kwargs:: Any
            Additional keyword arguments to pass to the resolved feature.

        Returns
        -------
        Any
            The processed image after resolving the appropriate feature. If 
            neither `on_true` nor `on_false` is provided for the corresponding 
            condition, the input image is returned unchanged.

        """

        # Evaluate the condition.
        _condition = condition
        if isinstance(condition, str):
            _condition = kwargs.get(condition, False)

        # Resolve the appropriate feature.
        if _condition and self.on_true:
            return self.on_true(image)
        if not _condition and self.on_false:
            return self.on_false(image)
        return image


class Lambda(Feature):
    """Apply a user-defined function to the input.

    This feature allows applying a custom function to individual inputs in the
    input pipeline. The `function` parameter must be wrapped in an **outer
    function** that can depend on other properties of the pipeline. 
    The **inner function** processes a single input.

    Parameters
    ----------
    function: Callable[..., Callable[[Image], Image]]
        A callable that produces a function. The outer function can accept 
        additional arguments from the pipeline, while the inner function 
        operates on a single image.
    **kwargs: dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: Any, function: Callable[[Any], Any], **kwargs: Any) -> Any`
        Applies the custom function to the input image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Define a factory function that returns a scaling function:
    >>> def scale_function_factory(scale=2):
    ...     def scale_function(image):
    ...         return image * scale
    ...     return scale_function

    Create a `Lambda` feature that scales images by a factor of 5:
    >>> lambda_feature = dt.Lambda(function=scale_function_factory, scale=5)

    Create an image:
    >>> import numpy as np
    >>> 
    >>> input_image = np.ones((2, 3))
    >>> input_image
    array([[1., 1., 1.],
        [1., 1., 1.]])

    Apply the feature to the image:
    >>> output_image = lambda_feature(input_image)
    >>> output_image
    array([[5., 5., 5.],
        [5., 5., 5.]])

    """

    def __init__(
        self: Feature,
        function: Callable[..., Callable[[Any], Any]],
        **kwargs: Any,
    ):
        """Initialize the Lambda feature.

        This feature applies a user-defined function to process an input. The 
        `function` parameter must be a callable that returns another function, 
        where the inner function operates on the input.

        Parameters
        ----------
        function: Callable[..., Callable[[Any], Any]]
            A callable that produces a function. The outer function can accept 
            additional arguments from the pipeline, while the inner function 
            processes a single input.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(function=function, **kwargs)

    def get(
        self: Feature,
        image: Any,
        function: Callable[[Any], Any],
        **kwargs: Any,
    ) -> Any:
        """Apply the custom function to the input.

        This method applies a user-defined function to transform the input. The
        function should be a callable that takes an input and returns a
        modified version of it.

        Parameters
        ----------
        image: Any
            The input to be processed.
        function: Callable[[Any], Any]
            A callable function that takes an input and returns a transformed 
            output.
        **kwargs: Any
            Additional keyword arguments (unused in this implementation).

        Returns
        -------
        Any
            The transformed output after applying the function.

        """

        return function(image)


class Merge(Feature):
    """Apply a custom function to a list of inputs.

    This feature allows applying a user-defined function to a list of inputs. 
    The `function` parameter must be a callable that returns another function, 
    where:
      - The **outer function** can depend on other properties in the pipeline.
      - The **inner function** takes a list of inputs and returns a single 
      outputs or a list of outputs.
    
    The function must be wrapped in an outer layer to enable dependencies on
    other properties while ensuring correct execution.

    Parameters
    ----------
    function: Callable[..., Callable[[list[Any]], Any or list[Any]]
        A callable that produces a function. The outer function can depend on
        other properties of the pipeline, while the inner function processes a
        list of inputs and returns either a single output or a list of outputs.
    **kwargs: Any
        Additional parameters passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        It defaults to `False`.

    Methods
    -------
    `get(list_of_images: list[Any], function: Callable[[list[Any]], Any or list[Any]], **kwargs: Any) -> Any or list[Any]`
        Applies the custom function to the list of inputs.

    Examples
    --------
    >>> import deeptrack as dt

    Define a merge function that averages multiple images:
    >>> def merge_function_factory():
    ...     def merge_function(images):
    ...         return np.mean(np.stack(images), axis=0)
    ...     return merge_function

    Create a Merge feature:
    >>> merge_feature = dt.Merge(function=merge_function_factory)

    Create some images:
    >>> import numpy as np
    >>>
    >>> image_1 = np.ones((2, 3)) * 2
    >>> image_2 = np.ones((2, 3)) * 4

    Apply the feature to a list of images:
    >>> output_image = merge_feature([image_1, image_2])
    >>> output_image
    array([[3., 3., 3.],
        [3., 3., 3.]])

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        function: Callable[..., 
                           Callable[[list[np.ndarray] | list[Image]], np.ndarray | list[np.ndarray] | Image | list[Image]]],
        **kwargs: dict[str, Any]
    ):
        """Initialize the Merge feature.

        Parameters
        ----------
        function: Callable[..., Callable[list[Any]], Any or list[Any]]
            A callable that returns a function for processing a list of images.
            The outer function can depend on other properties in the pipeline.
            The inner function takes a list of inputs and returns either a
            single output or a list of outputs.
        **kwargs: Any
            Additional parameters passed to the parent `Feature` class.

        """

        super().__init__(function=function, **kwargs)

    def get(
        self: Feature,
        list_of_images: list[np.ndarray] | list[Image],
        function: Callable[[list[np.ndarray] | list[Image]], np.ndarray | list[np.ndarray] | Image | list[Image]],
        **kwargs: Any,
    ) -> Image | list[Image]:
        """Apply the custom function to a list of inputs.

        Parameters
        ----------
        list_of_images: list[Any]
            A list of inputs to be processed by the function.
        function: Callable[[list[Any]], Any | list[Any]]
            The function that processes the list of images and returns either a
            single transformed input or a list of transformed inputs.
        **kwargs: Any
            Additional arguments (unused in this implementation).

        Returns
        -------
        Image | list[Image]
            The processed image(s) after applying the function.

        """

        return function(list_of_images)


class OneOf(Feature):
    """Resolve one feature from a given collection.

    This feature selects and applies one of multiple features from a given 
    collection. The default behavior selects a feature randomly, but this 
    behavior can be controlled by specifying a `key`, which determines the 
    index of the feature to apply.

    The `collection` should be an iterable (e.g., list, tuple, or set), and it 
    will be converted to a tuple internally to ensure consistent indexing.

    Parameters
    ----------
    collection: Iterable[Feature]
        A collection of features to choose from.
    key: int | None, optional
        The index of the feature to resolve from the collection. If not 
        provided, a feature is selected randomly at each execution.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        It defaults to `False`.

    Methods
    -------
    `_process_properties(propertydict: dict) -> dict`
        It processes the properties to determine the selected feature index.
    `get(image: Any, key: int, _ID: tuple[int, ...], **kwargs: Any) -> Any`
        It applies the selected feature to the input.
  
    Examples
    --------
    >>> import deeptrack as dt

    Define multiple features:
    >>> feature_1 = dt.Add(value=10)
    >>> feature_2 = dt.Multiply(value=2)
    
    Create a `OneOf` feature that randomly selects a transformation:
    >>> one_of_feature = dt.OneOf([feature_1, feature_2])

    Create an input image:
    >>> import numpy as np
    >>>
    >>> input_image = np.array([1, 2, 3])

    Apply the `OneOf` feature to the input image:
    >>> output_image = one_of_feature(input_image)
    >>> output_image  # The output depends on the randomly selected feature.

    Use `key` to apply a specific feature:
    >>> controlled_feature = dt.OneOf([feature_1, feature_2], key=0)
    >>> output_image = controlled_feature(input_image)
    >>> output_image
    array([11, 12, 13])

    >>> controlled_feature.key.set_value(1)
    >>> output_image = controlled_feature(input_image)
    >>> output_image
    array([2, 4, 6])

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        collection: Iterable[Feature],
        key: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the OneOf feature.

        Parameters
        ----------
        collection: Iterable[Feature]
            A collection of features to choose from. It will be stored as a
            tuple.
        key: int | None, optional
            The index of the feature to resolve from the collection. If not 
            provided, a feature is selected randomly at execution.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(key=key, **kwargs)

        self.collection = tuple(collection)

        # Add all features in the collection as dependencies.
        for feature in self.collection:
            self.add_feature(feature)

    def _process_properties(
        self: Feature,
        propertydict: dict,
    ) -> dict:
        """Process the properties to determine the feature index.

        If `key` is not provided, a random feature index is assigned.
        
        Parameters
        ----------
        propertydict: dict
            The dictionary containing properties of the feature.

        Returns
        -------
        dict
            The updated property dictionary with the `key` property set.

        """

        super()._process_properties(propertydict)

        # Randomly sample a feature index if `key` is not specified.
        if propertydict["key"] is None:
            propertydict["key"] = np.random.randint(len(self.collection))

        return propertydict

    def get(
        self: Feature,
        image: Any,
        key: int,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Apply the selected feature to the input image.

        Parameters
        ----------
        image: Any
            The input image or data to process.
        key: int
            The index of the feature to apply from the collection.
        _ID: tuple[int, ...], optional
            A unique identifier for caching and parallel processing.
        **kwargs: Any
            Additional parameters passed to the selected feature.

        Returns
        -------
        Any
            The output of the selected feature applied to the input image.

        """

        return self.collection[key](image, _ID=_ID)


class OneOfDict(Feature):
    """Resolve one feature from a dictionary and apply it to an input.

    This feature selects a feature from a dictionary and applies it to an
    input.  The selection is made randomly by default, but it can be controlled
    using the `key` argument.

    If `key` is not specified, a random key from the dictionary is selected, 
    and the corresponding feature is applied. Otherwise, the feature mapped to 
    `key` is resolved.

    Parameters
    ----------
    collection: dict[Any, Feature]
        A dictionary where keys are identifiers and values are features.
    key: Any | None, optional
        The key of the feature to resolve from the dictionary. If `None`, 
        a random key is selected.
    **kwargs: Any
        Additional parameters passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        It defaults to `False`.

    Methods
    -------
    `_process_properties(propertydict: dict) -> dict`
        It determines which feature to use based on `key`.
    `get(image: Any, key: Any, _ID: tuple[int, ...], **kwargs: Any) -> Any`
        It resolves the selected feature and applies it to the input image.
   
    Examples
    --------
    >>> import deeptrack as dt

    Define a dictionary of features:
    >>> features_dict = {
    ...     "add": dt.Add(value=10),
    ...     "multiply": dt.Multiply(value=2),
    ... }

    Create a `OneOfDict` feature that randomly selects a transformation:
    >>> one_of_dict_feature = dt.OneOfDict(features_dict)

    Creare an image:
    >>> import numpy as np
    >>>
    >>> input_image = np.array([1, 2, 3])

    Apply a randomly selected feature to the image:
    >>> output_image = one_of_dict_feature(input_image)
    >>> output_image  # The output depends on the randomly selected feature.

    Potentially select a different feature:
    >>> output_image = one_of_dict_feature.update()(input_image)
    >>> output_image

    Use a specific key to apply a predefined feature:
    >>> controlled_feature = dt.OneOfDict(features_dict, key="add")
    >>> output_image = controlled_feature(input_image)
    >>> output_image
    array([11, 12, 13])

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        collection: dict[Any, Feature],
        key: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize the OneOfDict feature.

        Parameters
        ----------
        collection: dict[Any, Feature]
            A dictionary where keys are identifiers and values are features.
        key: Any | None, optional
            The key of the feature to resolve from the dictionary. If `None`, 
            a random key is selected.
        **kwargs: Any
            Additional parameters passed to the parent `Feature` class.

        """

        super().__init__(key=key, **kwargs)

        self.collection = collection

        # Add all features in the dictionary as dependencies.
        for feature in self.collection.values():
            self.add_feature(feature)

    def _process_properties(
        self: Feature,
        propertydict: dict,
    ) -> dict:
        """Determine which feature to apply based on the selected key.

        If no key is provided, a random key from `collection` is selected.

        Parameters
        ----------
        propertydict: dict
            The dictionary containing feature properties.

        Returns
        -------
        dict
            The updated property dictionary with the `key` property set.

        """

        super()._process_properties(propertydict)

        # Randomly sample a key if `key` is not specified.
        if propertydict["key"] is None:
            propertydict["key"] = np.random.choice(list(self.collection.keys()))

        return propertydict

    def get(
        self: Feature,
        image: Any,
        key: Any,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    )-> Any:
        """Resolve the selected feature and apply it to the input.

        Parameters
        ----------
        image: Any
            The input image or data to be processed.
        key: Any
            The key of the feature to apply from the dictionary.
        _ID: tuple[int, ...], optional
            A unique identifier for caching and parallel execution.
        **kwargs: Any
            Additional parameters passed to the selected feature.

        Returns
        -------
        Any
            The output of the selected feature applied to the input.

        """

        return self.collection[key](image, _ID=_ID)


class LoadImage(Feature):
    """Load an image from disk and preprocess it.

    This feature loads an image file using multiple fallback file readers 
    (`imageio`, `numpy`, `Pillow`, and `OpenCV`) until a suitable reader is 
    found. The image can be optionally converted to grayscale, reshaped to 
    ensure a minimum number of dimensions, or treated as a list of images if 
    multiple paths are provided.

    Parameters
    ----------
    path: PropertyLike[str or list[str]]
        The path(s) to the image(s) to load. Can be a single string or a list 
        of strings.
    load_options: PropertyLike[dict[str, Any]], optional
        Additional options passed to the file reader. It defaults to `None`.
    as_list: PropertyLike[bool], optional
        If `True`, the first dimension of the image will be treated as a list. 
        It defaults to `False`.
    ndim: PropertyLike[int], optional
        Ensures the image has at least this many dimensions. It defaults to
        `3`.
    to_grayscale: PropertyLike[bool], optional
        If `True`, converts the image to grayscale. It defaults to `False`.
    get_one_random: PropertyLike[bool], optional
        If `True`, extracts a single random image from a stack of images. Only 
        used when `as_list` is `True`. It defaults to `False`.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        It defaults to `False`.

    Methods
    -------
    `get(image: Any, path: str or list[str], load_options: dict[str, Any] | None, ndim: int, to_grayscale: bool, as_list: bool, get_one_random: bool, **kwargs: Any) -> array`
        Load the image(s) from disk and process them.

    Raises
    ------
    IOError
        If no file reader could parse the file or the file does not exist.

    Examples
    --------
    >>> import deeptrack as dt

    Create a temporary image file:
    >>> import numpy as np
    >>> import os, tempfile
    >>> 
    >>> temp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    >>> np.save(temp_file.name, np.random.rand(100, 100, 3))

    Load the image using `LoadImage`:
    >>> load_image_feature = dt.LoadImage(path=temp_file.name)
    >>> loaded_image = load_image_feature.resolve()

    Print image shape:
    >>> loaded_image.shape
    (100, 100, 3)

    If `to_grayscale=True`, the image is converted to single channel:
    >>> load_image_feature = dt.LoadImage(
    ...     path=temp_file.name,
    ...     to_grayscale=True,
    ... )
    >>> loaded_image = load_image_feature.resolve()
    >>> loaded_image.shape
    (100, 100, 1)

    If `ndim=4`, additional dimensions are added if necessary:
    >>> load_image_feature = dt.LoadImage(
    ...     path=temp_file.name,
    ...     ndim=4,
    ... )
    >>> loaded_image = load_image_feature.resolve()
    >>> loaded_image.shape
    (2, 2, 3, 1)

    Cleanup the temporary file:
    >>> os.remove(temp_file.name)

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        path: PropertyLike[str | list[str]],
        load_options: PropertyLike[dict] = None,
        as_list: PropertyLike[bool] = False,
        ndim: PropertyLike[int] = 3,
        to_grayscale: PropertyLike[bool] = False,
        get_one_random: PropertyLike[bool] = False,
        **kwargs: Any,
    ):
        """Initialize the LoadImage feature.

        Parameters
        ----------
        path: PropertyLike[str or list[str]]
            The path(s) to the image(s) to load. Can be a single string or a
            list of strings.
        load_options: PropertyLike[dict[str, Any]], optional
            Additional options passed to the file reader (e.g., `mode` for
            OpenCV, `allow_pickle` for NumPy). It defaults to `None`.
        as_list: PropertyLike[bool], optional
            If `True`, treats the first dimension of the image as a list of
            images. It defaults to `False`.
        ndim: PropertyLike[int], optional
            Ensures the image has at least this many dimensions. If the loaded
            image has fewer dimensions, extra dimensions are added. It defaults
            to `3`.
        to_grayscale: PropertyLike[bool], optional
            If `True`, converts the image to grayscale. It defaults to `False`.
        get_one_random: PropertyLike[bool], optional
            If `True`, selects a single random image from a stack when
            `as_list=True`. It defaults to `False`.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class, 
            allowing further customization.

        """

        super().__init__(
            path=path,
            load_options=load_options,
            as_list=as_list,
            ndim=ndim,
            to_grayscale=to_grayscale,
            get_one_random=get_one_random,
            **kwargs,
        )

    def get(
        self: Feature,
        *ign: Any,
        path: str | list[str],
        load_options: dict[str, Any] | None,
        ndim: int,
        to_grayscale: bool,
        as_list: bool,
        get_one_random: bool,
        **kwargs: Any,
    ) -> NDArray | torch.Tensor:
        """Load and process an image or a list of images from disk.

        This method attempts to load an image using multiple file readers 
        (`imageio`, `numpy`, `Pillow`, and `OpenCV`) until a valid format is 
        found. It supports optional processing steps such as ensuring a minimum
        number of dimensions, grayscale conversion, and treating multi-frame 
        images as lists.

        Parameters
        ----------
        path: str or list[str]
            The file path(s) to the image(s) to be loaded. A single string 
            loads one image, while a list of paths loads multiple images.
        load_options: dict of str to Any, optional
            Additional options passed to the file reader (e.g., `allow_pickle` 
            for NumPy, `mode` for OpenCV). It defaults to `None`.
        ndim: int
            Ensures the image has at least this many dimensions. If the loaded 
            image has fewer dimensions, extra dimensions are added.
        to_grayscale: bool
            If `True`, converts the image to grayscale. It defaults to `False`.
        as_list: bool
            If `True`, treats the first dimension as a list of images instead 
            of stacking them into a NumPy array.
        get_one_random: bool
            If `True`, selects a single random image from a multi-frame stack
            when `as_list=True`. It defaults to `False`.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        array
            The loaded and processed image(s). If `as_list=True`, returns a 
            list of images; otherwise, returns a single NumPy array or PyTorch
            tensor.

        Raises
        ------
        IOError
            If no valid file reader is found or if the specified file does not 
            exist.

        """

        path_is_list = isinstance(path, list)
        if not path_is_list:
            path = [path]
        if load_options is None:
            load_options = {}

        # Try to load the image using various readers.
        try:
            import imageio

            image = [imageio.v3.imread(file) for file in path]
        except (IOError, ImportError, AttributeError, KeyError):
            try:
                image = [np.load(file, **load_options) for file in path]
            except (IOError, ValueError):
                try:
                    import PIL.Image

                    image = [PIL.Image.open(file, **load_options)
                             for file in path]
                except (IOError, ImportError):
                    import cv2

                    image = [cv2.imread(file, **load_options) for file in path]
                    if not image:
                        raise IOError(
                            "No filereader available for file {0}".format(path)
                        )

        # Convert to list or stack as needed.
        if as_list:
            if get_one_random:
                image = image[np.random.randint(len(image))]
            else:
                image = list(image)
        elif path_is_list:
            image = np.stack(image, axis=-1)
        else:
            image = image[0]

        # Convert to grayscale if requested.
        if to_grayscale:
            try:
                import skimage

                image = skimage.color.rgb2gray(image)
            except ValueError:
                import warnings

                warnings.warn(
                    "Non-rgb image, ignoring to_grayscale",
                    UserWarning,
                )

        # Ensure the image has at least `ndim` dimensions.
        while ndim and image.ndim < ndim:
            image = np.expand_dims(image, axis=-1)

        # Convert to PyTorch tensor if needed.
        #TODO

        return image


class SampleToMasks(Feature):
    """Create a mask from a list of images.

    This feature applies a transformation function to each input image and 
    merges the resulting masks into a single multi-layer image. Each input 
    image must have a `position` property that determines its placement within 
    the final mask. When used with scatterers, the `voxel_size` property must 
    be provided for correct object sizing.

    Parameters
    ----------
    transformation_function: Callable[[Image], Image]
        A function that transforms each input image into a mask with 
        `number_of_masks` layers.
    number_of_masks: PropertyLike[int], optional
        The number of mask layers to generate. Default is 1.
    output_region: PropertyLike[tuple[int, int, int, int]], optional
        The size and position of the output mask, typically aligned with 
        `optics.output_region`.
    merge_method: PropertyLike[str | Callable | list[str | Callable]], optional
        Method for merging individual masks into the final image. Can be:
        - "add" (default): Sum the masks.
        - "overwrite": Later masks overwrite earlier masks.
        - "or": Combine masks using a logical OR operation.
        - "mul": Multiply masks.
        - Function: Custom function taking two images and merging them.

    **kwargs: dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray | Image, transformation_function: Callable[[Image], Image], **kwargs: dict[str, Any]) -> Image`
        Applies the transformation function to the input image.
    `_process_and_get(images: list[np.ndarray] | np.ndarray | list[Image] | Image, **kwargs: dict[str, Any]) -> Image | np.ndarray`
        Processes a list of images and generates a multi-layer mask.

    Returns
    -------
    Image or np.ndarray
        The final mask image with the specified number of layers.

    Raises
    ------
    ValueError
        If `merge_method` is invalid.

    Examples
    -------
    >>> import deeptrack as dt

    Define number of particles:
    >>> n_particles = 12

    Define optics and particles:
    >>> import numpy as np
    >>>    
    >>> optics = dt.Fluorescence(output_region=(0, 0, 64, 64))
    >>> particle = dt.PointParticle(
    >>>     position=lambda: np.random.uniform(5, 55, size=2),
    >>> )
    >>> particles = particle ^ n_particles

    Define pipelines:
    >>> sim_im_pip = optics(particles)
    >>> sim_mask_pip = particles >> dt.SampleToMasks(
    ...     lambda: lambda particles: particles > 0,
    ...     output_region=optics.output_region,
    ...     merge_method="or",
    ... )
    >>> pipeline = sim_im_pip & sim_mask_pip
    >>> pipeline.store_properties()

    Generate image and mask:
    >>> image, mask = pipeline.update()()

    Get particle positions:
    >>> positions = np.array(image.get_property("position", get_one=False))

    Visualize results:
    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(image, cmap="gray")
    >>> plt.title("Original Image")
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(mask, cmap="gray")
    >>> plt.scatter(positions[:,1], positions[:,0], c="y", marker="x", s = 50)
    >>> plt.title("Mask")
    >>> plt.show()

    """

    def __init__(
        self: Feature,
        transformation_function: Callable[[Image], Image],
        number_of_masks: PropertyLike[int] = 1,
        output_region: PropertyLike[tuple[int, int, int, int]] = None,
        merge_method: PropertyLike[str | Callable | list[str | Callable]] = "add",
        **kwargs: Any,
    ):
        """Initialize the SampleToMasks feature.

        Parameters
        ----------
        transformation_function: Callable[[Image], Image]
            Function to transform input images into masks.
        number_of_masks: PropertyLike[int], optional
            Number of mask layers. Default is 1.
        output_region: PropertyLike[tuple[int, int, int, int]], optional
            Output region of the mask. Default is None.
        merge_method: PropertyLike[str | Callable | list[str | Callable]], optional
            Method to merge masks. Default is "add".
        **kwargs: dict[str, Any]
            Additional keyword arguments passed to the parent class.
        
        """

        super().__init__(
            transformation_function=transformation_function,
            number_of_masks=number_of_masks,
            output_region=output_region,
            merge_method=merge_method,
            **kwargs,
        )

    def get(
        self: Feature,
        image: np.ndarray | Image,
        transformation_function: Callable[[Image], Image],
        **kwargs: Any,
    ) -> Image:
        """Apply the transformation function to a single image.

        Parameters
        ----------
        image: np.ndarray | Image
            The input image.
        transformation_function: Callable[[Image], Image]
            Function to transform the image.
        **kwargs: dict[str, Any]
            Additional parameters.

        Returns
        -------
        Image
            The transformed image.

        """

        return transformation_function(image)

    def _process_and_get(
        self: Feature,
        images: list[np.ndarray] | np.ndarray | list[Image] | Image,
        **kwargs: Any,
    ) -> Image | np.ndarray:
        """Process a list of images and generate a multi-layer mask.

        Parameters
        ----------
        images: np.ndarray or list[np.ndarrray] or  Image or list[Image]
            List of input images or a single image.
        **kwargs: dict[str, Any]
            Additional parameters including `output_region`, `number_of_masks`, 
            and `merge_method`.

        Returns
        -------
        Image or np.ndarray
            The final mask image.
            
        """

        # Handle list of images.
        if isinstance(images, list) and len(images) != 1:
            list_of_labels = super()._process_and_get(images, **kwargs)
            if not self._wrap_array_with_image:
                for idx, (label, image) in enumerate(zip(list_of_labels, 
                                                         images)):
                    list_of_labels[idx] = \
                        Image(label, copy=False).merge_properties_from(image)
        else:
            if isinstance(images, list):
                images = images[0]
            list_of_labels = []
            for prop in images.properties:

                if "position" in prop:

                    inp = Image(np.array(images))
                    inp.append(prop)
                    out = Image(self.get(inp, **kwargs))
                    out.merge_properties_from(inp)
                    list_of_labels.append(out)

        # Create an empty output image.
        output_region = kwargs["output_region"]
        output = np.zeros(
            (
                output_region[2] - output_region[0],
                output_region[3] - output_region[1],
                kwargs["number_of_masks"],
            )
        )

        from deeptrack.optics import _get_position

        # Merge masks into the output.
        for label in list_of_labels:
            position = _get_position(label)
            p0 = np.round(position - output_region[0:2])

            if np.any(p0 > output.shape[0:2]) or \
                np.any(p0 + label.shape[0:2] < 0):
                continue

            crop_x = int(-np.min([p0[0], 0]))
            crop_y = int(-np.min([p0[1], 0]))
            crop_x_end = int(
                label.shape[0]
                - np.max([p0[0] + label.shape[0] - output.shape[0], 0])
            )
            crop_y_end = int(
                label.shape[1]
                - np.max([p0[1] + label.shape[1] - output.shape[1], 0])
            )

            labelarg = label[crop_x:crop_x_end, crop_y:crop_y_end, :]

            p0[0] = np.max([p0[0], 0])
            p0[1] = np.max([p0[1], 0])

            p0 = p0.astype(int)

            output_slice = output[
                p0[0] : p0[0] + labelarg.shape[0],
                p0[1] : p0[1] + labelarg.shape[1],
            ]

            for label_index in range(kwargs["number_of_masks"]):

                if isinstance(kwargs["merge_method"], list):
                    merge = kwargs["merge_method"][label_index]
                else:
                    merge = kwargs["merge_method"]

                if merge == "add":
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] += labelarg[..., label_index]

                elif merge == "overwrite":
                    output_slice[
                        labelarg[..., label_index] != 0, label_index
                    ] = labelarg[labelarg[..., label_index] != 0, \
                        label_index]
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] = output_slice[..., label_index]

                elif merge == "or":
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] = (output_slice[..., label_index] != 0) | (
                        labelarg[..., label_index] != 0
                    )

                elif merge == "mul":
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] *= labelarg[..., label_index]

                else:
                    # No match, assume function
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] = merge(
                        output_slice[..., label_index],
                        labelarg[..., label_index],
                    )

        if not self._wrap_array_with_image:
            return output
        output = Image(output)
        for label in list_of_labels:
            output.merge_properties_from(label)
        return output


class AsType(Feature):
    """Convert the data type of images.

    This feature changes the data type (`dtype`) of input images to a specified 
    type. The accepted types are the same as those used by NumPy arrays, such 
    as `float64`, `int32`, `uint16`, `int16`, `uint8`, and `int8`.

    Parameters
    ----------
    dtype: PropertyLike[str], optional
        The desired data type for the image. It defaults to `"float64"`.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: array, dtype: str, **kwargs: Any) -> array`
        Convert the data type of the input image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input array:
    >>> import numpy as np
    >>>
    >>> input_image = np.array([1.5, 2.5, 3.5])

    Apply an AsType feature to convert to `int32`:
    >>> astype_feature = dt.AsType(dtype="int32")
    >>> output_image = astype_feature.get(input_image, dtype="int32")
    >>> output_image
    array([1, 2, 3], dtype=int32)

    Verify the data type:
    >>> output_image.dtype
    dtype('int32')

    """

    def __init__(
        self: Feature,
        dtype: PropertyLike[str] = "float64",
        **kwargs: Any,
    ):
        """
        Initialize the AsType feature.

        Parameters
        ----------
        dtype: PropertyLike[str], optional
            The desired data type for the image. It defaults to `"float64"`.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(dtype=dtype, **kwargs)

    def get(
        self: Feature,
        image: NDArray | torch.Tensor | Image,
        dtype: str,
        **kwargs: Any,
    ) -> NDArray | torch.Tensor | Image:
        """Convert the data type of the input image.

        Parameters
        ----------
        image: array
            The input image to process. It can be a NumPy array, a PyTorch
            tensor, or an Image.
        dtype: str
            The desired data type for the image.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array
            The input image converted to the specified data type. It can be a
            NumPy array, a PyTorch tensor, or an Image.

        """

        return image.astype(dtype)


class ChannelFirst2d(Feature):  # DEPRECATED
    """Convert an image to a channel-first format.

    This feature rearranges the axes of a 3D image so that the specified axis 
    (e.g., channel axis) is moved to the first position. If the input image is 
    2D, it adds a new dimension at the front, effectively treating the 2D 
    image as a single-channel image.

    Parameters
    ----------
    axis: int, optional
        The axis to move to the first position. It defaults to `-1` (last axis).
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, axis: int, **kwargs: dict[str, Any]) -> np.ndarray`
        Rearrange the axes of an image to channel-first format.

    Examples
    --------
    >>> import numpy as np
    >>> from deeptrack.features import ChannelFirst2d

    Create a 2D input array:
    >>> input_image_2d = np.random.rand(10, 10)
    >>> print(input_image_2d.shape)
    (10, 10)

    Convert it to channel-first format:
    >>> channel_first_feature = ChannelFirst2d()
    >>> output_image = channel_first_feature.get(input_image_2d, axis=-1)
    >>> print(output_image.shape)
    (1, 10, 10)

    Create a 3D input array:
    >>> input_image_3d = np.random.rand(10, 10, 3)
    >>> print(input_image_3d.shape)
    (10, 10, 3)

    Convert it to channel-first format:
    >>> output_image = channel_first_feature.get(input_image_3d, axis=-1)
    >>> print(output_image.shape)
    (3, 10, 10)

    """

    def __init__(
        self: Feature,
        axis: int = -1,
        **kwargs: Any,
    ):
        """Initialize the ChannelFirst2d feature.

        Parameters
        ----------
        axis: int, optional
            The axis to move to the first position. 
            It defaults to `-1` (last axis).
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Feature,
        image: np.ndarray,
        axis: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Rearrange the axes of an image to channel-first format.

        Rearrange the axes of a 3D image to channel-first format or add a 
        channel dimension to a 2D image.

        Parameters
        ----------
        image: np.ndarray
            The input image to process. Can be 2D or 3D.
        axis: int
            The axis to move to the first position (for 3D images).
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray
            The processed image in channel-first format.

        Raises
        ------
        ValueError
            If the input image is neither 2D nor 3D.

        """

        ndim = image.ndim

        # Add a new dimension for 2D images.
        if ndim == 2:
            return image[None]

        # Move the specified axis to the first position for 3D images.
        if ndim == 3:
            return np.moveaxis(image, axis, 0)

        raise ValueError("ChannelFirst2d only supports 2D or 3D images. "
                         f"Received {ndim}D image.")


class Upscale(Feature):
    """Simulate a pipeline at a higher resolution.

    This feature scales up the resolution of the input pipeline by a specified 
    factor, performs computations at the higher resolution, and then 
    downsamples the result back to the original size. This is useful for 
    simulating effects at a finer resolution while preserving compatibility 
    with lower-resolution pipelines.
    
    Internally, this feature redefines the scale of physical units (e.g., 
    `units.pixel`) to achieve the effect of upscaling. It does not resize the 
    input image itself but affects features that rely on physical units.

    Parameters
    ----------
    feature: Feature
        The pipeline or feature to resolve at a higher resolution.
    factor: int or tuple[int, int, int], optional
        The factor by which to upscale the simulation. If a single integer is 
        provided, it is applied uniformly across all axes. If a tuple of three 
        integers is provided, each axis is scaled individually. It defaults to 1.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        Always `False` for `Upscale`.

    Methods
    -------
    `get(image: np.ndarray | Image, factor: int | tuple[int, int, int], **kwargs) -> np.ndarray`
        Simulates the pipeline at a higher resolution and returns the result at 
        the original resolution.

    Notes
    -----
    - This feature does **not** directly resize the image. Instead, it modifies
      the unit conversions within the pipeline, making physical units smaller, 
      which results in more detail being simulated.
    - The final output is downscaled back to the original resolution using 
      `block_reduce` from `skimage.measure`.
    - The effect is only noticeable if features use physical units (e.g., 
      `units.pixel`, `units.meter`). Otherwise, the result will be identical.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import matplotlib.pyplot as plt

    Define an optical pipeline and a spherical particle:
    >>> optics = dt.Fluorescence()
    >>> particle = dt.Sphere()
    >>> simple_pipeline = optics(particle)

    Create an upscaled pipeline with a factor of 4:
    >>> upscaled_pipeline = dt.Upscale(optics(particle), factor=4) 
    
    Resolve the pipelines:
    >>> image = simple_pipeline()
    >>> upscaled_image = upscaled_pipeline()

    Visualize the images:
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(image, cmap="gray")
    >>> plt.title("Original Image")
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(upscaled_image, cmap="gray")
    >>> plt.title("Simulated at Higher Resolution")
    >>> plt.show()
    
    Compare the shapes (both are the same due to downscaling):
    >>> print(image.shape)
    (128, 128, 1)
    >>> print(upscaled_image.shape)
    (128, 128, 1)
    
    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        feature: Feature,
        factor: int | tuple[int, int, int] = 1,
        **kwargs: Any,
    ):
        """Initialize the Upscale feature.

        Parameters
        ----------
        feature: Feature
            The pipeline or feature to resolve at a higher resolution.
        factor: int or tuple[int, int, int], optional
            The factor by which to upscale the simulation. If a single integer 
            is provided, it is applied uniformly across all axes. If a tuple of
            three integers is provided, each axis is scaled individually. 
            It defaults to `1`.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(factor=factor, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Feature,
        image: np.ndarray,
        factor: int | tuple[int, int, int],
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulate the pipeline at a higher resolution and return result.

        Parameters
        ----------
        image: np.ndarray
            The input image to process.
        factor: int or tuple[int, int, int]
            The factor by which to upscale the simulation. If a single integer 
            is provided, it is applied uniformly across all axes. If a tuple of
            three integers is provided, each axis is scaled individually.
        **kwargs: Any
            Additional keyword arguments passed to the feature.

        Returns
        -------
        np.ndarray
            The processed image at the original resolution.

        Raises
        ------
        ValueError
            If the input `factor` is not a valid integer or tuple of integers.

        """

        # Ensure factor is a tuple of three integers.
        if np.size(factor) == 1:
            factor = (factor,) * 3
        elif len(factor) != 3:
            raise ValueError(
                "Factor must be an integer or a tuple of three integers."
            )

        # Create a context for upscaling and perform computation.
        ctx = create_context(None, None, None, *factor)
        with units.context(ctx):
            image = self.feature(image)

        # Downscale the result to the original resolution.        
        import skimage.measure

        image = skimage.measure.block_reduce(
            image, (factor[0], factor[1]) + (1,) * (image.ndim - 2), np.mean
        )

        return image


class NonOverlapping(Feature):
    """Ensure volumes are placed non-overlapping in a 3D space.

    This feature ensures that a list of 3D volumes are positioned such that 
    their non-zero voxels do not overlap. If volumes overlap, their positions 
    are resampled until they are non-overlapping. If the maximum number of 
    attempts is exceeded, the feature regenerates the list of volumes and 
    raises a warning if non-overlapping placement cannot be achieved.
    
    Note: `min_distance` refers to the distance between the edges of volumes, 
    not their centers. Due to the way volumes are calculated, slight rounding 
    errors may affect the final distance.
    
    This feature is incompatible with non-volumetric scatterers such as 
    `MieScatterers`.
    
    Parameters
    ----------
    feature: Feature
        The feature that generates the list of volumes to place 
        non-overlapping.
    min_distance: float, optional
        The minimum distance between volumes in pixels. It defaults to `1`. 
        It can be negative to allow for partial overlap.
    max_attempts: int, optional
        The maximum number of attempts to place volumes without overlap.
        It defaults to `5`. 
    max_iters: int, optional
        The maximum number of resamplings. If this number is exceeded, a 
            new list of volumes is generated. It defaults to `100`.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        Always `False` for `NonOverlapping`.

    Methods
    -------
    `get(_: Any, min_distance: float, max_attempts: int, **kwargs: dict[str, Any]) -> list[np.ndarray]`
        Generate a list of non-overlapping 3D volumes.
    `_check_non_overlapping(list_of_volumes: list[np.ndarray]) -> bool`
        Check if all volumes in the list are non-overlapping.
    `_check_bounding_cubes_non_overlapping(bounding_cube_1: list[int], bounding_cube_2: list[int], min_distance: float) -> bool`
        Check if two bounding cubes are non-overlapping.
    `_get_overlapping_cube(bounding_cube_1: list[int], bounding_cube_2: list[int]) -> list[int]`
        Get the overlapping cube between two bounding cubes.
    `_get_overlapping_volume(volume: np.ndarray, bounding_cube: tuple[float, float, float, float, float, float], overlapping_cube: tuple[float, float, float, float, float, float]) -> np.ndarray`
        Get the overlapping volume between a volume and a bounding cube.
    `_check_volumes_non_overlapping(volume_1: np.ndarray, volume_2: np.ndarray, min_distance: float) -> bool`
        Check if two volumes are non-overlapping.
    `_resample_volume_position(volume: np.ndarray | Image) -> Image`
        Resample the position of a volume to avoid overlap.
    
    Notes
    -----
    - This feature performs **bounding cube checks first** to **quickly 
      reject** obvious overlaps before voxel-level checks.
    - If the bounding cubes overlap, precise **voxel-based checks** are 
      performed.

    Examples
    ---------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Define an ellipse scatterer with randomly positioned objects:
    >>> scatterer = dt.Ellipse(
    >>>    radius= 13 * dt.units.pixels,
    >>>    position=lambda: np.random.uniform(5, 115, size=2)* dt.units.pixels,
    >>> )

    Create multiple scatterers:
    >>> scatterers = (scatterer ^ 8)  

    Define the optics and create the image with possible overlap:
    >>> optics = dt.Fluorescence()
    >>> im_with_overlap = optics(scatterers)
    >>> im_with_overlap.store_properties()
    >>> im_with_overlap_resolved = image_with_overlap()

    Gather position from image:
    >>> pos_with_overlap = np.array(
    >>>     im_with_overlap_resolved.get_property(
    >>>         "position", 
    >>>         get_one=False
    >>>     )
    >>> )

    Enforce non-overlapping and create the image without overlap:
    >>> non_overlapping_scatterers = dt.NonOverlapping(scatterers, min_distance=4)
    >>> im_without_overlap =  optics(non_overlapping_scatterers)
    >>> im_without_overlap.store_properties()
    >>> im_without_overlap_resolved = im_without_overlap()

    Gather position from image:
    >>> pos_without_overlap = np.array(
    >>>     im_without_overlap_resolved.get_property(
    >>>         "position",
    >>>        get_one=False
    >>>     )
    >>> )

    Create a figure with two subplots to visualize the difference:
    >>> fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    >>> axes[0].imshow(im_with_overlap_resolved, cmap="gray")
    >>> axes[0].scatter(pos_with_overlap[:,1],pos_with_overlap[:,0])
    >>> axes[0].set_title("Overlapping Objects")
    >>> axes[0].axis("off")
    >>> axes[1].imshow(im_without_overlap_resolved, cmap="gray")
    >>> axes[1].scatter(pos_without_overlap[:,1],pos_without_overlap[:,0])
    >>> axes[1].set_title("Non-Overlapping Objects")
    >>> axes[1].axis("off")
    >>> plt.tight_layout()
    >>> plt.show()

    Define function to calculate minimum distance:
    >>> def calculate_min_distance(positions):
    >>> distances = [
    >>>     np.linalg.norm(positions[i] - positions[j])
    >>>     for i in range(len(positions))
    >>>         for j in range(i + 1, len(positions))
    >>> ]
    >>> return min(distances)

    Print minimum distances with and without overlap:
    >>> print(calculate_min_distance(pos_with_overlap))
    10.768742383382174
    >>> print(calculate_min_distance(pos_without_overlap))
    30.82531120942446

    """

    __distributed__: bool = False

    def __init__(
        self: NonOverlapping,
        feature: Feature,
        min_distance: float = 1,
        max_attempts: int = 5,
        max_iters: int = 100,
        **kwargs: Any,
    ):
        """Initializes the NonOverlapping feature.

        Ensures that volumes are placed **non-overlapping** by iteratively 
        resampling their positions. If the maximum number of attempts is 
        exceeded, the feature regenerates the list of volumes.

        Parameters
        ----------
        feature: Feature
            The feature that generates the list of volumes.
        min_distance: float, optional
            The minimum separation distance **between volume edges**, in 
            pixels. It defaults to `1`. Negative values allow for partial
            overlap.
        max_attempts: int, optional
            The maximum number of attempts to place the volumes without 
            overlap. It defaults to `5`.
        max_iters: int, optional
            The maximum number of resampling iterations per attempt. If 
            exceeded, a new list of volumes is generated. It defaults to `100`.
        
        """

        super().__init__(
            min_distance=min_distance, 
            max_attempts=max_attempts, 
            max_iters=max_iters,
            **kwargs)
        self.feature = self.add_feature(feature, **kwargs)

    def get(
        self: NonOverlapping,
        _: Any,
        min_distance: float,
        max_attempts: int,
        max_iters: int,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Generates a list of non-overlapping 3D volumes within a defined 
        field of view (FOV).

        This method **iteratively** attempts to place volumes while ensuring 
        they maintain at least `min_distance` separation. If non-overlapping 
        placement is not achieved within `max_attempts`, a warning is issued, 
        and the best available configuration is returned.

        Parameters
        ----------
        _: Any
            Placeholder parameter, typically for an input image.
        min_distance: float
            The minimum required separation distance between volumes, in 
            pixels.
        max_attempts: int
            The maximum number of attempts to generate a valid non-overlapping 
            configuration.
        max_iters: int
            The maximum number of resampling iterations per attempt.
        **kwargs: dict[str, Any]
            Additional parameters that may be used by subclasses.

        Returns
        -------
        list[np.ndarray]
            A list of 3D volumes represented as NumPy arrays. If 
            non-overlapping placement is unsuccessful, the best available 
            configuration is returned.

        Warns
        -----
        UserWarning
            If non-overlapping placement is **not** achieved within 
            `max_attempts`, suggesting parameter adjustments such as increasing
            the FOV or reducing `min_distance`.

        Notes
        -----
        - The placement process **prioritizes bounding cube checks** for 
          efficiency.
        - If bounding cubes overlap, **voxel-based overlap checks** are 
          performed.
        
        """

        for _ in range(max_attempts):
            list_of_volumes = self.feature()

            if not isinstance(list_of_volumes, list):
                list_of_volumes = [list_of_volumes]

            for _ in range(max_iters):

                list_of_volumes = [
                    self._resample_volume_position(volume) 
                    for volume in list_of_volumes
                ]

                if self._check_non_overlapping(list_of_volumes):
                    return list_of_volumes

            # Generate a new list of volumes if max_attempts is exceeded.
            self.feature.update()

        import warnings

        warnings.warn(
            "Non-overlapping placement could not be achieved. Consider "
            "adjusting parameters: reduce object radius, increase FOV, "
            "or decrease min_distance.",
            UserWarning,
        )
        return list_of_volumes

    def _check_non_overlapping(
        self: NonOverlapping, 
        list_of_volumes: list[np.ndarray],
    ) -> bool:
        """Determines whether all volumes in the provided list are 
        non-overlapping.

        This method verifies that the non-zero voxels of each 3D volume in 
        `list_of_volumes` are at least `min_distance` apart. It first checks 
        bounding boxes for early rejection and then examines actual voxel 
        overlap when necessary. Volumes are assumed to have a `position` 
        attribute indicating their placement in 3D space.

        Parameters
        ----------
        list_of_volumes: list[np.ndarray]
            A list of 3D arrays representing the volumes to be checked for 
            overlap. Each volume is expected to have a position attribute.

        Returns
        -------
        bool
            `True` if all volumes are non-overlapping, otherwise `False`.

        Notes
        -----
        - If `min_distance` is negative, volumes are shrunk using isotropic 
          erosion before checking overlap.
        - If `min_distance` is positive, volumes are padded and expanded using 
          isotropic dilation.
        - Overlapping checks are first performed on bounding cubes for 
            efficiency.
        - If bounding cubes overlap, voxel-level checks are performed.

        """

        from skimage.morphology import isotropic_erosion, isotropic_dilation

        from deeptrack.augmentations import CropTight, Pad
        from deeptrack.optics import _get_position

        min_distance = self.min_distance()
        crop = CropTight()
        
        if min_distance < 0:
            list_of_volumes = [
                Image(
                    crop(isotropic_erosion(volume != 0, -min_distance/2)),
                    copy=False,
                ).merge_properties_from(volume) 
                for volume in list_of_volumes
            ]
        else:
            pad = Pad(px = [int(np.ceil(min_distance/2))]*6, keep_size=True)
            list_of_volumes = [    
                Image(
                    crop(isotropic_dilation(pad(volume) != 0, min_distance/2)),
                    copy=False,
                ).merge_properties_from(volume) 
            for volume in list_of_volumes 
            ]
        min_distance = 1

        # The position of the top left corner of each volume (index (0, 0, 0)).
        volume_positions_1 = [
            _get_position(volume, mode="corner", return_z=True).astype(int)
            for volume in list_of_volumes
        ]

        # The position of the bottom right corner of each volume 
        # (index (-1, -1, -1)).
        volume_positions_2 = [
            p0 + np.array(v.shape) 
            for v, p0 in zip(list_of_volumes, volume_positions_1)
        ]

        # (x1, y1, z1, x2, y2, z2) for each volume.
        volume_bounding_cube = [
            [*p0, *p1] 
            for p0, p1 in zip(volume_positions_1, volume_positions_2)
        ]

        for i, j in itertools.combinations(range(len(list_of_volumes)), 2):

            # If the bounding cubes do not overlap, the volumes do not overlap.
            if self._check_bounding_cubes_non_overlapping(
                volume_bounding_cube[i], volume_bounding_cube[j], min_distance
            ):
                continue

            # If the bounding cubes overlap, get the overlapping region of each 
            # volume.
            overlapping_cube = self._get_overlapping_cube(
                volume_bounding_cube[i], volume_bounding_cube[j]
            )
            overlapping_volume_1 = self._get_overlapping_volume(
                list_of_volumes[i], volume_bounding_cube[i], overlapping_cube
            )
            overlapping_volume_2 = self._get_overlapping_volume(
                list_of_volumes[j], volume_bounding_cube[j], overlapping_cube
            )

            # If either the overlapping regions are empty, the volumes do not 
            # overlap (done for speed).
            if (np.all(overlapping_volume_1 == 0)
                or np.all(overlapping_volume_2 == 0)):
                continue

            # If products of overlapping regions are non-zero, return False.
            # if np.any(overlapping_volume_1 * overlapping_volume_2):
            #     return False

            # Finally, check that the non-zero voxels of the volumes are at 
            # least min_distance apart.
            if not self._check_volumes_non_overlapping(
                overlapping_volume_1, overlapping_volume_2, min_distance
            ):
                return False

        return True

    def _check_bounding_cubes_non_overlapping(
        self: NonOverlapping,
        bounding_cube_1: list[int],
        bounding_cube_2: list[int], 
        min_distance: float,
    ) -> bool:
        """Determines whether two 3D bounding cubes are non-overlapping.

        This method checks whether the bounding cubes of two volumes are 
        **separated by at least** `min_distance` along **any** spatial axis.

        Parameters
        ----------
        bounding_cube_1: list[int]
            A list of six integers `[x1, y1, z1, x2, y2, z2]` representing 
            the first bounding cube.
        bounding_cube_2: list[int]
            A list of six integers `[x1, y1, z1, x2, y2, z2]` representing 
            the second bounding cube.
        min_distance: float
            The required **minimum separation distance** between the two 
            bounding cubes.

        Returns
        -------
        bool
            `True` if the bounding cubes are non-overlapping (separated by at 
            least `min_distance` along **at least one axis**), otherwise 
            `False`.

        Notes
        -----
        - This function **only checks bounding cubes**, **not actual voxel 
          data**.
        - If the bounding cubes are non-overlapping, the corresponding 
          **volumes are also non-overlapping**.
        - This check is much **faster** than full voxel-based comparisons.
        
        """

        # bounding_cube_1 and bounding_cube_2 are (x1, y1, z1, x2, y2, z2).
        # Check that the bounding cubes are non-overlapping.
        return (
        (bounding_cube_1[0] >= bounding_cube_2[3] + min_distance) or
        (bounding_cube_2[0] >= bounding_cube_1[3] + min_distance) or
        (bounding_cube_1[1] >= bounding_cube_2[4] + min_distance) or
        (bounding_cube_2[1] >= bounding_cube_1[4] + min_distance) or
        (bounding_cube_1[2] >= bounding_cube_2[5] + min_distance) or
        (bounding_cube_2[2] >= bounding_cube_1[5] + min_distance)
        )

    def _get_overlapping_cube(
        self: NonOverlapping,
        bounding_cube_1: list[int],
        bounding_cube_2: list[int],
    ) -> list[int]:
        """Computes the overlapping region between two 3D bounding cubes.

        This method calculates the coordinates of the intersection of two 
        axis-aligned bounding cubes, each represented as a list of six 
        integers:

        - `[x1, y1, z1]`: Coordinates of the **top-left-front** corner.
        - `[x2, y2, z2]`: Coordinates of the **bottom-right-back** corner.

        The resulting overlapping region is determined by:
        - Taking the **maximum** of the starting coordinates (`x1, y1, z1`).
        - Taking the **minimum** of the ending coordinates (`x2, y2, z2`).

        If the cubes **do not** overlap, the resulting coordinates will not 
        form a valid cube (i.e., `x1 > x2`, `y1 > y2`, or `z1 > z2`).

        Parameters
        ----------
        bounding_cube_1: list[int]
            The first bounding cube, formatted as `[x1, y1, z1, x2, y2, z2]`.
        bounding_cube_2: list[int]
            The second bounding cube, formatted as `[x1, y1, z1, x2, y2, z2]`.

        Returns
        -------
        list[int]
            A list of six integers `[x1, y1, z1, x2, y2, z2]` representing the 
            overlapping bounding cube. If no overlap exists, the coordinates 
            will **not** define a valid cube.

        Notes
        -----
        - This function does **not** check for valid input or ensure the 
          resulting cube is well-formed.
        - If no overlap exists, downstream functions must handle the invalid 
          result.
        
        """

        return [
            max(bounding_cube_1[0], bounding_cube_2[0]),
            max(bounding_cube_1[1], bounding_cube_2[1]),
            max(bounding_cube_1[2], bounding_cube_2[2]),
            min(bounding_cube_1[3], bounding_cube_2[3]),
            min(bounding_cube_1[4], bounding_cube_2[4]),
            min(bounding_cube_1[5], bounding_cube_2[5]),
        ]

    def _get_overlapping_volume(
        self: NonOverlapping,
        volume: np.ndarray,  # 3D array.
        bounding_cube: tuple[float, float, float, float, float, float],
        overlapping_cube: tuple[float, float, float, float, float, float],
    ) -> np.ndarray:
        """Extracts the overlapping region of a 3D volume within the specified 
        overlapping cube.

        This method identifies and returns the subregion of `volume` that 
        lies within the `overlapping_cube`. The bounding information of the 
        volume is provided via `bounding_cube`.

        Parameters
        ----------
        volume: np.ndarray
            A 3D NumPy array representing the volume from which the 
            overlapping region is extracted.
        bounding_cube: tuple[float, float, float, float, float, float]
            The bounding cube of the volume, given as a tuple of six floats: 
            `(x1, y1, z1, x2, y2, z2)`. The first three values define the 
            **top-left-front** corner, while the last three values define the 
            **bottom-right-back** corner.
        overlapping_cube: tuple[float, float, float, float, float, float]
            The overlapping region between the volume and another volume, 
            represented in the same format as `bounding_cube`.

        Returns
        -------
        np.ndarray
            A 3D NumPy array representing the portion of `volume` that 
            lies within `overlapping_cube`. If the overlap does not exist, 
            an empty array may be returned.

        Notes
        -----
        - The method computes the relative indices of `overlapping_cube` 
          within `volume` by subtracting the bounding cube's starting 
          position.
        - The extracted region is determined by integer indices, meaning 
          coordinates are implicitly **floored to integers**.
        - If `overlapping_cube` extends beyond `volume` boundaries, the 
          returned subregion is **cropped** to fit within `volume`.
        
        """

        # The position of the top left corner of the overlapping cube in the volume
        overlapping_cube_position = np.array(overlapping_cube[:3]) - np.array(
            bounding_cube[:3]
        )

        # The position of the bottom right corner of the overlapping cube in the volume
        overlapping_cube_end_position = np.array(
            overlapping_cube[3:]
            ) - np.array(bounding_cube[:3])

        # cast to int
        overlapping_cube_position = overlapping_cube_position.astype(int)
        overlapping_cube_end_position = overlapping_cube_end_position.astype(int)

        return volume[
            overlapping_cube_position[0] : overlapping_cube_end_position[0],
            overlapping_cube_position[1] : overlapping_cube_end_position[1],
            overlapping_cube_position[2] : overlapping_cube_end_position[2],
        ]

    def _check_volumes_non_overlapping(
        self: NonOverlapping,
        volume_1: np.ndarray,
        volume_2: np.ndarray,
        min_distance: float,
    ) -> bool:
        """Determines whether the non-zero voxels in two 3D volumes are at 
        least `min_distance` apart.

        This method checks whether the active regions (non-zero voxels) in 
        `volume_1` and `volume_2` maintain a minimum separation of 
        `min_distance`. If the volumes differ in size, the positions of their 
        non-zero voxels are adjusted accordingly to ensure a fair comparison.

        Parameters
        ----------
        volume_1: np.ndarray
            A 3D NumPy array representing the first volume.
        volume_2: np.ndarray
            A 3D NumPy array representing the second volume.
        min_distance: float
            The minimum Euclidean distance required between any two non-zero 
            voxels in the two volumes.

        Returns
        -------
        bool
            `True` if all non-zero voxels in `volume_1` and `volume_2` are at 
            least `min_distance` apart, otherwise `False`.

        Notes
        -----
        - This function assumes both volumes are correctly aligned within a 
          shared coordinate space.
        - If the volumes are of different sizes, voxel positions are scaled 
          or adjusted for accurate distance measurement.
        - Uses **Euclidean distance** for separation checking.
        - If either volume is empty (i.e., no non-zero voxels), they are 
          considered non-overlapping.
        
        """

        # Get the positions of the non-zero voxels of each volume.
        positions_1 = np.argwhere(volume_1)
        positions_2 = np.argwhere(volume_2)

        # if positions_1.size == 0 or positions_2.size == 0:
        #     return True  # If either volume is empty, they are "non-overlapping"

        # # If the volumes are not the same size, the positions of the non-zero 
        # # voxels of each volume need to be scaled.
        # if positions_1.size == 0 or positions_2.size == 0:
        #     return True  # If either volume is empty, they are "non-overlapping"

        # If the volumes are not the same size, the positions of the non-zero 
        # voxels of each volume need to be scaled.
        if volume_1.shape != volume_2.shape:
            positions_1 = (
                positions_1 * np.array(volume_2.shape) 
                / np.array(volume_1.shape)
            )
            positions_1 = positions_1.astype(int)

        # Check that the non-zero voxels of the volumes are at least 
        # min_distance apart.
        return np.all(
            cdist(positions_1, positions_2) > min_distance
        )

    def _resample_volume_position(
        self: NonOverlapping,
        volume: np.ndarray | Image,
    ) -> Image:
        """Resamples the position of a 3D volume using its internal position 
        sampler.

        This method updates the `position` property of the given `volume` by 
        drawing a new position from the `_position_sampler` stored in the 
        volume's `properties`. If the sampled position is a `Quantity`, it is 
        converted to pixel units.

        Parameters
        ----------
        volume: np.ndarray or Image
            The 3D volume whose position is to be resampled. The volume must 
            have a `properties` attribute containing dictionaries with 
            `position` and `_position_sampler` keys.

        Returns
        -------
        Image
            The same input volume with its `position` property updated to the 
            newly sampled value.

        Notes
        -----
        - The `_position_sampler` function is expected to return a **tuple of 
        three floats** (e.g., `(x, y, z)`).
        - If the sampled position is a `Quantity`, it is converted to pixels.
        - **Only** dictionaries in `volume.properties` that contain both 
        `position` and `_position_sampler` keys are modified.
        
        """

        for pdict in volume.properties:
            if "position" in pdict and "_position_sampler" in pdict:
                new_position = pdict["_position_sampler"]()
                if isinstance(new_position, Quantity):
                    new_position = new_position.to("pixel").magnitude
                pdict["position"] = new_position

        return volume


class Store(Feature):
    """Store the output of a feature for reuse.

    The `Store` feature evaluates a given feature and stores its output in an 
    internal dictionary. Subsequent calls with the same key will return the 
    stored value unless the `replace` parameter is set to `True`. This enables 
    caching and reuse of computed feature outputs.

    Parameters
    ----------
    feature: Feature
        The feature to evaluate and store.
    key: Any
        The key used to identify the stored output.
    replace: bool, optional
        If `True`, replaces the stored value with a new computation. It defaults 
        to `False`.
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        Always `False` for `Store`, as it handles caching locally.
    _store: dict[Any, Image]
        A dictionary used to store the outputs of the evaluated feature.

    Methods
    -------
    `get(_: Any, key: Any, replace: bool, **kwargs: dict[str, Any]) -> Any`
        Evaluate and store the feature output, or return the cached result.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    >>> value_feature = dt.Value(lambda: np.random.rand())

    Create a `Store` feature with a key:
    >>> store_feature = dt.Store(feature=value_feature, key="example")

    Retrieve and store the value:
    >>> output = store_feature(None, key="example", replace=False)

    Retrieve the stored value without recomputing:
    >>> value_feature.update()
    >>> cached_output = store_feature(None, key="example", replace=False)
    >>> print(cached_output == output)
    True

    Retrieve the stored value recomputing:
    >>> value_feature.update()
    >>> cached_output = store_feature(None, key="example", replace=True)
    >>> print(cached_output == output)
    False

    """

    __distributed__: bool = False

    def __init__(
        self: Store,
        feature: Feature,
        key: Any,
        replace: bool = False,
        **kwargs: Any,
    ):
        """Initialize the Store feature.

        Parameters
        ----------
        feature: Feature
            The feature to evaluate and store.
        key: Any
            The key used to identify the stored output.
        replace: bool, optional
            If `True`, replaces the stored value with a new computation. 
            It defaults to `False`.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(key=key, replace=replace, **kwargs)
        self.feature = self.add_feature(feature, **kwargs)
        self._store: dict[Any, Image] = {}

    def get(
        self: Store,
        _: Any,
        key: Any,
        replace: bool,
        **kwargs: Any,
    ) -> Any:
        """Evaluate and store the feature output, or return the cached result.

        Parameters
        ----------
        _: Any
            Placeholder for unused image input.
        key: Any
            The key used to identify the stored output.
        replace: bool
            If `True`, replaces the stored value with a new computation.
        **kwargs: Any
            Additional keyword arguments passed to the feature.

        Returns
        -------
        Any
            The stored output or a newly computed result.

        """

        # Check if the value should be recomputed or retrieved from the store
        if replace or not (key in self._store):
            self._store[key] = self.feature()

        # Return the stored or newly computed result
        if self._wrap_array_with_image:
            return Image(self._store[key], copy=False)
        else:
            return self._store[key]


class Squeeze(Feature):
    """Squeeze the input image to the smallest possible dimension.

    This feature removes axes of size 1 from the input image. By default, it 
    removes all singleton dimensions. If a specific axis or axes are specified, 
    only those axes are squeezed.

    Parameters
    ----------
    axis: int or tuple[int, ...], optional
        The axis or axes to squeeze. It defaults to `None`, squeezing all axes.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: array, axis: int | tuple[int, ...], **kwargs: Any) -> array`
        Squeeze the input image by removing singleton dimensions. The input and
        output arrays can be a NumPy array, a PyTorch tensor, or an Image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input array with extra dimensions:
    >>> import numpy as np
    >>>
    >>> input_image = np.array([[[[1], [2], [3]]]])
    >>> input_image.shape
    (1, 1, 3, 1)

    Create a Squeeze feature:
    >>> squeeze_feature = dt.Squeeze(axis=0)
    >>> output_image = squeeze_feature(input_image)
    >>> output_image.shape
    (1, 3, 1)

    Without specifying an axis:
    >>> squeeze_feature = dt.Squeeze()
    >>> output_image = squeeze_feature(input_image)
    >>> output_image.shape
    (3,)

    """

    def __init__(
        self: Squeeze,
        axis: int | tuple[int, ...] | None = None,
        **kwargs: Any,
    ):
        """Initialize the Squeeze feature.

        Parameters
        ----------
        axis: int or tuple[int, ...], optional
            The axis or axes to squeeze. It defaults to `None`, which squeezes 
            all singleton axes.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Squeeze,
        image: NDArray | torch.Tensor | Image,
        axis: int | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> NDArray | torch.Tensor | Image:
        """Squeeze the input image by removing singleton dimensions.

        Parameters
        ----------
        image: array
            The input image to process. The input array can be a NumPy array, a
            PyTorch tensor, or an Image.
        axis: int or tuple[int, ...], optional
            The axis or axes to squeeze. It defaults to `None`, which squeezes 
            all singleton axes.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array
            The squeezed image with reduced dimensions. The output array can be
            a NumPy array, a PyTorch tensor, or an Image.

        """

        if apc.is_torch_array(image):
            if axis is None:
                return image.squeeze()
            if isinstance(axis, int):
                return image.squeeze(axis)
            for ax in sorted(axis, reverse=True):
                image = image.squeeze(ax)
            return image

        return xp.squeeze(image, axis=axis)


class Unsqueeze(Feature):
    """Unsqueeze the input image to the smallest possible dimension.

    This feature adds new singleton dimensions to the input image at the 
    specified axis or axes. If no axis is specified, it defaults to adding 
    a singleton dimension at the last axis.

    Parameters
    ----------
    axis: int or tuple[int, ...], optional
        The axis or axes where new singleton dimensions should be added. It
        defaults to `None`, which adds a singleton dimension at the last axis.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: array, axis: int | tuple[int, ...] | None, **kwargs: Any) -> array`
        Add singleton dimensions to the input image. The input and output
        arrays can be a NumPy array, a PyTorch tensor, or an Image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input array:
    >>> import numpy as np
    >>>
    >>> input_image = np.array([1, 2, 3])
    >>> input_image.shape
    (3,)

    Apply Unsqueeze feature:
    >>> unsqueeze_feature = dt.Unsqueeze(axis=0)
    >>> output_image = unsqueeze_feature(input_image)
    >>> output_image.shape
    (1, 3)

    Without specifying an axis, in unsqueezes the last dimension:
    >>> unsqueeze_feature = dt.Unsqueeze()
    >>> output_image = unsqueeze_feature(input_image)
    >>> output_image.shape
    (3, 1)

    """

    def __init__(
        self: Unsqueeze,
        axis: int | tuple[int, ...] | None = -1,
        **kwargs: Any,
    ):
        """Initialize the Unsqueeze feature.

        Parameters
        ----------
        axis: int or tuple[int, ...], optional
            The axis or axes where new singleton dimensions should be added. It
            defaults to -1, which adds a singleton dimension at the last axis.
        **kwargs:: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Unsqueeze,
        image: np.ndarray | torch.Tensor | Image,
        axis: int | tuple[int, ...] | None = -1,
        **kwargs: Any,

    ) -> np.ndarray | torch.Tensor | Image:
        """Add singleton dimensions to the input image.

        Parameters
        ----------
        image: array
            The input image to process. The input array can be a NumPy array, a
            PyTorch tensor, or an Image.
        axis: int or tuple[int, ...], optional
            The axis or axes where new singleton dimensions should be added. 
            It defaults to -1, which adds a singleton dimension at the last
            axis.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array
            The input image with the specified singleton dimensions added. The
            output array can be a NumPy array, a PyTorch tensor, or an Image.

        """

        if apc.is_torch_array(image):
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                image = image.unsqueeze(ax)
            return image

        return xp.expand_dims(image, axis=axis)


ExpandDims = Unsqueeze


class MoveAxis(Feature):
    """Moves the axis of the input image.

    This feature rearranges the axes of an input image, moving a specified 
    source axis to a new destination position. All other axes remain in their 
    original order.

    Parameters
    ----------
    source: int
        The source position of the axis to move.
    destination: int
        The destination position of the axis.
    **kwargs:: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: array, source: int, destination: int, **kwargs: Any) -> array`
        Move the specified axis of the input image to a new position. The input
        and output array can be a NumPy array, a PyTorch tensor, or an Image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input array:
    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(2, 3, 4)
    >>> input_image.shape
    (2, 3, 4)

    Apply a MoveAxis feature:
    >>> move_axis_feature = dt.MoveAxis(source=0, destination=2)
    >>> output_image = move_axis_feature(input_image)
    >>> output_image.shape
    (3, 4, 2)

    """

    def __init__(
        self: MoveAxis,
        source: int,
        destination: int,
        **kwargs: Any,
    ):
        """Initialize the MoveAxis feature.

        Parameters
        ----------
        source: int
            The axis to move.
        destination: int
            The destination position of the axis.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(source=source, destination=destination, **kwargs)

    def get(
        self: MoveAxis,
        image: NDArray | torch.Tensor | Image,
        source: int,
        destination: int,
        **kwargs: Any,
    ) -> NDArray | torch.Tensor | Image:
        """Move the specified axis of the input image to a new position.

        Parameters
        ----------
        image: array
            The input image to process. The input array can be a NumPy array, a
            PyTorch tensor, or an Image.
        source: int
            The axis to move.
        destination: int
            The destination position of the axis.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array
            The input image with the specified axis moved to the destination.
            The output array can be a NumPy array, a PyTorch tensor, or an
            Image.

        """

        if apc.is_torch_array(image):
            axes = list(range(image.ndim))
            axis = axes.pop(source)
            axes.insert(destination, axis)
            return image.permute(*axes)

        return xp.moveaxis(image, source, destination)


class Transpose(Feature):
    """Transpose the input image.

    This feature rearranges the axes of an input image according to the 
    specified order. The `axes` parameter determines the new order of the 
    dimensions.

    Parameters
    ----------
    axes: tuple[int, ...], optional
        A tuple specifying the permutation of the axes. If `None`, the axes are 
        reversed by default.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: array, axes: tuple[int, ...] | None, **kwargs: Any) -> array`
        Transpose the axes of the input image(s). The input and output array
        can be a NumPy array, a PyTorch tensor, or an Image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input array:
    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(2, 3, 4)
    >>> input_image.shape
    (2, 3, 4)

    Apply a Transpose feature:
    >>> transpose_feature = dt.Transpose(axes=(1, 2, 0))
    >>> output_image = transpose_feature(input_image)
    >>> output_image.shape
    (3, 4, 2)

    Without specifying axes:
    >>> transpose_feature = dt.Transpose()
    >>> output_image = transpose_feature(input_image)
    >>> output_image.shape
    (4, 3, 2)

    """

    def __init__(
        self: Transpose,
        axes: tuple[int, ...] | None = None,
        **kwargs: Any,
    ):
        """Initialize the Transpose feature.

        Parameters
        ----------
        axes: tuple[int, ...], optional
            A tuple specifying the permutation of the axes. If `None`, the 
            axes are reversed by default.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.
        
        """

        super().__init__(axes=axes, **kwargs)

    def get(
        self: Transpose,
        image: NDArray | torch.Tensor | Image,
        axes: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> NDArray | torch.Tensor | Image:
        """Transpose the axes of the input image.

        Parameters
        ----------
        image: array
            The input image to process. The input array can be a NumPy array, a
            PyTorch tensor, or an Image.
        axes: tuple[int, ...], optional
            A tuple specifying the permutation of the axes. If `None`, the 
            axes are reversed by default.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array
            The transposed image with rearranged axes. The output array can be
            a NumPy array, a PyTorch tensor, or an Image.

        """

        return xp.transpose(image, axes)


Permute = Transpose


class OneHot(Feature):
    """Convert the input to a one-hot encoded array.

    This feature takes an input array of integer class labels and converts it 
    into a one-hot encoded array. The last dimension of the input is replaced 
    by the one-hot encoding.

    Parameters
    ----------
    num_classes: int
        The total number of classes for the one-hot encoding.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: array, num_classes: int, **kwargs: Any) -> array`
        Convert the input array of class labels into a one-hot encoded array.
        The input and output arrays can be a NumPy array, a PyTorch tensor, or
        an Image.

    Examples
    --------
    >>> import deeptrack as dt
    
    Create an input array of class labels:
    >>> import numpy as np
    >>>
    >>> input_data = np.array([0, 1, 2])

    Apply a OneHot feature:
    >>> one_hot_feature = dt.OneHot(num_classes=3)
    >>> one_hot_encoded = one_hot_feature.get(input_data, num_classes=3)
    >>> one_hot_encoded
    array([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])

    """

    def __init__(
        self: OneHot,
        num_classes: int,
        **kwargs: Any,
    ):
        """Initialize the OneHot feature.

        Parameters
        ----------
        num_classes: int
            The total number of classes for the one-hot encoding.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(num_classes=num_classes, **kwargs)

    def get(
        self: OneHot,
        image: NDArray | torch.Tensor | Image,
        num_classes: int,
        **kwargs: Any,
    ) -> NDArray | torch.Tensor | Image:
        """Convert the input array of labels into a one-hot encoded array.

        Parameters
        ----------
        image: array
            The input array of class labels. The last dimension should contain 
            integers representing class indices. The input array can be a NumPy
            array, a PyTorch tensor, or an Image.
        num_classes: int
            The total number of classes for the one-hot encoding.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array
            The one-hot encoded array. The last dimension is replaced with 
            one-hot vectors of length `num_classes`. The output array can be a
            NumPy array, a PyTorch tensor, or an Image. In all cases, it is of
            data type float32 (e.g., np.float32 or torch.float32).

        """

        # Flatten the last dimension if it's singleton.
        if image.shape[-1] == 1:
            image = image[..., 0]

        if apc.is_torch_array(image):
            return (torch.nn.functional
                    .one_hot(image, num_classes=num_classes)
                    .to(dtype=torch.float32))

        # Create the one-hot encoded array.
        return xp.eye(num_classes, dtype=np.float32)[image]


class TakeProperties(Feature):
    """Extract all instances of a set of properties from a pipeline.

    Only extracts the properties if the feature contains all given 
    property-names. The order of the properties is not guaranteed to be the 
    same as the evaluation order.

    If there is only a single property name, this will return a list of the 
    property values.

    Parameters
    ----------
    feature: Feature
        The feature from which to extract properties.
    names: list[str]
        The names of the properties to extract
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.
        Always `False` for `TakeProperties`, as it processes sequentially.
    __list_merge_strategy__: int
        Specifies how lists of properties are merged. Set to 
        `MERGE_STRATEGY_APPEND` to append values to the result list.

    Methods
    -------
    `get(image: Any, names: tuple[str, ...], **kwargs: dict[str, Any]) -> np.ndarray | tuple[np.ndarray, ...]`
        Extract the specified properties from the feature pipeline.
    
    Examples
    --------
    >>> import deeptrack as dt
    
    >>> class ExampleFeature(Feature):
    ...     def __init__(self, my_property, **kwargs):
    ...         super().__init__(my_property=my_property, **kwargs)

    Create an example feature with a property:
    >>> feature = ExampleFeature(my_property=Property(42))

    Use `TakeProperties` to extract the property:
    >>> take_properties = dt.TakeProperties(feature)
    >>> output = take_properties.get(image=None, names=["my_property"])
    >>> print(output)
    [42]

    Create a `Gaussian` feature:
    >>> noise_feature = dt.Gaussian(mu=7, sigma=12)
    
    Use `TakeProperties` to extract the property:
    >>> take_properties = dt.TakeProperties(noise_feature)
    >>> output = take_properties.get(image=None, names=["mu"])
    >>> print(output)
    [7]

    """

    __distributed__: bool = False
    __list_merge_strategy__: int = MERGE_STRATEGY_APPEND

    def __init__(
        self: TakeProperties,
        feature: Feature,
        *names: str,
        **kwargs: Any,
    ):
        """Initialize the TakeProperties feature.

        Parameters
        ----------
        feature: Feature
            The feature from which to extract properties.
        *names: str
            One or more names of the properties to extract.
=        **kwargs: Any, optional
            Additional keyword arguments passed to the parent `Feature` class.
        
        """

        super().__init__(names=names, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: TakeProperties,
        image: Any,
        names: tuple[str, ...],
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """Extract the specified properties from the feature pipeline.

        This method retrieves the values of the specified properties from the 
        feature's dependency graph and returns them as NumPy arrays.

        Parameters
        ----------
        image: Any
            The input image (unused in this method).
        names: tuple[str, ...]
            The names of the properties to extract.
        _ID: tuple[int, ...], optional
            A unique identifier for the current computation, ensuring that 
            dependencies are correctly matched. It defaults to an empty tuple.
        **kwargs: Any, optional
            Additional keyword arguments (unused in this method).

        Returns
        -------
        np.ndarray or tuple[np.ndarray, ...]
            If a single property name is provided, a NumPy array containing the 
            property values is returned. If multiple property names are 
            provided, a tuple of NumPy arrays is returned, where each array 
            corresponds to a property.

        """

        # Ensure the feature is valid for the given _ID.
        if not self.feature.is_valid(_ID=_ID):
            self.feature(_ID=_ID)

        # Initialize a dictionary to store property values.
        res = {}
        for name in names:
            res[name] = []

        # Traverse the dependencies of the feature.
        for dep in self.feature.recurse_dependencies():
            # Check if the dependency contains all required property names.
            if (isinstance(dep, PropertyDict) 
                and all(name in dep for name in names)):
                for name in names:
                    # Extract property values that match the current _ID.
                    data = dep[name].data.dict
                    for key, value in data.items():
                        if key[:len(_ID)] == _ID:
                            res[name].append(value.current_value())

        # Convert the results to NumPy arrays.
        res = tuple([np.array(res[name]) for name in names])

        # Return a single array if only one property name is specified.
        if len(res) == 1:
            res = res[0]

        return res
