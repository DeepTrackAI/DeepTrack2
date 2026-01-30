"""Core features for building and processing pipelines in DeepTrack2.  # TODO

This module defines the core classes and utilities used to create and
manipulate features in DeepTrack2, enabling users to build sophisticated data
processing pipelines with modular, reusable, and composable components.

Key Features
------------
- **Features**

    A `Feature` is a building block of a data processing pipeline.
    It represents a transformation applied to data, such as image manipulation,
    data augmentation, or computational operations. Features are highly
    customizable and can be combined into pipelines for complex workflows.

- **Structural Features**

    Structural features extend the basic `StructuralFeature` class by adding
    hierarchical or logical structures, such as chains, branches, or
    probabilistic choices. They enable the construction of pipelines with
    advanced data flow requirements.

- **Feature Properties**

    Features can have dynamically sampled properties, enabling parameterization
    of transformations. These properties are defined at initialization and can
    be updated during pipeline execution.

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

    In general, a feature represents a modular data transformation with
    properties and methods for customization.

- `StructuralFeature`: Base class for features providing structure.

    Base class for specialized features for organizing and managing
    hierarchical or logical structures in the pipeline without input
    transformations.

- `ArithmeticOperationFeature`: Apply arithmetic operation element-wise.

    Base class for features performing arithmetic operations like addition,
    subtraction, multiplication, and division.

Structural Feature Classes:
- `Chain`: Sequentially apply multiple features to the input data (>>).
- `Branch`: Alias of `Chain`.
- `Probability`: Resolve a feature with a certain probability.
- `Repeat`: Apply a feature multiple times in sequence (^).
- `Combine`: Combine multiple features into a single feature.
- `Bind`: Bind a feature with property arguments.
- `BindResolve`: DEPRECATED Alias of `Bind`.
- `BindUpdate`: DEPRECATED Bind a feature with certain arguments.
- `ConditionalSetProperty`: DEPRECATED Conditionally override child properties.
- `ConditionalSetFeature`: DEPRECATED Conditionally resolve features.

Other Feature Classes:
- `DummyFeature`: A no-op feature that simply returns the input unchanged.
- `Value`: Store a constant value as a feature.
- `Stack`: Stack the input and the value.
- `Arguments`: A convenience container for pipeline arguments.
- `Slice`: Dynamically apply array indexing to inputs.
- `Lambda`: Apply a user-defined function to the input.
- `Merge`: Apply a custom function to a list of inputs.
- `OneOf`: Resolve one feature from a given collection.
- `OneOfDict`: Resolve one feature from a dictionary and apply it to an input.
- `LoadImage`: Load an image from disk and preprocess it.
- `AsType`: Convert the data type of the input.
- `ChannelFirst2d`: DEPRECATED Convert an image to a channel-first format.
- `Store`: Store the output of a feature for reuse.
- `Squeeze`: Squeeze the input to the smallest possible dimension.
- `Unsqueeze`: Unsqueeze the input.
- `ExpandDims`: Alias of `Unsqueeze`.
- `MoveAxis`: Move the axis of the input.
- `Transpose`: Transpose the input.
- `Permute`: Alias of `Transpose`.
- `OneHot`: Convert the input to a one-hot encoded array.
- `TakeProperties`: Extract all instances of properties from a pipeline.

Arithmetic Feature Classes:
- `Add`: Add a value to the input.@dataclass
- `Subtract`: Subtract a value from the input.
- `Multiply`: Multiply the input by a value.
- `Divide`: Divide the input by a value.
- `FloorDivide`: Divide the input by a value.
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

- `propagate_data_to_dependencies(feature, _ID, **kwargs) -> None`

    Propagates data to all dependencies of a feature, updating their properties
    with the provided values.

Examples
--------
Define a simple pipeline with features.

>>> import deeptrack as dt

Create a basic addition feature:

>>> class BasicAdd(dt.Feature):
...     def get(self, data, value, **kwargs):
...         return data + value

Create two features:

>>> add_five = BasicAdd(value=5)
>>> add_ten = BasicAdd(value=10)

Chain features together:

>>> pipeline = dt.Chain(add_five, add_ten)

Or equivalently:

>>> pipeline = add_five >> add_ten

Process an input array:

>>> import numpy as np
>>>
>>> input = np.array([[1, 2, 3], [4, 5, 6]])
>>> output = pipeline(input)
>>> output
array([[16, 17, 18],
       [19, 20, 21]])

"""


from __future__ import annotations

import itertools
import operator
import random
import warnings
from typing import Any, Callable, Iterable, Literal, TYPE_CHECKING

import array_api_compat as apc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pint import Quantity

from deeptrack.backend import config, TORCH_AVAILABLE, xp
from deeptrack.backend.core import DeepTrackNode
from deeptrack.backend.units import ConversionTable
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
    "LoadImage",
    "AsType",
    "ChannelFirst2d",
    "Store",
    "Squeeze",
    "Unsqueeze",
    "ExpandDims",
    "MoveAxis",
    "Transpose",
    "Permute",
    "OneHot",
    "TakeProperties",
]


if TYPE_CHECKING:
    import torch


# Return the newly generated outputs, discarding the existing list of inputs.
MERGE_STRATEGY_OVERRIDE: int = 0

# Append newly generated outputs to the existing list of inputs.
MERGE_STRATEGY_APPEND: int = 1


class Feature(DeepTrackNode):
    """Base feature class.  # TODO

    Features define the data generation and transformation process.
    
    All features operate on lists of data, often lists of images. Most
    features, such as noise, apply a tranformation to all data in the list.
    The transformation can be additive, such as adding some Gaussian noise or a
    background illumination to images, or non-additive, such as introducing
    Poisson noise or performing a low-pass filter. The transformation is
    defined by the `.get(data, **kwargs)` method, which all implementations of
    the `Feature` class need to define. This method operates on a single data
    at a time.

    Whenever a feature is initialized, it wraps all keyword arguments passed to
    the constructor as `Property` objects, and stores them in the `.properties`
    attribute as a `PropertyDict`.
    
    When a feature is resolved, the current value of each property is sent as
    input to the `.get()` method.

    **Computational Backends and Data Types**
    
    The `Feature` class also provides mechanisms for managing numerical types
    and computational backends.

    Supported backends include NumPy and PyTorch. The active backend is
    determined at initialization and stored in the `._backend` attribute, which
    is used internally to control how computations are executed. The backend
    can be switched using the `.numpy()` and `.torch()` methods.

    Numerical types used in computation (float, int, complex, and bool) can be
    configured using the `.dtype()` method. The chosen types are retrieved
    via the properties `.float_dtype`, `.int_dtype`, `.complex_dtype`, and
    `.bool_dtype`. These are resolved dynamically using the backend's internal
    type resolution system and are used in downstream computations.

    The computational device (e.g., "cpu" or a specific GPU) is managed through
    the `.to()` method and accessed via the `.device` property. This is
    especially relevant for PyTorch backends, which support GPU acceleration.

    Parameters
    ----------
    data: Any, optional
        The input data for the feature. If left empty, no initial input is set.
        It is most commonly a NumPy array, a PyTorch tensor, or a list of NumPy
        arrays or PyTorch tensors; however, it can be anything.
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
        dynamically sampled values during pipeline execution. A sampled copy of
        this dictionary is passed to the `.get()` function and appended to the
        properties of the output.
    _input: DeepTrackNode
        A node representing the input data for the feature. It is most commonly
        a NumPy array, PyTorch tensor, or a list of NumPy arrays or PyTorch
        tensors; however, it can be anything.
        It supports lazy evaluation and graph traversal.
    _random_seed: DeepTrackNode
        A node representing the feature’s random seed. This allows for
        deterministic behavior when generating random elements, and ensures
        reproducibility during evaluation.
    arguments: Feature or None
        An optional feature whose properties are bound to this feature. This
        allows dynamic property sharing and centralized parameter management
        in complex pipelines.
    __list_merge_strategy__: int
        Specifies how the output of `.get(data, **kwargs)` is merged with the
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
    _backend: "numpy" or "torch"
        The computational backend.

    Methods
    -------
    `get(data, **kwargs) -> Any`
        Abstract method that defines how the feature transforms the input data.
        The input is most commonly a NumPy array or a PyTorch tensor, but it
        can be anything.
    `__call__(data_list, _ID, **kwargs) -> Any`
        Executes the feature or pipeline on the input and applies property
        overrides from `kwargs`.
    `resolve(data_list, _ID, **kwargs) -> Any`
        Alias of `__call__()`.
    `to_sequential(**kwargs) -> Feature`
        Converts a feature to be resolved as a sequence.
    `torch(device, recursive) -> Feature`
        Sets the backend to PyTorch.
    `numpy(recursice) -> Feature`
        Sets the backend to NumPy.
    `get_backend() -> "numpy" or "torch"`
        Returns the current backend of the feature.
    `dtype(float, int, complex, bool) -> Feature`
        Sets the dtype to be used during evaluation.
    `to(device) -> Feature`
        Sets the device to be used during evaluation.
    `batch(batch_size) -> tuple`
        Batches the feature for repeated execution.
    `action(_ID) -> Any or list[Any]`
        Implements the core logic to create or transform the input(s).
    `update(**global_arguments) -> Feature`
        Refreshes the feature to create a new output.
    `add_feature(feature) -> Feature`
        Adds a feature to the dependency graph of this one.
    `seed(updated_seed, _ID) -> int`
        Sets the random seed for the feature, ensuring deterministic behavior.
    `bind_arguments(arguments) -> Feature`
        Binds another feature’s properties as arguments to this feature.
    `plot(input_image, resolve_kwargs, interval, **kwargs) -> Any`
        Visualizes the output of the feature when it is an image.

    **Private and internal methods.**
    `_normalize(**properties) -> dict[str, Any]`
        Normalizes the properties of the feature.
    `_process_properties(propertydict) -> dict[str, Any]`
        Preprocesses the input properties before calling the `get` method.
    `_activate_sources(x) -> None`
        Activates sources in the input data.
    `__getattr__(key) -> Any`
        Provides custom attribute access for the `Feature` class.
    `__iter__() -> Feature`
        Returns an iterator for the feature.
    `__next__() -> Any`
        Return the next element iterating over the feature.
    `__rshift__(other) -> Feature`
        Allows chaining of features.
    `__rrshift__(other) -> Feature`
        Allows right chaining of features.
    `__add__(other) -> Feature`
        Overrides add operator.
    `__radd__(other) -> Feature`
        Overrides right add operator.
    `__sub__(other) -> Feature`
        Overrides subtraction operator.
    `__rsub__(other) -> Feature`
        Overrides right subtraction operator.
    `__mul__(other) -> Feature`
        Overrides multiplication operator.
    `__rmul__(other) -> Feature`
        Overrides right multiplication operator.
    `__truediv__(other) -> Feature`
        Overrides division operator.
    `__rtruediv__(other) -> Feature`
        Overrides right division operator.
    `__floordiv__(other) -> Feature`
        Overrides floor division operator.
    `__rfloordiv__(other) -> Feature`
        Overrides right floor division operator.
    `__pow__(other) -> Feature`
        Overrides power operator.
    `__rpow__(other) -> Feature`
        Overrides right power operator.
    `__gt__(other) -> Feature`
        Overrides greater than operator.
    `__rgt__(other) -> Feature`
        Overrides right greater than operator.
    `__lt__(other) -> Feature`
        Overrides less than operator.
    `__rlt__(other) -> Feature`
        Overrides right less than operator.
    `__le__(other) -> Feature`
        Overrides less than or equal to operator.
    `__rle__(other) -> Feature`
        Overrides right less than or equal to operator.
    `__ge__(other) -> Feature`
        Overrides greater than or equal to operator.
    `__rge__(other) -> Feature`
        Overrides right greater than or equal to operator.
    `__xor__(other) -> Feature`
        Overrides XOR operator.
    `__and__(other) -> Feature`
        Overrides and operator.
    `__rand__(other) -> Feature`
        Overrides right and operator.
    `__getitem__(key) -> Feature`
        Allows direct slicing of the data.
    `_format_input(data_list, **kwargs) -> list[Any]`
        Formats the input data for the feature.
    `_process_and_get(data_list, **kwargs) -> list[Any]`
        Calls the `.get()` method according to the `__distributed__` attribute.

    Examples
    --------
    >>> import deeptrack as dt

    **Define and evaluate a simple feature**

    >>> import numpy as np
    >>>
    >>> feature = dt.Value(np.array([1, 2, 3]))
    >>> result = feature()
    >>> result
    array([1, 2, 3])

    **Chain features using '>>'**

    >>> pipeline = dt.Value(np.array([1, 2, 3])) >> dt.Add(2)
    >>> pipeline()
    array([3, 4, 5])

    **Use arithmetic operators**

    >>> feature = dt.Value(np.array([1, 2, 3]))
    >>> result = (feature + 1) * 2 - 1
    >>> result()
    array([3, 5, 7])

    This is equivalent to chaining with `Add`, `Multiply`, and `Subtract`.

    **Evaluate a dynamic feature using `.update()` or `.new()`**

    >>> feature = dt.Value(lambda: np.random.rand())
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

    >>> output4 = feature.new()  # Combine update and resolve
    >>> output4
    0.28477040978587476

    **Generate a batch of outputs**

    >>> feature = dt.Value(lambda: np.random.rand()) + 1
    >>> batch = feature.batch(batch_size=3)
    >>> batch
    (array([1.6888222 , 1.88422131, 1.90027316]),)

    **Switch computational backend to torch**

    >>> import torch
    >>>
    >>> feature = dt.Add(b=5).torch()
    >>> input_tensor = torch.tensor([1.0, 2.0])
    >>> feature(input_tensor)
    tensor([6., 7.])

    **Use `.seed()` for reproducibility**

    >>> feature = dt.Value(lambda: np.random.randint(0, 100))
    >>> seed = feature.seed()
    >>> v1 = feature.new()
    >>> v1
    76

    >>> feature.seed(seed)
    >>> v2 = feature.new()
    >>> v2
    76

    **Sequential feature with evolving property**

    >>> def rotate(sequence_length, previous_value):
    ...     return previous_value + 2 * np.pi / sequence_length

    >>> rotating = dt.Ellipse(
    ...     position=(16, 16),
    ...     radius=(1.5e-6, 1e-6),
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
    ...         value=lambda freq: np.linspace(0, 2 * np.pi * freq, 100),
    ...         freq=arguments.frequency,
    ...     )
    ...     >> np.sin
    ...     >> dt.Multiply(
    ...         b=lambda amp: amp,
    ...         amp=arguments.amplitude,
    ...     )
    ... )
    >>> wave.bind_arguments(arguments)

    >>> from matplotlib import pyplot as plt
    >>>
    >>> plt.plot(wave())
    >>> plt.show()

    >>> plt.plot(wave(frequency=2, amplitude=1))
    >>> plt.show()

    """

    properties: PropertyDict
    _input: DeepTrackNode
    _random_seed: DeepTrackNode
    arguments: Feature | None

    __list_merge_strategy__: int = MERGE_STRATEGY_OVERRIDE
    __distributed__: bool = True
    __conversion_table__: ConversionTable = ConversionTable()

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

    def __init__(  # TODO
        self: Feature,
        _input: Any | None = None,
        **kwargs: Any,
    ):
        """Initialize a new Feature instance.

        This constructor sets up the feature as a `DeepTrackNode` whose
        executable logic is defined by the `_action()` method. All keyword
        arguments are wrapped as `Property` objects and stored in a
        `PropertyDict`, enabling dynamic sampling and dependency tracking
        during evaluation.

        The input is wrapped internally as a `DeepTrackNode`, allowing it to
        participate in lazy evaluation, caching, and graph traversal.


        Initialization proceeds in the following order:
        1. Backend, dtypes, and device are set from the global configuration.
        2. The feature is registered as a `DeepTrackNode` with `_action` as its
        executable logic.
        3. Properties are wrapped into a `PropertyDict` and attached as
        dependencies.
        4. The input is wrapped as a `DeepTrackNode`.
        5. A random seed node is created for reproducible stochastic behavior.

        This ordering is required to ensure correct dependency tracking and
        evaluation behavior.

        Parameters
        ----------
        _input: Any, optional
            The initial input(s) for the feature. Commonly a NumPy array, a
            PyTorch tensor, or a list of such objects, but may be any value.
            If `None`, the input defaults to an empty list.
        **kwargs: Any
            Keyword arguments used to configure the feature. Each keyword
            argument is wrapped as a `Property` and added to the feature's
            `properties` attribute. These properties are resolved dynamically
            at call time and passed to the `.get()` method.

        """

        if _input is None:
            _input = []

        # Store backend, dtypes and device on initialization.
        self._backend = config.get_backend()
        self._float_dtype = "default"
        self._int_dtype = "default"
        self._complex_dtype = "default"
        self._bool_dtype = "default"
        self._device = config.get_device()

        # Pass Feature core logic to DeepTrackNode as its action with _ID.
        # NOTE: _action must be registered before adding dependencies.
        super().__init__(action=self._action)

        # Ensure the feature has a 'name' property; default = class name.
        self.node_name = kwargs.setdefault("name", type(self).__name__)

        # Create a PropertyDict to hold the feature’s properties.
        self.properties = PropertyDict(node_name="properties", **kwargs)
        self.properties.add_child(self)

        # Initialize the input as a DeepTrackNode.
        self._input = DeepTrackNode(node_name="_input", action=_input)
        self._input.add_child(self)

        # Random seed node (for deterministic behavior if desired).
        self._random_seed = DeepTrackNode(
            node_name="_random_seed",
            action=lambda: random.randint(0, 2147483648),
        )
        self._random_seed.add_child(self)

        # Initialize arguments to None.
        self.arguments = None

    def get(
        self: Feature,
        data: Any,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Transform input data (abstract method).

        Abstract method that defines how the feature transforms the input data.
        The current values of all properties are passed as keyword arguments.

        Parameters
        ----------
        data: Any
            The input data to be transformed, most commonly a NumPy array or a
            PyTorch tensor, but it can be anything.
        _ID: tuple[int], optional
            The unique identifier for the current execution. Defaults to ().
        **kwargs: Any
            The current value of all properties in the `properties` attribute,
            as well as any global arguments passed to the feature.

        Returns
        -------
        Any
            The transformed data.

        Raises
        ------
        NotImplementedError
            Raised if this method is not overridden by subclasses.

        """

        raise NotImplementedError

    def __call__(  # TODO
        self: Feature,
        data_list: Any = None,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Execute the feature or pipeline.

        The `.__call__()` method executes the feature or pipeline on the
        provided input data and updates the computation graph if necessary.
        It overrides properties using the keyword arguments.

        The actual computation is performed by calling the parent `.__call__()`
        method in the `DeepTrackNode` class, which manages lazy evaluation and
        caching.

        Parameters
        ----------
        data_list: Any, optional
            The input data to the feature or pipeline. It is most commonly a
            list of NumPy arrays or PyTorch tensors, but it can be anything.
            Defaults to `None`, in which case the feature uses the previous set
            of input values or propagates properties.
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
            typically a list of NumPy arrays or PyTorch tensors, but it can be
            anything.

        Examples
        --------
        >>> import deeptrack as dt

        Define a feature:
    
        >>> feature = dt.Add(b=2)

        Call this feature with an input:
    
        >>> import numpy as np
        >>>
        >>> feature(np.array([1, 2, 3]))
        array([3, 4, 5])

        Execute the feature with previously set input:
    
        >>> feature()  # Uses stored input
        array([3, 4, 5])

        Execute the feature with new input:
    
        >>> feature(np.array([10, 20, 30]))  # Uses new input
        array([12, 22, 32])

        Override a property:
    
        >>> feature(np.array([10, 20, 30]), b=1)
        array([11, 21, 31])

        """

        with config.with_backend(self._backend):
            # If data_list is a Source, activate it.
            self._activate_sources(data_list)

            # Potentially fragile.
            # Maybe a special variable dt._last_input instead?
            # If the input is not empty, set the value of the input.
            if (
                data_list is not None
                and not (isinstance(data_list, list) and len(data_list) == 0)
                and not (isinstance(data_list, tuple)
                        and any(isinstance(x, SourceItem) for x in data_list))
            ):
                self._input.set_value(data_list, _ID=_ID)

            # A dict to store values of self.arguments before updating them.
            original_values = {}

            # If there are no self.arguments, instead propagate the values of
            # the kwargs to all properties in the computation graph.
            if kwargs and self.arguments is None:
                propagate_data_to_dependencies(self, _ID=_ID, **kwargs)

            # If there are self.arguments, update the values of self.arguments
            # to match kwargs.
            if isinstance(self.arguments, Feature):
                for key, value in kwargs.items():
                    if key in self.arguments.properties:
                        original_values[key] = \
                            self.arguments.properties[key](_ID=_ID)
                        self.arguments.properties[key] \
                            .set_value(value, _ID=_ID)

            # This executes the feature.
            # DeepTrackNode will determine if it needs to be recalculated.
            # If it does, it will call the `.action()` method.
            output = super().__call__(_ID=_ID)

            # If there are self.arguments, reset the values of self.arguments
            # to their original values.
            for key, value in original_values.items():
                self.arguments.properties[key].set_value(value, _ID=_ID)

        return output

    resolve = __call__

    def to_sequential(  # TODO
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

    def torch(
        self: Feature,
        device: torch.device | None = None,
        recursive: bool = True,
    ) -> Feature:
        """Set the backend to torch.

        Parameters
        ----------
        device: torch.device, optional
            The device to use during evaluation (e.g. CPU, CUDA, or MPS).
            If provided, the feature's device is updated via `.to(device)`.
            Defaults to `None`.
        recursive: bool, optional
            If `True` (default), it also converts all dependent features.
            If `False`, it does not.

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt
        >>> import torch

        Create a feature and switch to the PyTorch backend:

        >>> feature = dt.Multiply(b=2)
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

        >>> f1 = dt.Add(b=1)
        >>> f2 = dt.Multiply(b=2)
        >>> pipeline = f1 >> f2
        >>> pipeline.torch()
        >>> output = pipeline(torch.tensor([1.0, 2.0]))
        >>> output
        tensor([4., 6.])

        """

        self._backend = "torch"

        if device is not None:
            self.to(device)

        if recursive:
            for dependency in self.recurse_dependencies():
                if isinstance(dependency, Feature):
                    dependency.torch(device=device, recursive=False)

        self.invalidate()

        return self

    def numpy(
        self: Feature,
        recursive: bool = True,
    ) -> Feature:
        """Set the backend to numpy.

        The NumPy backend does not support non-CPU devices. Calling `.numpy()`
        resets the feature's device to `"cpu"`.

        Parameters
        ----------
        recursive: bool, optional
            If `True` (default), also converts all dependent features.

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt
        >>> import numpy as np

        Create a feature and ensure it uses the NumPy backend:

        >>> feature = dt.Add(b=5)
        >>> feature.numpy()

        Evaluate the feature on a NumPy array:

        >>> output = feature(np.array([1, 2, 3]))
        >>> output
        array([6, 7, 8])

        Apply recursively in a pipeline:

        >>> f1 = dt.Multiply(b=2)
        >>> f2 = dt.Subtract(b=1)
        >>> pipeline = f1 >> f2
        >>> pipeline.numpy()
        >>> output = pipeline(np.array([1, 2, 3]))
        >>> output
        array([1, 3, 5])

        """

        self._backend = "numpy"

        # NumPy backend does not support non-CPU devices.
        self.to("cpu")

        if recursive:
            for dependency in self.recurse_dependencies():
                if isinstance(dependency, Feature):
                    dependency.numpy(recursive=False)

        self.invalidate()

        return self

    def get_backend(self: Feature) -> Literal["numpy", "torch"]:
        """Get the current backend of the feature.

        Returns
        -------
        "numpy" or "torch"
            The backend of this feature.

        Examples
        --------
        >>> import deeptrack as dt

        Create a feature:

        >>> feature = dt.Add(b=5)

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
        """Set the dtypes to be used during evaluation.

        It alters the dtypes used for array creation, but does not
        automatically cast the type.

        Parameters
        ----------
        float: str, optional
            The float dtype to set. Can be `"float32"`, `"float64"`,
            `"default"`, or `None`. Defaults to `None`.
        int: str, optional
            The int dtype to set. Can be `"int16"`, `"int32"`, `"int64"`,
            `"default"`, or `None`. Defaults to `None`.
        complex: str, optional
            The complex dtype to set. Can be `"complex64"`, `"complex128"`,
            `"default"`, or `None`. Defaults to `None`.
        bool: str, optional
            The bool dtype to set. Can be `"bool"`, `"default"`, or `None`.
            Defaults to `None`.

        Returns
        -------
        Feature
            self

        Examples
        --------
        >>> import deeptrack as dt

        Set float and int data types for a feature:

        >>> feature = dt.Multiply(b=2)
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
        dtype('float64')  # Depends on backend config

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

        >>> feature = dt.Add(b=1)
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

        # NumPy backend is CPU-only. We explicitly allow both "cpu" and
        # torch.device("cpu") to avoid spurious warnings, while normalizing
        # any other device request back to CPU.
        if self._backend == "numpy" and not (
            device == "cpu"
            or (
                TORCH_AVAILABLE
                and isinstance(device, torch.device)
                and device.type == "cpu"
            )
        ):
            warnings.warn(
                "NumPy backend only supports CPU; "
                "device has been reset to 'cpu'.",
                UserWarning,
            )
            device = "cpu"

        if device != self._device:
            self._device = device
            self.invalidate()

        return self

    def batch(
        self: Feature,
        batch_size: int = 32,
    ) -> tuple:
        """Batch the feature.

        This method produces a batch of outputs by repeatedly calling `.new()`.

        Parameters
        ----------
        batch_size: int, optional
            The number of times to sample or generate data. Defaults to 32.

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
        ...     >> dt.Add(b=lambda: np.random.rand())
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

        samples = [self.new() for _ in range(batch_size)]

        # Normalize the output structure:
        # If a sample is a tuple, treat it as multi-output, (y1, y2, ...).
        # Otherwise, treat it as a single-output feature and wrap it as (y,).
        # This preserves the number of output components and makes batching
        # consistent across single- and multi-output features.
        normalized: list[tuple[Any, ...]] = []
        for sample in samples:
            if isinstance(sample, tuple):
                normalized.append(sample)
            else:
                normalized.append((sample,))

        # Group outputs by component:
        # normalized = [(a1, b1), (a2, b2), (a3, b3)]
        # components = [(a1, a2, a3), (b1, b2, b3)]
        components = list(zip(*normalized))

        # Stack each component along a new leading batch axis.
        batched = [xp.stack(component) for component in components]

        return tuple(batched)

    def _action(  # TODO
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

        - `_process_properties()`: This hook can be overridden to pre-process
          properties before they are passed to `get()` (e.g., for unit
          normalization).

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
        ...     >> dt.Add(b=0.5)
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
        ...     >> dt.Add(b=0.5)
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

    def update(  # TODO
        self: Feature,
        **global_arguments: Any,
    ) -> Feature:
        """Refresh the feature to generate a new output.

        By default, when a feature is called multiple times, it returns the
        same value, which is cached.

        Calling `.update()` forces the feature to recompute and return a new
        value the next time it is evaluated.

        Calling `.new()` is equivalent to calling `.update()` plus evaulation.

        Parameters
        ----------
        **global_arguments: Any
            DEPRECATED. Has no effect. Previously used to inject values during
            update. Use `Arguments` or call-time overrides instead.

        Returns
        -------
        Feature
            The updated feature instance, ensuring the next evaluation produces
            a fresh result.

        Examples
        -------
        >>> import deeptrack as dt

        Create and resolve a feature:

        >>> import numpy as np
        >>>
        >>> feature = dt.Value(lambda: np.random.rand())
        >>> output1 = feature()
        >>> output1
        0.9173610765203623

        When resolving it again, it returns the same value:

        >>> output2 = feature()
        >>> output2  # Same as before
        0.9173610765203623

        Using `.update()` forces re-evaluation when resolved:

        >>> feature.update()  # Feature updated
        >>> output3 = feature()
        >>> output3
        0.13917950359184617

        Using `.new()` both updates and resolves the feature:

        >>> output4 = feature.new()
        >>> output4
        0.006278518685428169

        """

        if global_arguments:
            # Deprecated, but not necessary to raise hard error.
            warnings.warn(
                "Passing information through .update is no longer supported. "
                "A quick fix is to pass the information when resolving the "
                "feature. The prefered solution is to use dt.Arguments",
                DeprecationWarning,
                stacklevel=2,
            )

        super().update()

        return self

    def add_feature(  # TODO
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
        >>> feature = dt.Add(b=2)

        Define a side-effect feature:
        >>> dependency = dt.Value(b=42)

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

    def seed(  # TODO
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
        ...     print(f"output={feature.new()} seed={feature.seed()}")
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
        ...    print(f"output={feature.new()} seed={feature.seed()}")
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

    def bind_arguments(  # TODO
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
        >>> pipeline = dt.Value(value=3) >> dt.Add(b=1 * arguments.scale)
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

    def plot(  # TODO
        self: Feature,
        input_image: (
            np.ndarray
            | list[np.ndarray]
            | torch.Tensor
            | list[torch.Tensor]
        ) = None,
        resolve_kwargs: dict = None,
        interval: float = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize the output of the feature.

        The `.plot()` method resolves the feature and visualizes the result. If
        the output is a single image (NumPy array or PyTorch tensor), it is
        displayed using `pyplot.imshow()`. If the output is a list, an
        animation is created. In Jupyter notebooks, the animation is played
        inline using `to_jshtml()`. In scripts, the animation is displayed
        using the matplotlib backend.

        Any parameters in `kwargs` are passed to `pyplot.imshow`.

        Parameters
        ----------
        input_image: np.ndarray, torch.tensor, or Image or list[np.ndarray,
            torch.tensor, or Image], optional
            The input image or list of images passed as an argument to the
            `resolve` call. If `None`, uses previously set input values or
            propagates properties.
        resolve_kwargs: dict, optional
            Additional keyword arguments passed to the `resolve` call.
        interval: float, optional
            The time between frames in the animation, in milliseconds. The
            default value is 33 ms.
        **kwargs: dict, optional
            Additional keyword arguments passed to `pyplot.imshow`.

        Returns
        -------
        Any
            The output of the feature or pipeline after execution.
        
        Examples
        --------
        >>> import deeptrack as dt

        Create an instance of a dummy feature that returns the input:
        >>> feature = dt.DummyFeature()

        Generate and plot a grayscale image:
        >>> import numpy as np
        >>>
        >>> img = np.random.randint(0, 256, (64, 64))
        >>> feature.plot(img, cmap="gray");

        Generate and plot a grayscale video:
        >>> video = [np.random.randint(0, 256, (64, 64)) for _ in range(10)]
        >>> feature.plot(video, interval=100, cmap="gray");

        Generate a grayscale image using torch and plot it:
        >>> import torch
        >>>
        >>> img = torch.randint(0, 256, size=(64, 64))
        >>> feature.plot(img, cmap="gray");

        Generate a simulated image of a point particle visualized using
        brightfield microscopy and plot it:
        >>> particle = dt.PointParticle()
        >>> optics = dt.Brightfield()
        >>> imaged_particle = optics(particle)
        >>> imaged_particle.plot(cmap="gray");

        """

        from IPython.display import HTML, display

        output_image = self.resolve(input_image, **(resolve_kwargs or {}))

        # If a list, assume video
        if not isinstance(output_image, list):
            # Single image
            output_image = xp.squeeze(output_image)
            plt.imshow(output_image, **kwargs)
            return plt.gca()

        # Assume video
        fig = plt.figure()
        images = []
        plt.axis("off")
        for image in output_image:
            image = xp.squeeze(image)
            images.append([plt.imshow(image, **kwargs)])

        if not interval:
            if isinstance(output_image[0], Image):
                interval = (
                    output_image[0].get_property("interval") or (1 / 30 * 1000)
                )
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

    def _normalize(  # TODO
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

    def _process_properties(  # TODO
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

    def _activate_sources(  # TODO
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
        standard attribute syntax. For example,
        
        >>> feature.my_property
        
        is equivalent to

        >>> feature.properties["my_property"]

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

        An attempt to access a non-existent property raises an
        `AttributeError`:

        >>> feature.nonexistent
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
        protocol. The actual sampling and pipeline evaluation occur in
        `__next__()`, which is called at each iteration step.

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

        Use the feature in a loop (requiring manual termination):

        >>> for sample in feature:
        ...     print(sample)
        ...     if sample > 0.5:
        ...         break
        0.43126475134786546
        0.3270413736199965
        0.6734339603677173

        Use the feature for a predefined number of iterations:

        >>> from itertools import islice
        >>>
        >>> for sample in islice(feature, 2):
        ...     print(sample)
        0.43126475134786546
        0.3270413736199965

        """

        return self

    def __next__(
        self: Feature,
    ) -> Any:
        """Return the next resolved feature in the sequence.

        This method allows a `Feature` to be used as an iterator that yields
        a new result at each step. It is called automatically by
        `next(feature)` or when used in iteration.

        Each call to `__next__()` triggers a resampling of all properties and
        evaluation of the pipeline by calling `self.new()`.

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

        return self.new()

    def __rshift__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Chain this feature with another node or callable using `>>`.

        This operator enables pipeline-style chaining. The expression:

        >>> feature >> other

        is equivalent to

        >>> Chain(feature, other)

        It creates a new pipeline where the output of `feature` is passed as
        input to `other`:
        - If `other` is a `Feature` or `DeepTrackNode`, this returns a
          `Chain(feature, other)`.
        - If `other` is callable, it is wrapped in a `Lambda` node and
          chained as `Chain(feature, Lambda(lambda: other))`. The zero-argument
          lambda returns the callable, which is then invoked internally with
          the upstream output during evaluation.
        - Otherwise, this method returns `NotImplemented`.

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
        >>> feature2 = dt.Add(b=1)
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
            return Chain(self, Lambda(lambda: other))

        # The operator is not implemented for other inputs.
        return NotImplemented

    def __rrshift__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Reflected `>>` operator for chaining into this feature.

        This method is only invoked when the left operand implements
        `.__rshift__()` and returns `NotImplemented`. In that case, this
        method attempts to create a chain where `other` is evaluated before
        this feature.

        Important
        ---------
        Python does not call `.__rrshift__()` for most built-in types (e.g.,
        list, tuple, NumPy arrays, or PyTorch tensors) because these types do
        not define `.__rshift__()`. Therefore, expressions like:

        [1, 2, 3] >> feature

        raise `TypeError` and will not reach this method.

        To start a pipeline from a raw value, wrap it explicitly:

        Value(value=[1, 2, 3]) >> feature

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
            `NotImplemented`, which may raise a `TypeError`.

        """

        if isinstance(other, Feature):
            return Chain(other, self)

        return NotImplemented

    def __add__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Adds another value or feature using '+'.

        This operator is shorthand for chaining with `Add`. The expression

        >>> feature + other

        is equivalent to

        >>> feature >> dt.Add(b=other)

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

        >>> pipeline = feature >> dt.Add(b=5)

        Add a dynamic feature that samples values at each call:

        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = feature + noise
        >>> result = pipeline()
        >>> result
        [1.325563919290048, 2.325563919290048, 3.325563919290048]

        This is equivalent to:

        >>> pipeline = feature >> dt.Add(b=noise)

        """

        return self >> Add(b=other)

    def __radd__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Adds this feature to another value using right '+'.

        This operator is the right-hand version of `+`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other + feature

        is equivalent to

        >>> dt.Value(value=other) >> dt.Add(b=feature)

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

        >>> pipeline = dt.Value(value=5) >> dt.Add(b=feature)

        Add a feature to a dynamic value:

        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = noise + feature
        >>> result = pipeline()
        >>> result
        [1.5254613210875014, 2.5254613210875014, 3.5254613210875014]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.rand())
        ...     >> dt.Add(b=feature)
        ... )

        """

        return Value(value=other) >> Add(b=self)

    def __sub__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Subtract another value or feature using '-'.

        This operator is shorthand for chaining with `Subtract`. The expression

        >>> feature - other

        is equivalent to

        >>> feature >> dt.Subtract(b=other)

        Internally, this method constructs a new `Subtract` feature and uses
        the right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to be subtracted. It is passed to `Subtract`
            as the `value` argument.

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

        >>> pipeline = feature >> dt.Subtract(b=2)

        Subtract a dynamic feature that samples a value at each call:

        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = feature - noise
        >>> result = pipeline()
        >>> result
        [4.524072925059197, 5.524072925059197, 6.524072925059197]

        This is equivalent to:

        >>> pipeline = feature >> dt.Subtract(b=noise)
        
        """

        return self >> Subtract(b=other)

    def __rsub__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Subtract this feature from another value using right '-'.

        This operator is the right-hand version of `-`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other - feature

        is equivalent to

        >>> dt.Value(value=other) >> dt.Subtract(b=feature)

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

        >>> pipeline = dt.Value(value=5) >> dt.Subtract(b=feature)

        Subtract a feature from a dynamic value:

        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = noise - feature
        >>> result = pipeline()
        >>> result
        [-0.18761746914784516, -1.1876174691478452, -2.1876174691478454]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.rand())
        ...     >> dt.Subtract(b=feature)
        ... )

        """

        return Value(value=other) >> Subtract(b=self)

    def __mul__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Multiply this feature with another value using '*'.

        This operator is shorthand for chaining with `Multiply`. The expression

        >>> feature * other

        is equivalent to

        >>> feature >> dt.Multiply(b=other)

        Internally, this method constructs a new `Multiply` feature and uses
        the right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to be multiplied. It is passed to `Multiply`
            as the `value` argument.

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

        >>> pipeline = feature >> dt.Multiply(b=2)

        Multiply with a dynamic feature that samples a value at each call:

        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = feature * noise
        >>> result = pipeline()
        >>> result
        [0.2809370704818722, 0.5618741409637444, 0.8428112114456167]

        This is equivalent to:

        >>> pipeline = feature >> dt.Multiply(value=noise)

        """

        return self >> Multiply(b=other)

    def __rmul__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Multiply another value by this feature using right '*'.

        This operator is the right-hand version of `*`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other * feature

        is equivalent to

        >>> dt.Value(value=other) >> dt.Multiply(b=feature)

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

        >>> pipeline = dt.Value(value=2) >> dt.Multiply(b=feature)

        Multiply a feature to a dynamic value:

        >>> import numpy as np
        >>>
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = noise * feature
        >>> result = pipeline()
        >>> result
        [0.8784860790329121, 1.7569721580658242, 2.635458237098736]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.rand())
        ...     >> dt.Multiply(b=feature)
        ... )

        """

        return Value(value=other) >> Multiply(b=self)

    def __truediv__(
        self: Feature,
        other: Any,
        ) -> Feature:
        """Divide a feature (nominator) using `/` by a value (denominator).

        This operator is shorthand for chaining with `Divide`. The expression

        >>> feature / other

        is equivalent to

        >>> feature >> dt.Divide(value=other)

        Internally, this method constructs a new `Divide` feature and uses the
        right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to divide feature with. It is passed to
            `Divide` as the `value` argument.

        Returns
        -------
        Feature
            A new feature that is `self` divided by `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Divide a feature with a constant:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature / 5
        >>> result = pipeline()
        >>> result
        [0.2, 0.4, 0.6]

        This is equivalent to:

        >>> pipeline = feature >> dt.Divide(value=5)

        Implement a normalization pipeline:

        >>> feature = dt.Value(value=[1, 25, 20])
        >>> magnitude = dt.Value(value=lambda: max(feature()))
        >>> pipeline = feature / magnitude
        >>> result = pipeline()
        >>> result
        [0.04, 1.0, 0.8]

        This is equivalent to:

        >>> pipeline = (
        ...     feature
        ...     >> dt.Divide(value=lambda: max(feature()))
        ... )

        """

        return self >> Divide(b=other)

    def __rtruediv__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Divide other value (nominator) by feature (denominator) using '/'.

        This operator is shorthand for chaining with `Divide`, and is the
        right-hand side version of  `__truediv__`.

        The expression

        >>> other / feature

        is equivalent to

        >>> other >> dt.Divide(b=feature)

        Internally, this method constructs a new `Value` feature from `other`
        and uses the right-shift operator (`>>`) to chain it into a `Divide`
        feature that divides the current feature as a dynamic value.

        Parameters
        ----------
        other: Any
            The constant or `Feature` to be divided by `self`. It is passed to
            `Divide` as the input to `Value`.

        Returns
        -------
        Feature
            A new feature that is `other` divided by `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Divide a constant with a feature:

        >>> feature = dt.Value(value=[-1, 2, 2])
        >>> pipeline = 5 / feature
        >>> result = pipeline()
        >>> result
        [-5.0, 2.5, 2.5]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=5)
        ...     >> dt.Divide(b=feature)
        ... )

        Divide a dynamic value with a feature:

        >>> import numpy as np
        >>>
        >>> scale_factor = dt.Value(value=5)
        >>> noise = dt.Value(value=lambda: np.random.rand())
        >>> pipeline = noise / scale_factor
        >>> result = pipeline()
        >>> result
        0.13736078990870043

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.rand())
        ...     >> dt.Divide(value=scale_factor)
        ... )

        """

        return Value(value=other) >> Divide(b=self)

    def __floordiv__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Perform floor division of feature with other value using `//`.
    
        It performs the floor division of `feature` (numerator) with `other`
        value (denominator) using `//`.
    
        This operator is shorthand for chaining with `FloorDivide`.
        The expression
    
        >>> feature // other

        is equivalent to

        >>> feature >> dt.FloorDivide(value=other)
    
        Internally, this method constructs a new `FloorDivide` feature and uses
        the right-shift operator (`>>`) to chain the current feature with it.
    
        Parameters
        ----------
        other: Any
            A constant or `Feature` by which `self` will be floor-divided. It
            is passed as the input to `value`.
    
        Returns
        -------
        Feature
            A new feature that floor divides `self` with `other`.
    
        Examples
        --------
        >>> import deeptrack as dt
    
        Floor divide a feature with a constant:

        >>> feature = dt.Value(value=[5, 9, 12])
        >>> pipeline = feature // 2
        >>> result = pipeline()
        >>> result
        [2, 4, 6]
    
        This is equivalent to:

        >>> pipeline = feature >> dt.FloorDivide(value=2)
    
        Floor divide a dynamic feature by another feature:

        >>> import numpy as np
        >>>
        >>> randint = dt.Value(value=lambda: np.random.randint(1, 5))
        >>> feature = dt.Value(value=[20, 30, 40])
        >>> pipeline = feature // randint
        >>> result = pipeline()
        >>> result
        [6, 10, 13]
        
        This is equivalent to:

        >>> pipeline = (
        ...     feature
        ...     >> dt.FloorDivide(value=lambda: np.random.randint(1, 5))
        ... )

        """

        return self >> FloorDivide(b=other)

    def __rfloordiv__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Perform floor division of other with feature using '//'.
    
        This operator performs the floor division of `other` (numerator) with
        `feature` (denominator) using '//'.
    
        This operator is shorthand for chaining with `FloorDivide`.
        The expression
    
        >>> other // feature
    
        is equivalent to
    
        >>> dt.Value(value=other) >> dt.FloorDivide(b=feature)
    
        Internally, this method constructs a `Value` feature from `other` and
        chains it into a `FloorDivide` feature that divides with the current
        feature.
    
        Parameters
        ----------
        other: Any
            A constant or `Feature` which will be floor divided with `self`.
            It is passed as the input to `Value`.
    
        Returns
        -------
        Feature
            A new feature that floor divides `other` with `self`.
    
        Examples
        --------
        >>> import deeptrack as dt
    
        Floor divide a feature with a constant:

        >>> feature = dt.Value(value=[5, 9, 12])
        >>> pipeline = 10 // feature
        >>> result = pipeline()
        >>> result
        [2, 1, 0]
    
        This is equivalent to:

        >>> pipeline = dt.Value(value=10) >> dt.FloorDivide(b=feature)
    
        Floor divide a dynamic feature by another feature:

        >>> import numpy as np
        >>>
        >>> randint = dt.Value(value=lambda: np.random.randint(1, 5))
        >>> feature = dt.Value(value=[2, 3, 4])
        >>> pipeline = randint // feature
        >>> result = pipeline()
        >>> result
        [1, 1, 0]
        
        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.randint(1, 5))
        ...     >> dt.FloorDivide(b=feature)
        ... )
        
        """

        return Value(value=other) >> FloorDivide(b=self)

    def __pow__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Raise this feature (base) to a power (exponent) using '**'.

        This operator is shorthand for chaining with `Power`. The expression

        >>> feature ** other

        is equivalent to

        >>> feature >> dt.Power(b=other)

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

        This is equivalent to

        >>> pipeline = feature >> dt.Power(value=3)

        Raise to a dynamic exponent that samples values at each call:

        >>> import numpy as np
        >>>
        >>> random_exponent = dt.Value(value=lambda: np.random.randint(10))
        >>> pipeline = feature ** random_exponent
        >>> result = pipeline()
        >>> result
        [1, 64, 729]

        This is equivalent to

        >>> pipeline = feature >> dt.Power(b=random_exponent)
 
        """

        return self >> Power(b=other)

    def __rpow__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Raise another value (base) to this feature (exponent) using '**'.

        This operator is the right-hand version of `**`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other ** feature

        is equivalent to

        >>> dt.Value(value=other) >> dt.Power(b=feature)

        Internally, this method constructs a `Value` feature from `other`
        (base) and chains it into a `Power` feature (exponent).

        Parameters
        ----------
        other: Any
            A constant or `Feature` representing the base. It is passed as the
            `value` argument to `Value`.

        Returns
        -------
        Feature
            A new feature representing `other` to the power of `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Raise a static base to a constant exponent:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 5 ** feature
        >>> result = pipeline()
        >>> result
        [5, 25, 125]

        This is equivalent to:

        >>> pipeline = dt.Value(value=5) >> dt.Power(b=feature)

        Raise a dynamic base that samples values at each call to the static
        exponent:

        >>> import numpy as np
        >>>
        >>> random_base = dt.Value(value=lambda: np.random.randint(10))
        >>> pipeline = random_base ** feature
        >>> result = pipeline()
        >>> result
        [9, 81, 729]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=lambda: np.random.randint(10))
        ...     >> dt.Power(b=feature)
        ... )

        """

        return Value(value=other) >> Power(b=self)

    def __gt__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if this feature is greater than another using '>'.

        This operator is shorthand for chaining with `GreaterThan`.
        The expression

        >>> feature > other

        is equivalent to

        >>> feature >> dt.GreaterThan(b=other)

        Internally, this method constructs a new `GreaterThan` feature and uses
        the right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to compare against. It is passed to
            `GreaterThan` as the `value` argument.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of greater-than
            comparison between `self` and `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare each element in a feature to a constant:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature > 2
        >>> result = pipeline()
        >>> result
        [False, False, True]

        This is equivalent to:

        >>> pipeline = feature >> dt.GreaterThan(b=2)

        Compare to a dynamic cutoff that samples values at each call:

        >>> import numpy as np
        >>>
        >>> random_cutoff = dt.Value(value=lambda: np.random.randint(3))
        >>> pipeline = feature > random_cutoff
        >>> result = pipeline()
        >>> result
        [False, True, True]

        This is equivalent to:

        >>> pipeline = feature >> dt.GreaterThan(b=random_cutoff)

        """

        return self >> GreaterThan(b=other)

    def __rgt__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if another value is greater than feature using right '>'.
 
        This operator is the right-hand version of `>`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other > feature

        is equivalent to:

        >>> dt.Value(value=other) >> dt.GreaterThan(b=feature)

        Internally, this method constructs a `Value` feature from `other`
        and chains it into a `GreaterThan` feature.

        Parameters
        ----------
        other: Any
            A constant or `Feature` to compare against. It is passed as
            the `value` argument to `Value`.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of greater-than
            comparison between `other` and `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare a constant to each element in a feature:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 2 > feature
        >>> result = pipeline()
        >>> result
        [True, False, False]

        This is equivalent to:

        >>> pipeline = dt.Value(value=2) >> dt.GreaterThan(b=feature)

        Compare a constant to each element in a dynamic feature that samples
        values at each call:

        >>> from random import randint
        >>>
        >>> random = dt.Value(value=lambda: [randint(0, 3) for _ in range(3)])
        >>> pipeline = 2 > random
        >>> result = pipeline()
        >>> result
        [False, False, True]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=2)
        ...     >> dt.GreaterThan(b=lambda: [randint(0, 3) for _ in range(3)])
        ... )

        """

        return Value(value=other) >> GreaterThan(b=self)

    def __lt__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if this feature is less than another using '<'.

        This operator is shorthand for chaining with `LessThan`.
        The expression

        >>> feature < other

        is equivalent to

        >>> feature >> dt.LessThan(b=other)

        Internally, this method constructs a new `LessThan` feature and
        uses the right-shift operator (`>>`) to chain the current feature
        into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to compare against. It is passed to
            `LessThan` as the `value` argument.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of less-than
            comparison between `self` and `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare each element in a feature to a constant:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature < 2
        >>> result = pipeline()
        >>> result
        [True, False, False]

        This is equivalent to:

        >>> pipeline = feature >> dt.LessThan(b=2)

        Compare to a dynamic cutoff that samples values at each call:

        >>> import numpy as np
        >>>
        >>> random_cutoff = dt.Value(value=lambda: np.random.randint(3))
        >>> pipeline = feature < random_cutoff
        >>> result = pipeline()
        >>> result
        [False, False, False]

        This is equivalent to:

        >>> pipeline = feature >> dt.LessThan(b=random_cutoff)

        """

        return self >> LessThan(b=other)

    def __rlt__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if another value is less than this feature using right '<'.

        This operator is the right-hand version of `<`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other < feature

        is equivalent to

        >>> dt.Value(value=other) >> dt.LessThan(b=feature)

        Internally, this method constructs a `Value` feature from `other`
        and chains it into a `LessThan` feature.

        Parameters
        ----------
        other: Any
            A constant or `Feature` to compare against. It is passed as
            the `value` argument to `Value`.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of less-than
            comparison between `other` and `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare a constant to each element in a feature:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 2 < feature
        >>> result = pipeline()
        >>> result
        [False, False, True]

        This is equivalent to:

        >>> pipeline = dt.Value(value=2) >> dt.LessThan(b=feature)

        Compare a constant to each element in a dynamic feature that samples
        values at each call:

        >>> from random import randint
        >>>
        >>> random = dt.Value(value=lambda: [randint(0, 3) for _ in range(3)])
        >>> pipeline = 2 < random
        >>> result = pipeline()
        >>> result
        [False, True, False]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=2)
        ...     >> dt.LessThan(b=lambda: [randint(0, 3) for _ in range(3)])
        ... )

        """

        return Value(value=other) >> LessThan(b=self)

    def __le__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if this feature is less than or equal to another using '<='.

        This operator is shorthand for chaining with `LessThanOrEquals`.
        The expression

        >>> feature <= other

        is equivalent to

        >>> feature >> dt.LessThanOrEquals(b=other)

        Internally, this method constructs a new `LessThanOrEquals` feature
        and uses the right-shift operator (`>>`) to chain the current feature
        into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to compare against. It is passed to
            `LessThanOrEquals` as the `value` argument.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of
            less-than-or-equals comparison between `self` and `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare each element in a feature to a constant:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature <= 2
        >>> result = pipeline()
        >>> result
        [True, True, False]

        This is equivalent to:

        >>> pipeline = feature >> dt.LessThanOrEquals(b=2)

        Compare to a dynamic cutoff that samples values at each call:

        >>> import numpy as np
        >>>
        >>> random_cutoff = dt.Value(value=lambda: np.random.randint(3))
        >>> pipeline = feature <= random_cutoff
        >>> result = pipeline()
        >>> result
        [False, False, False]

        This is equivalent to:

        >>> pipeline = feature >> dt.LessThanOrEquals(b=random_cutoff)

        """

        return self >> LessThanOrEquals(b=other)

    def __rle__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if other is less than or equal to feature using right '<='.

        This operator is the right-hand version of `<=`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other <= feature

        is equivalent to

        >>> dt.Value(value=other) >> dt.LessThanOrEquals(b=feature)

        Internally, this method constructs a `Value` feature from `other`
        and chains it into a `LessThanOrEquals` feature.

        Parameters
        ----------
        other: Any
            A constant or `Feature` to compare against. It is passed as
            the `value` argument to `Value`.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of
            less-than-or-equals comparison between `other` and `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare a constant to each element in a feature:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 2 <= feature
        >>> result = pipeline()
        >>> result
        [False, True, True]

        This is equivalent to:

        >>> pipeline = dt.Value(value=2) >> dt.LessThanOrEquals(b=feature)

        Compare a constant to each element in a dynamic feature that samples
        values at each call:

        >>> from random import randint
        >>>
        >>> random = dt.Value(value=lambda: [randint(0, 3) for _ in range(3)])
        >>> pipeline = 2 <= random
        >>> result = pipeline()
        >>> result
        [True, False, False]

        This is equivalent to:

        >>> pipeline = (
        ...     dt.Value(value=2)
        ...     >> dt.LessThanOrEquals(
        ...         b=lambda: [randint(0, 3) for _ in range(3)]
        ...     )
        ... )

        """

        return Value(value=other) >> LessThanOrEquals(b=self)

    def __ge__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if this feature is greater than or equal to other using '>='.

        This operator is shorthand for chaining with `GreaterThanOrEquals`.
        The expression

        >>> feature >= other

        is equivalent to

        >>> feature >> dt.GreaterThanOrEquals(b=other)

        Internally, this method constructs a new `GreaterThanOrEquals` feature
        and uses the right-shift operator (`>>`) to chain the current feature
        into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to compare against. It is passed to
            `GreaterThanOrEquals` as the `value` argument.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of
            greater-than-or-equals comparison between `self` and `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare each element in a feature to a constant:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature >= 2
        >>> result = pipeline()
        >>> result
        [False, True, True]

        This is equivalent to:

        >>> pipeline = feature >> dt.GreaterThanOrEquals(b=2)

        Compare to a dynamic cutoff that samples values at each call:

        >>> import numpy as np
        >>>
        >>> random_cutoff = dt.Value(value=lambda: np.random.randint(3))
        >>> pipeline = feature >= random_cutoff
        >>> result = pipeline()
        >>> result
        [True, True, True]

        This is equivalent to:

        >>> pipeline = feature >> dt.GreaterThanOrEquals(b=random_cutoff)

        """

        return self >> GreaterThanOrEquals(b=other)

    def __rge__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Check if other is greater than or equal to feature using right '>='.

        This operator is the right-hand version of `>=`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression

        >>> other >= feature

        is equivalent to

        >>> dt.Value(value=other) >> dt.GreaterThanOrEquals(b=feature)

        Internally, this method constructs a `Value` feature from `other`
        and chains it into a `GreaterThanOrEquals` feature.

        Parameters
        ----------
        other: Any
            A constant or `Feature` to compare against. It is passed as
            the `value` argument to `Value`.

        Returns
        -------
        Feature
            A new feature representing the element-wise result of
            greater-than-or-equals comparison between `other` and `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Compare a constant to each element in a feature:

        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = 2 >= feature
        >>> result = pipeline()
        >>> result
        [True, True, False]

        This is equivalent to:

        >>> pipeline = (dt.Value(value=2) >> dt.GreaterThanOrEquals(b=feature))

        Compare a constant to each element in a dynamic feature that samples
        values at each call:

        >>> from random import randint
        >>>
        >>> random = dt.Value(value=lambda: [randint(0, 3) for _ in range(3)])
        >>> pipeline = 2 >= random
        >>> result = pipeline()
        >>> result
        [True, False, True]

        This is equivalent to:
        >>> pipeline = (
        ...     dt.Value(value=2)
        ...     >> dt.GreaterThanOrEquals(
        ...         b=lambda: [randint(0, 3) for _ in range(3)]
        ...     )
        ... )

        """

        return Value(value=other) >> GreaterThanOrEquals(b=self)

    def __xor__(  # TODO
        self: Feature,
        other: int,
    ) -> Feature:
        """Repeat the feature a given number of times using '^'.

        This operator is shorthand for chaining with `Repeat`. The expression:

        >>> feature ^ other

        is equivalent to:

        >>> dt.Repeat(feature, N=other)

        Internally, this method constructs a new `Repeat` feature taking
        `self` and `other` as argument.

        Parameters
        ----------
        other: int
            The int value representing the repeat times. It is passed to
            `Repeat` as the `N` argument.

        Returns
        -------
        Feature
            A new feature that applies `self` repeatedly `other` times.

        Examples
        --------
        >>> import deeptrack as dt

        Repeat the `Add` feature by 3 times:
        >>> add_ten = dt.Add(value=10)
        >>> pipeline = add_ten ^ 3
        >>> result = pipeline([1, 2, 3])
        >>> result
        [31, 32, 33]

        This is equivalent to:
        >>> pipeline = dt.Repeat(add_ten, N=3)

        Repeat by random times that samples values at each call:
        >>> import numpy as np
        >>>
        >>> random_times = dt.Value(value=lambda: np.random.randint(10))
        >>> pipeline = add_ten ^ random_times
        >>> result = pipeline.update()([1, 2, 3])
        >>> result
        [81, 82, 83]

        This is equivalent to:
        >>> pipeline = dt.Repeat(add_ten, N=random_times)

        """

        return Repeat(self, other)

    def __and__(  # TODO
        self: Feature,
        other: Any,
    ) -> Feature:
        """Stack this feature with another using '&'.

        This operator is shorthand for chaining with `Stack`. The expression:

        >>> feature & other

        is equivalent to:

        >>> feature >> dt.Stack(value=other)

        Internally, this method constructs a new `Stack` feature and uses the
        right-shift operator (`>>`) to chain the current feature into it.

        Parameters
        ----------
        other: Any
            The value or `Feature` to stack with `self`.

        Returns
        -------
        Feature
            A new feature containing all elements from `self` and `other`.

        Examples
        --------
        >>> import deeptrack as dt

        Stack with the fixed data:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = feature & [4, 5, 6]
        >>> result = pipeline()
        >>> result
        [1, 2, 3, 4, 5, 6]

        This is equivalent to:
        >>> pipeline = feature >> dt.Stack(value=[4, 5, 6])

        Stack with the dynamic data that samples values at each call:
        >>> from random import randint
        >>>
        >>> random = dt.Value(value=lambda: [randint(0, 3) for _ in range(3)])
        >>> pipeline = feature & random
        >>> result = pipeline()
        >>> result
        [1, 2, 3, 3, 1, 3]

        This is equivalent to:
        >>> pipeline = feature >> dt.Stack(value=random)
 
        """

        return self >> Stack(other)

    def __rand__(  # TODO
        self: Feature,
        other: Any,
    ) -> Feature:
        """Stack another value with this feature using right '&'.

        This operator is the right-hand version of `&`, enabling expressions
        where the `Feature` appears on the right-hand side. The expression:

        >>> other & feature

        is equivalent to:

        >>> dt.Value(value=other) >> dt.Stack(value=feature)

        Internally, this method constructs a `Value` feature from `other`
        and chains it into a `Stack` feature.

        Parameters
        ----------
        other: Any
            The value or `Feature` to stack with `self`.

        Returns
        -------
        Feature
            A new feature containing all elements from `other` and `self`.

        Examples
        --------
        >>> import deeptrack as dt

        Stack with the fixed data:
        >>> feature = dt.Value(value=[1, 2, 3])
        >>> pipeline = [4, 5, 6] & feature
        >>> result = pipeline()
        >>> result
        [4, 5, 6, 1, 2, 3]

        This is equivalent to:
        >>> pipeline = dt.Value(value=[4, 5, 6]) >> dt.Stack(value=feature)

        Stack with the dynamic data that samples values at each call:
        >>> from random import randint
        >>>
        >>> random = dt.Value(value=lambda: [randint(0, 3) for _ in range(3)])
        >>> pipeline = random & feature
        >>> result = pipeline()
        >>> result
        [0, 3, 1, 1, 2, 3]

        This is equivalent to:
        >>> pipeline = (
        ...     dt.Value(value=lambda:
        ...         [randint(0, 3) for _ in range(3)])
        ...     >> dt.Stack(value=feature)
        ... )
        
        """

        return Value(other) >> Stack(self)

    def __getitem__(  # TODO
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
    def _format_input(self: Feature) -> Callable[[Any], list[Any or Image]]:  # TODO
        """Select the appropriate input formatting function for configuration.

        Returns either `_image_wrapped_format_input` or
        `_no_wrap_format_input`, depending on whether image metadata
        (properties) should be preserved and processed downstream.

        Returns
        -------
        Callable
            A function that formats the input into a list of Image objects or
            raw arrays, depending on the configuration.

        """

        return self._no_wrap_format_input

    @property
    def _process_and_get(self: Feature) -> Callable[[Any], list[Any or Image]]:  # TODO
        """Select the appropriate processing function based on configuration.

        Returns a method that applies the feature’s transformation (`get`) to
        the input data, either with or without wrapping and preserving `Image`
        metadata.

        Returns
        -------
        Callable
            A function that applies `.get()` to the input, either preserving
            or ignoring metadata depending on configuration.

        """

        return self._no_wrap_process_and_get

    def _no_wrap_format_input(  # TODO
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

    def _no_wrap_process_and_get(  # TODO
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


def propagate_data_to_dependencies(
    feature: Feature,
    _ID: tuple[int, ...] = (),
    **kwargs: Any,
) -> None:
    """Propagate values to existing properties in the dependency tree.

    This function traverses the dependency tree of `feature` and sets cached
    values for matching properties. Only properties that already exist in a
    dependency's `PropertyDict` are updated.

    Parameters
    ----------
    feature: Feature
        The feature whose dependency tree will be traversed.
    _ID: tuple[int, ...], optional
        The dataset identifier to store the propagated values at. Defaults to
        an empty tuple.
    **kwargs: Any
        Key-value pairs mapping property names to values. A value is propagated
        only if the corresponding property already exists in the dependency
        tree.

    Examples
    --------
    >>> import deeptrack as dt

    Update the properties of a feature and its dependencies:

    >>> feature = dt.DummyFeature(value=10)
    >>> dt.propagate_data_to_dependencies(feature, value=20)
    >>> feature.value()
    20

    >>> Update the properties of a feature and its dependencies at given `_ID`:

    >>> feature = dt.Value(value=1) >> dt.Add(b=1.0) >> dt.Multiply(b=2.0)
    >>> dt.propagate_data_to_dependencies(feature, _ID=(1,), b=3.0)
    >>> feature(_ID=(0,))
    4.0
    >>> feature(_ID=(1,))
    12.0

    """

    # TODO Decide whether to keep warning
    #matched_keys: set[str] = set()

    for dependency in feature.recurse_dependencies():
        if isinstance(dependency, PropertyDict):
            for key, value in kwargs.items():
                if key in dependency:
                    dependency[key].set_value(value, _ID=_ID)

                    #matched_keys.add(key)

    #unmatched_keys = set(kwargs) - matched_keys
    #if unmatched_keys:
    #    warnings.warn(
    #        "The following properties were not found in the dependency "
    #        f"tree and were ignored: {sorted(unmatched_keys)}",
    #        UserWarning,
    #        stacklevel=2,
    #    )


class StructuralFeature(Feature):
    """Provide the structure of a feature set without input transformations.

    A `StructuralFeature` serves as a logical and organizational tool for
    grouping, chaining, or structuring pipelines. It does not modify the input
    data or introduce new properties.

    This feature is typically used to:
    - group or chain sub-features (e.g., `Chain`)
    - apply conditional or sequential logic (e.g., `Probability`)
    - organize pipelines without affecting data flow (e.g., `Combine`)

    `StructuralFeature` inherits all behavior from `Feature`, without
    overriding the `.__init__()` or `.get()` methods.

    Attributes
    ----------
    __distributed__: bool
        If `False` (default), processes the entire input list as a single unit.
        If `True`, applies `.get()` to each element in the list individually.

    """

    __distributed__: bool = False  # Process the entire image list in one call


class Chain(StructuralFeature):
    """Resolve two features sequentially.

    `Chain` applies two features sequentially: the outputs of `feature_1` are
    passed as inputs to `feature_2`. This allows combining simple operations
    into complex pipelines.

    The use of `Chain`

    >>> dt.Chain(A, B)

    is equivalent to using the `>>` operator

    >>> A >> B

    Parameters
    ----------
    feature_1: Feature
        The first feature in the chain. Its outputs are passed to `feature_2`.
    feature_2: Feature
        The second feature in the chain proceses the outputs from `feature_1`.
    **kwargs: Any, optional
        Additional keyword arguments passed to the parent `StructuralFeature`
        (and, therefore, `Feature`).

    Attributes
    ----------
    feature_1: Feature
        The first feature in the chain. Its outputs are passed to `feature_2`.
    feature_2: Feature
        The second feature in the chain processes the outputs from `feature_1`.

    Methods
    -------
    `get(inputs, _ID, **kwargs) -> Any`
        Apply the two features in sequence on the given inputs.

    Examples
    --------
    >>> import deeptrack as dt

    Create a feature chain where the first feature adds a constant offset, and 
    the second feature multiplies the result by a constant:

    >>> A = dt.Add(b=10)
    >>> M = dt.Multiply(b=0.5)
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

    feature_1: Feature
    feature_2: Feature

    def __init__(
        self: Chain,
        feature_1: Feature,
        feature_2: Feature,
        **kwargs: Any,
    ):
        """Initialize the chain with two sub-features.

        Initializes the feature chain by setting `feature_1` and `feature_2`
        as dependencies. Updates to these sub-features automatically propagate
        through the DeepTrack2 computation graph, ensuring consistent
        evaluation and execution.

        Parameters
        ----------
        feature_1: Feature
            The first feature to be applied.
        feature_2: Feature
            The second feature, applied to the outputs of `feature_1`.
        **kwargs: Any
            Additional keyword arguments passed to the parent constructor
            (e.g., name, properties).

        """

        super().__init__(**kwargs)

        self.feature_1 = self.add_feature(feature_1)
        self.feature_2 = self.add_feature(feature_2)

    def get(
        self: Feature,
        inputs: Any,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Apply the two features sequentially to the given inputs.

        This method first applies `feature_1` to the inputs and then passes
        the outputs through `feature_2`.

        Parameters
        ----------
        inputs: Any
            The input data to transform sequentially. Most typically, this is
            a NumPy array or a PyTorch tensor.
        _ID: tuple[int, ...], optional
            A unique identifier for caching or parallel execution. 
            Defaults to an empty tuple.
        **kwargs: Any
            Additional parameters passed to or sampled by the features. These
            are unused here, as each sub-feature fetches its required
            properties internally.

        Returns
        -------
        Any
            The final outputs after `feature_1` and then `feature_2` have
            processed the inputs.

        """

        outputs = self.feature_1(inputs, _ID=_ID)
        outputs = self.feature_2(outputs, _ID=_ID)
        return outputs


Branch = Chain  # Alias for backwards compatibility


class DummyFeature(Feature):
    """A no-op feature that simply returns the inputs unchanged.

    `DummyFeature` can serve as a container for properties that do not directly
    transform the data but need to be logically grouped.
    
    Any keyword arguments passed to the constructor are stored as `Property`
    instances in `self.properties`, enabling dynamic behavior or
    parameterization without performing any transformations on the input data.

    Parameters
    ----------
    inputs: Any, optional
        Optional inputs for the feature. Defaults to an empty list.
    **kwargs: Any
        Additional keyword arguments are wrapped as `Property` instances and 
        stored in `self.properties`.

    Methods
    -------
    `get(inputs, **kwargs) -> Any`
        Simply returns the inputs unchanged.

    Examples
    --------
    >>> import deeptrack as dt

    Pass some input through a `DummyFeature` to demonstrate no changes.

    Create the input:

    >>> dummy_input = [1, 2, 3, 4, 5]

    Initialize the DummyFeature with two property:

    >>> dummy_feature = dt.DummyFeature(prop1=42, prop2=3.14)

    Pass the input through the DummyFeature:

    >>> dummy_output = dummy_feature(dummy_input)
    >>> dummy_output
    [1, 2, 3, 4, 5]

    The output is identical to the input.

    Access a property stored in DummyFeature:

    >>> dummy_feature.prop1()
    42

    """

    def get(
        self: DummyFeature,
        inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Return the input unchanged.

        This method simply returns the input without any transformation.
        It adheres to the `Feature` interface by accepting additional keyword
        arguments for consistency, although they are not used.

        Parameters
        ----------
        inputs: Any
            The input to pass through without modification.
        **kwargs: Any
            Additional properties sampled from `self.properties` or passed
            externally. These are unused here but provided for consistency
            with the `Feature` interface.

        Returns
        -------
        Any
            The input without modifications.

        """

        return inputs


class Value(Feature):
    """Represent a constant value in a DeepTrack2 pipeline.

    `Value` holds a constant value (e.g., a scalar or array) and supplies it on
    demand to other parts of the pipeline.
    
    If called with an input, it ignores it and still returns the stored value.

    Parameters
    ----------
    value: PropertyLike[Any], optional
        The value to store. Defaults to 0.
    **kwargs: Any
        Additional named properties passed to the `Feature` constructor.

    Attributes
    ----------
    __distributed__: bool
        Set to `False`, indicating that this feature’s `.get()` method
        processes the entire input at once even if it is a list, rather than 
        distributing calls for each item of the list.

    Methods
    -------
    `get(inputs, value, **kwargs) -> Any`
        Returns the stored value, ignoring the inputs.

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

    __distributed__: bool = False  # Process as a single batch

    def __init__(
        self: Value,
        value: PropertyLike[Any],
        **kwargs: Any,
    ):
        """Initialize the feature to store a constant value.

        `Value` holds a constant value and returns it as needed.

        Parameters
        ----------
        value: Any, optional
            The initial value to store. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the `Feature` constructor, 
            such as custom properties or the feature name.

        """

        super().__init__(value=value, **kwargs)

    def get(
        self: Value,
        inputs: Any,
        value: Any,
        **kwargs: Any,
    ) -> Any:
        """Return the stored value, ignoring the inputs.

        The `.get()` method simply returns the stored numerical value, allowing 
        for dynamic overrides when the feature is called.

        Parameters
        ----------
        inputs: Any
            `Value` ignores its input data.
        value: Any
            The current value to return. This may be the initial value or an 
            overridden value supplied during the method call.
        **kwargs: Any
            Additional keyword arguments, which are ignored but included for 
            consistency with the `Feature` interface.

        Returns
        -------
        Any
            The stored or overridden `value`, returned unchanged.

        """

        return value


class ArithmeticOperationFeature(Feature):
    """Apply an arithmetic operation element-wise to the inputs.

    This feature performs an arithmetic operation (e.g., addition, subtraction,
    multiplication) on the input data. The input can be a single value or a
    list of values.

    If a list is passed, the operation is applied to each element.

    If the inputs are lists of different lengths, the shorter list is cycled.

    Parameters
    ----------
    op: Callable[[Any, Any], Any]
        The arithmetic operation to apply, such as a built-in operator
        (e.g., `operator.add`, `operator.mul`) or a custom callable.
    b: Any or list[Any], optional
        The second operand for the operation. Defaults to 0. If a list is
        provided, the operation will apply element-wise.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature`.

    Attributes
    ----------
    __distributed__: bool
        Set to `False`, indicating that this feature’s `.get()` method
        processes the entire input at once even if it is a list, rather than 
        distributing calls for each item of the list.

    Methods
    -------
    `get(a, b, **kwargs) -> list[Any]`
        Apply the arithmetic operation element-wise to the input data.

    Examples
    --------
    >>> import deeptrack as dt

    Define a simple addition operation:

    >>> import operator
    >>>
    >>> addition = dt.ArithmeticOperationFeature(operator.add, b=10)

    Create a list of input values:

    >>> input_values = [1, 2, 3, 4]

    Apply the operation:

    >>> output_values = addition(input_values)
    >>> output_values
    [11, 12, 13, 14]

    """

    __distributed__: bool = False

    def __init__(
        self: ArithmeticOperationFeature,
        op: Callable[[Any, Any], Any],
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the base class for arithmetic operations.

        Parameters
        ----------
        op: Callable[[Any, Any], Any]
            The arithmetic operation to apply, such as `operator.add`,
            `operator.mul`, or any custom callable that takes two arguments and
            returns a single output value.
        b: PropertyLike[Any or list[Any]], optional
            The second operand(s) for the operation. Typically, it is a number
            or an array. If a list is provided, the  operation is applied
            element-wise. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature`
            constructor.

        """

        # Backward compatibility with deprecated 'value' parameter
        if "value" in kwargs:
            b = kwargs.pop("value")
            warnings.warn(
                "The 'value' parameter is deprecated and will be removed"
                "in a future version. Use 'b' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(b=b, **kwargs)

        self.op = op

    def get(
        self: ArithmeticOperationFeature,
        a: list[Any],
        b: Any | list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Apply the operation element-wise to the input data.

        Parameters
        ----------
        a: list[Any]
            The input data, either a single value or a list of values, to be
            transformed by the arithmetic operation.
        b: Any or list[Any]
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

        # Note that a is ensured to be a list by the parent class.

        # If b is a scalar, wrap it in a list for uniform processing.
        if not isinstance(b, (list, tuple)):
            b = [b]

        # Cycle the shorter list to match the length of the longer list.
        if len(a) < len(b):
            a = itertools.cycle(a)
        elif len(b) < len(a):
            b = itertools.cycle(b)

        # Apply the operation element-wise.
        return [self.op(x, y) for x, y in zip(a, b)]


class Add(ArithmeticOperationFeature):
    """Add a value to the input.
    
    This feature performs element-wise addition (+) to the input.

    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to add to the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Create a pipeline using `Add`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Add(b=5)
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
    >>> sum_feature = dt.Add(b=5)
    >>> pipeline = sum_feature(input_value)
    >>> pipeline.resolve()
    [6, 7, 8]

    """

    def __init__(
        self: Add,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the Add feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to add to the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature`.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.add, b=b, **kwargs)


class Subtract(ArithmeticOperationFeature):
    """Subtract a value from the input.

    This feature performs element-wise subtraction (-) from the input.
    
    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to subtract from the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Create a pipeline using `Subtract`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Subtract(b=2)
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
    >>> sub_feature = dt.Subtract(b=2)
    >>> pipeline = sub_feature(input_value)
    >>> pipeline.resolve()
    [-1, 0, 1]

    """

    def __init__(
        self: Subtract,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the Subtract feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to subtract from the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature`.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.sub, b=b, **kwargs)


class Multiply(ArithmeticOperationFeature):
    """Multiply the input by a value.

    This feature performs element-wise multiplication (*) of the input.
    
    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to multiply the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Multiply`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Multiply(b=5)
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
    >>> mul_feature = dt.Multiply(b=5)
    >>> pipeline = mul_feature(input_value)
    >>> pipeline.resolve()
    [5, 10, 15]

    """

    def __init__(
        self: Multiply,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the Multiply feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to multiply the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.mul, b=b, **kwargs)


class Divide(ArithmeticOperationFeature):
    """Divide the input with a value.

    This feature performs element-wise division (/) of the input.
    
    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to divide the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Divide`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Divide(b=5)
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
    >>> truediv_feature = dt.Divide(b=5)
    >>> pipeline = truediv_feature(input_value)
    >>> pipeline.resolve()
    [0.2 0.4 0.6]

    """

    def __init__(
        self: Divide,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the Divide feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to divide the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.truediv, b=b, **kwargs)


class FloorDivide(ArithmeticOperationFeature):
    """Divide the input with a value.

    This feature performs element-wise floor division (//) of the input.
    
    Floor division produces an integer result when both operands are integers,
    but truncates towards negative infinity when operands are floating-point
    numbers.
    
    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to floor-divide the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `FloorDivide`:

    >>> pipeline = dt.Value([-3, 3, 6]) >> dt.FloorDivide(b=5)
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
    >>> floordiv_feature = dt.FloorDivide(b=5)
    >>> pipeline = floordiv_feature(input_value)
    >>> pipeline.resolve()
    [-1, 0, 1]

    """

    def __init__(
        self: FloorDivide,
        b: PropertyLike[Any |list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the FloorDivide feature.

        Parameters
        ----------
        b: PropertyLike[any or list[Any]], optional
            The value to fllor-divide the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.floordiv, b=b, **kwargs)


class Power(ArithmeticOperationFeature):
    """Raise the input to a power.

    This feature performs element-wise power (**) of the input.

    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to take the power of the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Power`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Power(b=3)
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
    >>> pow_feature = dt.Power(b=3)
    >>> pipeline = pow_feature(input_value)
    >>> pipeline.resolve()
    [1, 8, 27]

    """

    def __init__(
        self: Power,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the Power feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to take the power of the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.pow, b=b, **kwargs)


class LessThan(ArithmeticOperationFeature):
    """Determine whether input is less than value.

    This feature performs element-wise comparison (<) of the input.

    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to compare (<) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `LessThan`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.LessThan(b=2)
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
    >>> lt_feature = dt.LessThan(b=2)
    >>> pipeline = lt_feature(input_value)
    >>> pipeline.resolve()
    [True, False, False]

    """

    def __init__(
        self: LessThan,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the LessThan feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to compare (<) with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.lt, b=b, **kwargs)


class LessThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is less than or equal to value.

    This feature performs element-wise comparison (<=) of the input.

    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to compare (<=) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `LessThanOrEquals`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.LessThanOrEquals(b=2)
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
    >>> le_feature = dt.LessThanOrEquals(b=2)
    >>> pipeline = le_feature(input_value)
    >>> pipeline.resolve()
    [True, True, False]

    """

    def __init__(
        self: LessThanOrEquals,
        b: PropertyLike[Any | list[Any]] = 0,
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

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.le, b=b, **kwargs)


LessThanOrEqual = LessThanOrEquals


class GreaterThan(ArithmeticOperationFeature):
    """Determine whether input is greater than value.

    This feature performs element-wise comparison (>) of the input.

    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to compare (>) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `GreaterThan`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.GreaterThan(b=2)
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
    >>> gt_feature = dt.GreaterThan(b=2)
    >>> pipeline = gt_feature(input_value)
    >>> pipeline.resolve()
    [False, False, True]

    """

    def __init__(
        self: GreaterThan,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the GreaterThan feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to compare (>) with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.gt, b=b, **kwargs)


class GreaterThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is greater than or equal to value.

    This feature performs element-wise comparison (>=) of the input.

    Parameters
    ----------
    b: PropertyLike[Any or list[Any]], optional
        The value to compare (<=) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `GreaterThanOrEquals`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.GreaterThanOrEquals(b=2)
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
    >>> ge_feature = dt.GreaterThanOrEquals(b=2)
    >>> pipeline = ge_feature(input_value)
    >>> pipeline.resolve()
    [False, True, True]

    """

    def __init__(
        self: GreaterThanOrEquals,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the GreaterThanOrEquals feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to compare (>=) with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.ge, b=b, **kwargs)


GreaterThanOrEqual = GreaterThanOrEquals


class Equals(ArithmeticOperationFeature):  # TODO
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
    b: PropertyLike[Any or list[Any]], optional
        The value to compare (==) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.
    
    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Equals`:

    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Equals(b=2)
    >>> pipeline.resolve()
    [False, True, False]
    
    Or:

    >>> input_values = [1, 2, 3]
    >>> eq_feature = dt.Equals(value=2)
    >>> output_values = eq_feature(input_values)
    >>> output_values
    [False, True, False]    
    
    These are the only correct ways to apply `Equals` in a pipeline.
    
    The following approaches are incorrect:
    
    Using `==` directly on a `Feature` instance does not work because `Feature`
    does not override `__eq__`:

    >>> pipeline = dt.Value([1, 2, 3]) == 2  # Incorrect
    >>> pipeline.resolve()
    AttributeError: 'bool' object has no attribute 'resolve'

    Similarly, directly calling `Equals` on an input feature immediately 
    evaluates the comparison, returning a boolean instead of a `Feature`:

    >>> pipeline = dt.Equals(b=2)(dt.Value([1, 2, 3]))  # Incorrect
    >>> pipeline.resolve()
    AttributeError: 'bool' object has no attribute 'resolve'

    """

    def __init__(
        self: Equals,
        b: PropertyLike[Any | list[Any]] = 0,
        **kwargs: Any,
    ):
        """Initialize the Equals feature.

        Parameters
        ----------
        b: PropertyLike[Any or list[Any]], optional
            The value to compare with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        # Backward compatibility with deprecated 'value' parameter taken care
        # of in ArithmeticOperationFeature

        super().__init__(operator.eq, b=b, **kwargs)


Equal = Equals


class Stack(Feature):  # TODO
    """Stack the input and the value.
    
    This feature combines the output of the input data (`inputs`) and the 
    value produced by the specified feature (`value`). The resulting output 
    is a list where the elements of the `inputs` and `value` are concatenated.

    If B is a feature, `Stack` can be visualized as:

    >>>   A >> Stack(B) = [*A(), *B()]

    It is equivalent to using the `&` operator:

    >>> A & B

    Parameters
    ----------
    value: PropertyLike[Any]
        The feature or data to stack with the input.
    **kwargs: Any
        Additional arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Set to `False`, indicating that this feature’s `.get()` method
        processes the entire input at once even if it is a list, rather than 
        distributing calls for each item of the list.

    Methods
    -------
    `get(inputs, value, **kwargs) -> list[Any]`
        Concatenate the inputs with the value.

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
    affect how it behaves when reused in chained pipelines. For example:

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
        inputs: Any | list[Any],
        value: Any | list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Concatenate the input with the value.

        It ensures that both the input (`inputs`) and the value (`value`) are 
        treated as lists before concatenation.

        Parameters
        ----------
        inputs: Any or list[Any]
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
        if not isinstance(inputs, list):
            inputs = [inputs]

        # Ensure the value is treated as a list.
        if not isinstance(value, list):
            value = [value]

        # Concatenate and return the lists.
        return [*inputs, *value]


class Arguments(Feature):  # TODO
    """A convenience container for pipeline arguments.

    `Arguments` allows dynamic control of pipeline behavior by providing a
    container for arguments that can be modified or overridden at runtime. This
    is particularly useful when working with parametrized pipelines, such as
    toggling behaviors based on whether an image is a label or a raw input.

    Methods
    -------
    `get(inputs, **kwargs) -> Any`
        It passes the inputs through unchanged, while allowing for property
        overrides.

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
        inputs: Any,
        **kwargs: Any,
    ) -> Any:

        """Return the inputs and allow property overrides.

        This method does not modify the inputs but provides a mechanism for
        overriding arguments dynamically during pipeline execution.

        Parameters
        ----------
        inputs: Any
            The inputs to be passed through unchanged.
        **kwargs: Any
            Key-value pairs for overriding pipeline properties.

        Returns
        -------
        Any
            The unchanged inputs.

        """

        return inputs


class Probability(StructuralFeature):  # TODO
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
        The probability (from 0 to 1) of resolving the feature.
    *args: Any
        Positional arguments passed to the parent `StructuralFeature` class.
    **kwargs: Any
        Additional keyword arguments passed to the parent `StructuralFeature` 
        class.

    Methods
    -------
    `get(inputs, probability, random_number, **kwargs) -> Any`
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

    Define inputs:

    >>> import numpy as np
    >>>
    >>> inputs = np.zeros((2, 3))

    Apply the feature:

    >>> probabilistic_feature.update()  # Update the random number
    >>> outputs = probabilistic_feature(inputs)

    With 70% probability, the output is:

    >>> outputs
    array([[2., 2., 2.],
        [2., 2., 2.]])

    With 30% probability, it remains:

    >>> outputs
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
        It can be updated using the `.update()` method.

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
        inputs: Any,
        probability: float,
        random_number: float,
        **kwargs: Any,
    ) -> Any:
        """Resolve the feature if random number is less than probability.

        Parameters
        ----------
        inputs: Any or list[Any]
            The inputs to process.
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
            The processed outputs. If the feature is resolved, this is the
            output of the feature; otherwise, it is the unchanged inputs.

        """

        if random_number < probability:
            outputs = self.feature.resolve(inputs, **kwargs)
            return outputs

        return inputs


class Repeat(StructuralFeature):  # TODO
    """Apply a feature multiple times.

    `Repeat` iteratively applies another feature, passing the output of each
    iteration as input to the next. This enables chained transformations,
    where each iteration builds upon the previous one. The number of
    repetitions is defined by `N`.

    Each iteration operates with its own set of properties, and the index of 
    the current iteration is accessible via `_ID`. `_ID` is extended to include
    the current iteration index, ensuring deterministic behavior when needed.

    The use of `Repeat`

    >>> dt.Repeat(A, 3)

    is equivalent to using the `^` operator:

    >>> A ^ 3
    
    Parameters
    ----------
    feature: Feature
        The feature to be repeated `N` times.
    N: int
        The number of times to apply the feature in sequence.
    **kwargs: Any

    Attributes
    ----------
    feature: Feature
        The feature to be applied sequentially `N` times.

    Methods
    -------
    `get(x, N, _ID, **kwargs) -> Any`
        It applies the feature `N` times in sequence, passing the output of
        each iteration as the input to the next.

    Examples
    --------
    >>> import deeptrack as dt
    
    Define an `Add` feature that adds `10` to its input:

    >>> add_ten_feature = dt.Add(value=10)

    Apply this feature 3 times using `Repeat`:

    >>> pipeline = dt.Repeat(add_ten_feature, N=3)

    Process an input list:

    >>> pipeline.resolve([1, 2, 3])
    [31, 32, 33]

    Alternative shorthand using `^` operator:

    >>> pipeline = add_ten_feature ^ 3
    >>> pipeline.resolve([1, 2, 3])
    [31, 32, 33]
    
    """

    feature: Feature

    def __init__(
        self: Repeat,
        feature: Feature,
        N: PropertyLike[int],
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

        super().__init__(N=N, **kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self: Repeat,
        x: Any,
        *,
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
        x: Any
            The input data to be transformed by the repeated feature.
        N: int
            The number of times to sequentially apply the feature, where each 
            iteration builds on the previous output.
        _ID: tuple[int, ...], optional
            A unique identifier for tracking the iteration index, ensuring 
            reproducibility, caching, and dynamic property updates.
            Defaults to ().
        **kwargs: Any
            Additional keyword arguments passed to the feature.

        Returns
        -------
        Any
            The output of the final iteration after `N` sequential applications 
            of the feature.

        """

        if not isinstance(N, int) or N < 0:
            raise ValueError("Using Repeat, N must be a non-negative integer.")

        for n in range(N):

            index = _ID + (n,)  # Track iteration index

            x = self.feature(
                x,
                _ID=index,
                replicate_index=index,  # Legacy property
            )

        return x


class Combine(StructuralFeature):  # TODO
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
    `get(inputs, **kwargs) -> list[Any]`
        Resolves each feature in the `features` list on the inputs and returns
        their results as a list.

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
        inputs: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Resolve each feature in the `features` list on the inputs.

        Parameters
        ----------
        image: Any
            The input or list of inputs to process.
        **kwargs: Any
            Additional arguments passed to each feature's `resolve` method.

        Returns
        -------
        list[Any]
            A list containing the outputs of each feature applied to the input.

        """

        return [f(inputs, **kwargs) for f in self.features]


class Slice(Feature):  # TODO
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
    `get(inputs, slices, **kwargs) -> array or list[array]`
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

    Using `Slice` for dynamic slicing (necessary when slices depend on computed
    properties):

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
        array: ArrayLike[Any] | list[ArrayLike[Any]],
        slices: slice | tuple[int | slice | Ellipsis, ...],
        **kwargs: Any,
    ) -> ArrayLike[Any] | list[ArrayLike[Any]]:
        """Apply the specified slices to the input image.

        Parameters
        ----------
        image: array or list[array]
            The input array(s) to be sliced.
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
            The sliced array(s).

        """

        try:
            # Convert slices to a tuple if possible
            slices = tuple(slices)
        except ValueError:
            # Leave slices as is if conversion fails
            pass

        return array[slices]


class Bind(StructuralFeature):  # TODO
    """Bind a feature with property arguments.

    When the feature is resolved, the kwarg arguments are passed to the child 
    feature. Thus, this feature allows passing additional keyword arguments 
    (`kwargs`) to a child feature when it is resolved. These properties can 
    dynamically control the behavior of the child feature.

    Parameters
    ----------
    feature: Feature
        The child feature.
    **kwargs: Any
        Properties to send to child.

    Methods
    -------
    `get(inputs, **kwargs) -> Any`
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
        inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve the child feature with the dynamically provided arguments.

        Parameters
        ----------
        inputs: Any
            The input data to process.
        **kwargs: Any
            Properties or arguments to pass to the child feature during
            resolution.

        Returns
        -------
        Any
            The result of resolving the child feature with the provided
            arguments.

        """

        return self.feature.resolve(inputs, **kwargs)


BindResolve = Bind


class BindUpdate(StructuralFeature):  # DEPRECATED  # TODO
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
    `get(inputs, **kwargs) -> Any`
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

        warnings.warn(
            "BindUpdate is deprecated and may be removed in a future release. "
            "The current implementation is not guaranteed to be exactly "
            "equivalent to prior implementations. "
            "Please use Bind instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(**kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self: Feature,
        inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve the child feature with the provided arguments.

        Parameters
        ----------
        inputs: Any
            The input data to process.
        **kwargs: Any
            Properties or arguments to pass to the child feature during 
            resolution.

        Returns
        -------
        Any
            The result of resolving the child feature with the provided 
            arguments.

        """

        return self.feature.resolve(inputs, **kwargs)


class ConditionalSetProperty(StructuralFeature):  # DEPRECATED  # TODO
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
    condition: PropertyLike[str or bool] or None, optional
        Either a boolean value (`True`, `False`) or the name of a boolean 
        property in the feature’s property dictionary. If the condition 
        evaluates to `True`, the specified properties are applied.
    **kwargs: Any
        The properties to be applied to the child feature if `condition` is 
        `True`.

    Methods
    -------
    `get(inputs, condition, **kwargs) -> Any`
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
        condition: PropertyLike[str or bool] or None, optional
            A boolean value or the name of a boolean property in the feature's 
            property dictionary. If the condition evaluates to `True`, the 
            specified properties are applied.
        **kwargs: Any
            Properties to apply to the child feature if the condition is 
            `True`.

        """

        warnings.warn(
            "ConditionalSetFeature is deprecated and may be removed in a "
            "future release. Please use Arguments instead when possible.",
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(condition, str):
            kwargs.setdefault(condition, True)

        super().__init__(condition=condition, **kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self: ConditionalSetProperty,
        inputs: Any,
        condition: str | bool,
        **kwargs: Any,
    ) -> Any:
        """Resolve the child, conditionally applying specified properties.

        Parameters
        ----------
        inputs: Any
            The input data to process.
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

        return self.feature(inputs)


class ConditionalSetFeature(StructuralFeature):  # DEPRECATED  # TODO
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

        warnings.warn(
            "ConditionalSetFeature is deprecated and may be removed in a "
            "future release. Please use Arguments instead when possible.",
            DeprecationWarning,
            stacklevel=2,
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
        inputs: Any,
        *,
        condition: str | bool,
        **kwargs: Any,
    ):
        """Resolve the appropriate feature based on the condition.

        Parameters
        ----------
        inputs: Any
            The inputs to process.
        condition: str or bool
            The name of the conditional property or a boolean value. If a 
            string is provided, it is looked up in `kwargs` to get the actual 
            boolean value.
        **kwargs:: Any
            Additional keyword arguments to pass to the resolved feature.

        Returns
        -------
        Any
            The processed data after resolving the appropriate feature. If 
            neither `on_true` nor `on_false` is provided for the corresponding 
            condition, the input is returned unchanged.

        """

        # Evaluate the condition.
        _condition = condition
        if isinstance(condition, str):
            _condition = kwargs.get(condition, False)

        # Resolve the appropriate feature.
        if _condition and self.on_true:
            return self.on_true(inputs)
        if not _condition and self.on_false:
            return self.on_false(inputs)
        return inputs


class Lambda(Feature):  # TODO
    """Apply a user-defined function to the input.

    This feature allows applying a custom function to individual inputs in the
    input pipeline. The `function` parameter must be wrapped in an outer
    function that can depend on other properties of the pipeline. 
    The inner function processes a single input.

    Parameters
    ----------
    function: Callable[..., Callable[[Any], Any]]
        A callable that produces a function. The outer function can accept 
        additional arguments from the pipeline, while the inner function 
        operates on a single input.
    **kwargs: dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(inputs, function, **kwargs) -> Any`
        Applies the custom function to the inputs.

    Examples
    --------
    >>> import deeptrack as dt

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
        inputs: Any,
        function: Callable[[Any], Any],
        **kwargs: Any,
    ) -> Any:
        """Apply the custom function to the input.

        This method applies a user-defined function to transform the input. The
        function should be a callable that takes an input and returns a
        modified version of it.

        Parameters
        ----------
        inputs: Any
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

        return function(inputs)


class Merge(Feature):  # TODO
    """Apply a custom function to a list of inputs.

    This feature allows applying a user-defined function to a list of inputs. 
    The `function` parameter must be a callable that returns another function, 
    where:
      - The outer function can depend on other properties in the pipeline.
      - The inner function takes a list of inputs and returns a single outputs
      or a list of outputs.
    
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
        Set to `False`, indicating that this feature’s `.get()` method
        processes the entire input at once even if it is a list, rather than 
        distributing calls for each item of the list.

    Methods
    -------
    `get(list_of_inputs, function, **kwargs) -> Any or list[Any]`
        Applies the custom function to the list of inputs.

    Examples
    --------
    >>> import deeptrack as dt

    Define a merge function that averages multiple images:

    >>> import numpy as np
    >>>
    >>> def merge_function_factory():
    ...     def merge_function(images):
    ...         return np.mean(np.stack(images), axis=0)
    ...     return merge_function

    Create a Merge feature:

    >>> merge_feature = dt.Merge(function=merge_function_factory)

    Create some images:

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
        function: Callable[..., Callable[[list[Any]], Any | list[Any]]],
        **kwargs: Any,
    ):
        """Initialize the Merge feature.

        Parameters
        ----------
        function: Callable[..., Callable[[list[Any]], Any or list[Any]]
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
        list_of_inputs: list[Any],
        function: Callable[[list[Any]], Any | list[Any]],
        **kwargs: Any,
    ) -> Any | list[Any]:
        """Apply the custom function to a list of inputs.

        Parameters
        ----------
        list_of_inputs: list[Any]
            A list of inputs to be processed by the function.
        function: Callable[[list[Any]], Any or list[Any]]
            The function that processes the list of inputs and returns either a
            single transformed input or a list of transformed inputs.
        **kwargs: Any
            Additional arguments (unused in this implementation).

        Returns
        -------
        Any or list[Any]
            The processed inputs after applying the function.

        """

        return function(list_of_inputs)


class OneOf(Feature):  # TODO
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
    key: int or None, optional
        The index of the feature to resolve from the collection. If not 
        provided, a feature is selected randomly at each execution.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Set to `False`, indicating that this feature’s `.get()` method
        processes the entire input at once even if it is a list, rather than 
        distributing calls for each item of the list.

    Methods
    -------
    `_process_properties(propertydict) -> dict`
        It processes the properties to determine the selected feature index.
    `get(image, key, _ID, **kwargs) -> Any`
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
    >>> output_image  # The output depends on the randomly selected feature

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

    collection: tuple[Feature, ...]

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
        inputs: Any,
        key: int,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Apply the selected feature to the input image.

        Parameters
        ----------
        inputs: Any
            The input data to process.
        key: int
            The index of the feature to apply from the collection.
        _ID: tuple[int, ...], optional
            A unique identifier for caching and parallel processing.
        **kwargs: Any
            Additional parameters passed to the selected feature.

        Returns
        -------
        Any
            The output of the selected feature applied to the input.

        """

        return self.collection[key](inputs, _ID=_ID)


class OneOfDict(Feature):  # TODO
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
        Set to `False`, indicating that this feature’s `.get()` method
        processes the entire input at once even if it is a list, rather than 
        distributing calls for each item of the list.

    Methods
    -------
    `_process_properties(propertydict) -> dict`
        It determines which feature to use based on `key`.
    `get(inputs, key, _ID, **kwargs) -> Any`
        It resolves the selected feature and applies it to the input.
   
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
    >>> output_image  # The output depends on the randomly selected feature

    Potentially select a different feature:

    >>> output_image = one_of_dict_feature.new(input_image)
    >>> output_image

    Use a specific key to apply a predefined feature:

    >>> controlled_feature = dt.OneOfDict(features_dict, key="add")
    >>> output_image = controlled_feature(input_image)
    >>> output_image
    array([11, 12, 13])

    """

    __distributed__: bool = False

    collection: tuple[Feature, ...]

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
            propertydict["key"] = \
                np.random.choice(list(self.collection.keys()))

        return propertydict

    def get(
        self: Feature,
        inputs: Any,
        key: Any,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    )-> Any:
        """Resolve the selected feature and apply it to the input.

        Parameters
        ----------
        inputs: Any
            The input data to be processed.
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

        return self.collection[key](inputs, _ID=_ID)


class LoadImage(Feature):  # TODO
    """Load an image from disk and preprocess it.

    `LoadImage` loads an image file using multiple fallback file readers
    (`ImageIO`, `NumPy`, `Pillow`, and `OpenCV`) until a suitable reader is
    found. The image can be optionally converted to grayscale, reshaped to
    ensure a minimum number of dimensions, or treated as a list of images if
    multiple paths are provided.

    Parameters
    ----------
    path: PropertyLike[str or list[str]]
        The path(s) to the image(s) to load. Can be a single string or a list
        of strings.
    load_options: PropertyLike[dict[str, Any]], optional
        Additional options passed to the file reader. Defaults to `None`.
    as_list: PropertyLike[bool], optional
        If `True`, the first dimension of the image will be treated as a list.
        Defaults to `False`.
    ndim: PropertyLike[int], optional
        Ensures the image has at least this many dimensions. Defaults to `3`.
    to_grayscale: PropertyLike[bool], optional
        If `True`, converts the image to grayscale. Defaults to `False`.
    get_one_random: PropertyLike[bool], optional
        If `True`, extracts a single random image from a stack of images. Only
        used when `as_list` is `True`. Defaults to `False`.

    Attributes
    ----------
    __distributed__: bool
        Set to `False`, indicating that this feature’s `.get()` method
        processes the entire input at once even if it is a list, rather than 
        distributing calls for each item of the list.

    Methods
    -------
    `get(...) -> array or tensor or list of arrays/tensors`
        Load the image(s) from disk and process them.

    Raises
    ------
    IOError
        If no file reader could parse the file or the file does not exist.

    Notes
    ----
    By default, `LoadImage` returns a NumPy array. If you want the output as
    a PyTorch tensor, convert the feature to torch by calling `.torch()` before
    resolving.

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
    (100, 100, 3, 1)

    Load an image as a PyTorch tensor by setting the backend of the feature:

    >>> load_image_feature = dt.LoadImage(path=temp_file.name)
    >>> load_image_feature.torch()
    >>> loaded_image = load_image_feature.resolve()
    >>> type(loaded_image)
    <class 'torch.Tensor'>

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
            OpenCV, `allow_pickle` for NumPy). Defaults to `None`.
        as_list: PropertyLike[bool], optional
            If `True`, treats the first dimension of the image as a list of
            images. Defaults to `False`.
        ndim: PropertyLike[int], optional
            Ensures the image has at least this many dimensions. If the loaded
            image has fewer dimensions, extra dimensions are added. Defaults to
            `3`.
        to_grayscale: PropertyLike[bool], optional
            If `True`, converts the image to grayscale. Defaults to `False`.
        get_one_random: PropertyLike[bool], optional
            If `True`, selects a single random image from a stack when
            `as_list=True`. Defaults to `False`.
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
        *_: Any,
        path: str | list[str],
        load_options: dict[str, Any] | None,
        ndim: int,
        to_grayscale: bool,
        as_list: bool,
        get_one_random: bool,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
        """Load and process an image or a list of images from disk.

        This method attempts to load an image using multiple file readers
        (`ImageIO`, `NumPy`, `Pillow`, and `OpenCV`) until a valid format is
        found. It supports optional processing steps such as ensuring a minimum
        number of dimensions, grayscale conversion, and treating multi-frame
        images as lists.

        The output is returned as a NumPy array by default. If `as_list=True`,
        the result is a Python list of arrays. If the backend of the feature is
        `"torch"`, the image is returned as a PyTorch tensor.

        Parameters
        ----------
        path: str or list[str]
            The file path(s) to the image(s) to be loaded. A single string
            loads one image, while a list of paths loads multiple images.
        load_options: dict of str to Any, optional
            Additional options passed to the file reader (e.g., `allow_pickle`
            for NumPy, `mode` for OpenCV). Defaults to `None`.
        ndim: int
            Ensures the image has at least this many dimensions. If the loaded
            image has fewer dimensions, extra dimensions are added. Defaults to
            `3`.
        to_grayscale: bool
            If `True`, converts the image to grayscale. Defaults to `False`.
        as_list: bool
            If `True`, treats the first dimension as a list of images instead
            of stacking them into a NumPy array. Defaults to `False`.
        get_one_random: bool
            If `True`, selects a single random image from a multi-frame stack
            when `as_list=True`. Defaults to `False`.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        array or list of arrays
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

                    image = [
                        PIL.Image.open(file, **load_options) for file in path
                    ]
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
                warnings.warn(
                    "Non-rgb image, ignoring to_grayscale",
                    UserWarning,
                    stacklevel=2,
                )

        # Ensure the image has at least `ndim` dimensions.
        if not isinstance(image, list) and ndim:
            while image.ndim < ndim:
                image = np.expand_dims(image, axis=-1)

        # Convert to PyTorch tensor if needed.
        if self.get_backend() == "torch":

            # Convert to stack if needed.
            if isinstance(image, list):
                image = np.stack(image, axis=0)

            image = torch.from_numpy(image)

        return image


class AsType(Feature):  # TODO
    """Convert the data type of arrays.

    `Astype` changes the data type (`dtype`) of input arrays to a specified
    type. The accepted types are standard NumPy or PyTorch data types (e.g.,
    `"float64"`, `"int32"`, `"uint8"`, `"int8"`, and `"torch.float32"`).

    Parameters
    ----------
    dtype: PropertyLike[str], optional
        The desired data type for the image. Defaults to `"float64"`.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image, dtype, **kwargs) -> array`
        Convert the data type of the input image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input array:

    >>> import numpy as np
    >>>
    >>> input_image = np.array([1.5, 2.5, 3.5])

    Apply an AsType feature to convert to "`int32"`:

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
        """Initialize the AsType feature.

        Parameters
        ----------
        dtype: PropertyLike[str], optional
            The desired data type for the image. Defaults to `"float64"`.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(dtype=dtype, **kwargs)

    def get(
        self: Feature,
        image: np.ndarray | torch.Tensor,
        dtype: str,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
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
            NumPy array or a PyTorch tensor.

        """

        if apc.is_torch_array(image):
            # Mapping from string to torch dtype
            torch_dtypes = {
                "float64": torch.float64,
                "double": torch.float64,
                "float32": torch.float32,
                "float": torch.float32,
                "float16": torch.float16,
                "half": torch.float16,
                "int64": torch.int64,
                "int32": torch.int32,
                "int16": torch.int16,
                "int8": torch.int8,
                "uint8": torch.uint8,
                "bool": torch.bool,
                "complex64": torch.complex64,
                "complex128": torch.complex128,
            }

            # Ensure `"torch.float32"` and `"float32"` are treated the same by
            # removing the `torch.` prefix if present
            dtype_str = str(dtype).replace("torch.", "")
            torch_dtype = torch_dtypes.get(dtype_str)

            if torch_dtype is None:
                raise ValueError(
                    f"Unsupported dtype for torch.Tensor: {dtype}"
                )

            return image.to(dtype=torch_dtype)

        return image.astype(dtype)


class ChannelFirst2d(Feature):  # DEPRECATED  # TODO
    """Convert an image to a channel-first format.

    This feature rearranges the axes of a 3D image so that the specified axis
    (e.g., channel axis) is moved to the first position. If the input image is
    2D, it adds a new dimension at the first index, effectively treating the 2D
    image as a single-channel image.

    Parameters
    ----------
    axis: int, optional
        The axis to move to the first position. Defaults to `-1` (last axis),
        which is typically the channel axis for NumPy arrays.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image, axis, **kwargs) -> array`
        It rearranges the axes of an image to channel-first format.

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
        axis: PropertyLike[int] = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize the ChannelFirst2d feature.

        Parameters
        ----------
        axis: int, optional
            The axis to move to the first position.
            Defaults to `-1` (last axis).
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        warnings.warn(
            "ChannelFirst2d is deprecated and may be removed in a "
            "future release. The current implementation is not guaranteed "
            "to be exactly equivalent to prior implementations.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Feature,
        array: np.ndarray | torch.Tensor,
        axis: int = -1,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Rearrange the axes of an image to channel-first format.

        Rearrange the axes of a 3D image to channel-first format or add a
        channel dimension to a 2D image.

        Parameters
        ----------
        image: array
            The input image to process. Can be 2D or 3D.
        axis: int
            The axis to move to the first position (for 3D images).
            For 2D images, this argument does nothing.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array
            The processed image in channel-first format.

        Raises
        ------
        ValueError
            If the input image is neither 2D nor 3D.

        """

        # Raise error if not 2D or 3D.
        ndim = array.ndim
        if ndim not in (2, 3):
            raise ValueError("ChannelFirst2d only supports 2D or 3D images. "
                             f"Received {ndim}D image.")

        # Add a new dimension for 2D images.
        if ndim == 2:
            if apc.is_torch_array(array):
                array = array.unsqueeze(0)
            else:
                array[None]

        # Move axis for 3D images.
        else:
            if apc.is_torch_array(array):
                axis = ndim + axis if axis < 0 else axis
                dims = [axis] + [i for i in range(ndim) if i != axis]
                array = array.permute(*dims)
            else:
                array = xp.moveaxis(array, axis, 0)

        return array


class Store(Feature):  # TODO
    """Store the output of a feature for reuse.

    `Store` evaluates a given feature and stores its output in an internal
    dictionary. Subsequent calls with the same key will return the stored value
    unless the `replace` parameter is set to `True`. This enables caching and
    reuse of computed feature outputs.

    Parameters
    ----------
    feature: Feature
        The feature to evaluate and store.
    key: Any
        The key used to identify the stored output.
    replace: PropertyLike[bool], optional
        If `True`, replaces the stored value with the current computation.
        Defaults to `False`.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Always `False` for `Store`, as it handles caching locally.
    _store: dict[Any, Any]
        A dictionary used to store the outputs of the evaluated feature.

    Methods
    -------
    `get(*_, key, replace, **kwargs) -> Any`
        Evaluate and store the feature output, or return the cached result.

    Examples
    --------
    >>> import deeptrack as dt

    Create a `Store` feature with a key:

    >>> import numpy as np
    >>>
    >>> value_feature = dt.Value(lambda: np.random.rand())
    >>> store_feature = dt.Store(feature=value_feature, key="example")

    Retrieve and store the value:

    >>> output = store_feature(None, key="example", replace=False)

    Retrieve the stored value without recomputing:

    >>> value_feature.update()
    >>> cached_output = store_feature(None, key="example", replace=False)
    >>> print(cached_output == output)
    True

    >>> print(cached_output == value_feature())
    False

    Retrieve the stored value recomputing:

    >>> value_feature.update()
    >>> cached_output = store_feature(None, key="example", replace=True)
    >>> print(cached_output == output)
    False

    >>> print(cached_output == value_feature())
    True

    """

    __distributed__: bool = False

    def __init__(
        self: Store,
        feature: Feature,
        key: Any,
        replace: PropertyLike[bool] = False,
        **kwargs: Any,
    ):
        """Initialize the Store feature.

        Parameters
        ----------
        feature: Feature
            The feature to evaluate and store.
        key: Any
            The key used to identify the stored output.
        replace: PropertyLike[bool], optional
            If `True`, replaces the stored value with a new computation.
            Defaults to `False`.
        **kwargs:: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(key=key, replace=replace, **kwargs)
        self.feature = self.add_feature(feature, **kwargs)
        self._store: dict[Any, Image] = {}

    def get(
        self: Store,
        *_: Any,
        key: Any,
        replace: bool,
        **kwargs: Any,
    ) -> Any:
        """Evaluate and store the feature output, or return the cached result.

        Parameters
        ----------
        *_: Any
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
        if replace or not key in self._store:
            self._store[key] = self.feature()

        # TODO TBE
        ## Return the stored or newly computed result
        #if self._wrap_array_with_image:
        #    return Image(self._store[key], copy=False)

        return self._store[key]


class Squeeze(Feature):  # TODO
    """Squeeze the input image to the smallest possible dimension.

    `Squeeze` removes axes of size 1 from the input image. By default, it 
    removes all singleton dimensions. If a specific axis or axes are specified, 
    only those axes are squeezed.

    Parameters
    ----------
    axis: int or tuple[int, ...], optional
        The axis or axes to squeeze. Defaults to `None`, squeezing all axes.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image, axis, **kwargs) -> array`
        Squeeze the input array by removing singleton dimensions. The input and
        output arrays can be a NumPy array or a PyTorch tensor.

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
        image: np.ndarray | torch.Tensor,
        axis: int | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Squeeze the input image by removing singleton dimensions.

        Parameters
        ----------
        image: array or tensor
            The input image to process. The input array can be a NumPy array or
            a PyTorch tensor.
        axis: int or tuple[int, ...], optional
            The axis or axes to squeeze. Defaults to `None`, which squeezes all
            singleton axes.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array or tensor
            The squeezed array with reduced dimensions. The output array can be
            a NumPy array or a PyTorch tensor.

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


class Unsqueeze(Feature):  # TODO
    """Unsqueeze the input image to the smallest possible dimension.

    This feature adds new singleton dimensions to the input image at the 
    specified axis or axes. If no axis is specified, it defaults to adding 
    a singleton dimension at the last axis.

    Parameters
    ----------
    axis: int or tuple[int, ...], optional
        The axis or axes where new singleton dimensions should be added.
        Defaults to `None`, which adds a singleton dimension at the last axis.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image, axis, **kwargs) -> array or tensor`
        Add singleton dimensions to the input image. The input and output
        arrays can be a NumPy array or a PyTorch tensor.

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
            The axis or axes where new singleton dimensions should be added. 
            Defaults to -1, which adds a singleton dimension at the last axis.
        **kwargs:: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Unsqueeze,
        image: np.ndarray | torch.Tensor,
        axis: int | tuple[int, ...] | None = -1,
        **kwargs: Any,

    ) -> np.ndarray | torch.Tensor:
        """Add singleton dimensions to the input image.

        Parameters
        ----------
        image: array
            The input image to process. The input array can be a NumPy array or
            a PyTorch tensor.
        axis: int or tuple[int, ...], optional
            The axis or axes where new singleton dimensions should be added. 
            It defaults to -1, which adds a singleton dimension at the last
            axis.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array or tensor
            The input image with the specified singleton dimensions added. The
            output array can be a NumPy array, or a PyTorch tensor.

        """

        if apc.is_torch_array(image):
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                image = image.unsqueeze(ax)
            return image

        return xp.expand_dims(image, axis=axis)


ExpandDims = Unsqueeze


class MoveAxis(Feature):  # TODO
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
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image, source, destination, **kwargs) -> array or tensor`
        Move the specified axis of the input image to a new position. The input
        and output can be NumPy arrays or PyTorch tensors.

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
        image: np.ndarray | torch.Tensor,
        source: int,
        destination: int,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Move the specified axis of the input image to a new position.

        Parameters
        ----------
        image: array or tensor
            The input image to process. The input can be a NumPy array or a
            PyTorch tensor.
        source: int
            The axis to move.
        destination: int
            The destination position of the axis.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array or tensor
            The input image with the specified axis moved to the destination.
            The output can be a NumPy array or a PyTorch tensor.

        """

        if apc.is_torch_array(image):
            axes = list(range(image.ndim))
            axis = axes.pop(source)
            axes.insert(destination, axis)
            return image.permute(*axes)

        return xp.moveaxis(image, source, destination)


class Transpose(Feature):  # TODO
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
    `get(image, axes, **kwargs) -> array or tensor`
        Transpose the axes of the input image(s). The input and output can be
        NumPy arrays or PyTorch tensors.

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
        image: np.ndarray | torch.Tensor,
        axes: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Transpose the axes of the input image.

        Parameters
        ----------
        image: array or tenor
            The input image to process. The input can be a NumPy array or a
            PyTorch tensor.
        axes: tuple[int, ...], optional
            A tuple specifying the permutation of the axes. If `None`, the 
            axes are reversed by default.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array or tensor
            The transposed image with rearranged axes. The output can be a
            NumPy array or a PyTorch tensor.

        """

        return xp.transpose(image, axes)


Permute = Transpose


class OneHot(Feature):  # TODO
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
    `get(image, num_classes, **kwargs) -> array or tensor`
        Convert the input array of class labels into a one-hot encoded array.
        The input and output can be NumPy arrays or PyTorch tensors.

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
        image: np.ndarray | torch.Tensor,
        num_classes: int,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Convert the input array of labels into a one-hot encoded array.

        Parameters
        ----------
        image: array or tensor
            The input array of class labels. The last dimension should contain 
            integers representing class indices. The input can be a NumPy array
            or a PyTorch tensor.
        num_classes: int
            The total number of classes for the one-hot encoding.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        array or tensor
            The one-hot encoded array. The last dimension is replaced with 
            one-hot vectors of length `num_classes`. The output can be a NumPy
            array or a PyTorch tensor. In all cases, it is of data type float32
            (e.g., np.float32 or torch.float32).

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


class TakeProperties(Feature):  # TODO
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
    **kwargs: Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Always `False` for `TakeProperties`, as it processes sequentially.
    __list_merge_strategy__: int
        Specifies how lists of properties are merged. Set to
        `MERGE_STRATEGY_APPEND` to append values to the result list.

    Methods
    -------
    `get(image, names, **kwargs) -> array or tensor or tuple of arrays/tensors`
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
        *names: PropertyLike[str],
        **kwargs: Any,
    ):
        """Initialize the TakeProperties feature.

        Parameters
        ----------
        feature: Feature
            The feature from which to extract properties.
        *names: PropertyLike[str]
            One or more names of the properties to extract.
        **kwargs: Any, optional
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(names=names, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Feature,
        image: np.ndarray | torch.Tensor,
        names: tuple[str, ...],
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> (
        np.ndarray
        | torch.Tensor
        | tuple[np.ndarray, ...]
        | tuple[torch.Tensor, ...]
    ):
        """Extract the specified properties from the feature pipeline.

        This method retrieves the values of the specified properties from the
        feature's dependency graph and returns them as NumPy arrays.

        Parameters
        ----------
        image: array or tensor
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
        array or tensor or tuple of arrays or tensors
            If a single property name is provided, a NumPy array or a PyTorch
            tensor containing the property values is returned. If multiple
            property names are provided, a tuple of NumPy arrays or PyTorch
            tensors is returned, where each array/tensor corresponds to a property.

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

        # Convert the results to tuple.
        res = tuple([res[name] for name in names])

        # Return a single array if only one property name is specified.
        if len(res) == 1:
            res = res[0]

        return res
