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

- `Feature`:

    Base class for all features in DeepTrack2. Represents a modular data 
    transformation with properties and methods for customization.

- `StructuralFeature`:

    A specialized feature for organizing and managing hierarchical or logical 
    structures in the pipeline.

- `Value`:

    Stores a constant value as a feature. Useful for passing parameters through
    the pipeline.

- `Chain`:

    Sequentially applies multiple features to the input data (>>).

- `DummyFeature`:

    A no-op feature that passes the input data unchanged.

- `ArithmeticOperationFeature`:

    A parent class for features performing arithmetic operations like addition,
    subtraction, multiplication, and division.

Functions:

- `propagate_data_to_dependencies`:

    def propagate_data_to_dependencies(
        feature: Feature,
        **kwargs: Any
    ) -> None

    Propagates data to all dependencies of a feature, updating their properties
    with the provided values.

- `merge_features`:

    def merge_features(
        features: list[Feature],
        merge_strategy: int = MERGE_STRATEGY_OVERRIDE,
    ) -> Feature

    Merges multiple features into a single feature using the specified merge
    strategy.

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
from typing import Any, Callable, Iterable

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pint import Quantity
from scipy.spatial.distance import cdist


from deeptrack import units
from deeptrack.backend import config
from deeptrack.backend.core import DeepTrackNode
from deeptrack.backend.units import ConversionTable, create_context
from deeptrack.image import Image
from deeptrack.properties import PropertyDict
from deeptrack.sources import SourceItem
from deeptrack.types import ArrayLike, PropertyLike

MERGE_STRATEGY_OVERRIDE: int = 0
MERGE_STRATEGY_APPEND: int = 1


class Feature(DeepTrackNode):
    """Base feature class.

    Features define the image generation process. All features operate on lists 
    of images. Most features, such as noise, apply some tranformation to all 
    images in the list. This transformation can be additive, such as adding 
    some Gaussian noise or a background illumination, or non-additive, such as 
    introducing Poisson noise or performing a low-pass filter. This 
    transformation is defined by the method `get(image, **kwargs)`, which all 
    implementations of the class `Feature` need to define.

    Whenever a Feature is initiated, all keyword arguments passed to the
    constructor will be wrapped as a `Property`, and stored in the `properties` 
    attribute as a `PropertyDict`. When a Feature is resolved, the current 
    value of each property is sent as input to the get method.

    Parameters
    ----------
    _input: np.ndarray or list of np.ndarray or Image or list of Image, 
        optional.
        A list of np.ndarray or `DeepTrackNode` objects or a single np.ndarray 
        or an `Image` object representing the input data for the feature. This
        parameter specifies what the feature will process. If left empty, no 
        initial input is set.
    **kwargs: dict of str and Any
        Keyword arguments to configure the feature. Each keyword argument is 
        wrapped as a `Property` and added to the `properties` attribute, 
        allowing dynamic sampling and parameterization during the feature's 
        execution.

    Attributes
    ----------
    properties: PropertyDict
        A dictionary containing all keyword arguments passed to the 
        constructor, wrapped as instances of `Property`. The properties can 
        dynamically sample values during pipeline execution. A sampled copy of
        this dictionary is passed to the `get` function and appended to the 
        properties of the output image.
    __list_merge_strategy__: int
        Specifies how the output of `.get(image, **kwargs)` is merged with the 
        input list. Options include:
        - `MERGE_STRATEGY_OVERRIDE` (0, default): The input list is replaced by
        the new list.
        - `MERGE_STRATEGY_APPEND` (1): The new list is appended to the end of 
        the input list.
    __distributed__: bool
        Determines whether `.get(image, **kwargs)` is applied to each element 
        of the input list independently (`__distributed__ = True`) or to the 
        list as a whole (`__distributed__ = False`).
    __property_memorability__: int
        Specifies whether to store the feature’s properties in the output 
        image. Properties with a memorability value of `1` or lower are stored
        by default.
    __conversion_table__: ConversionTable
        Defines the unit conversions used by the feature to convert its 
        properties into the desired units.
    __gpu_compatible__: bool
        Indicates whether the feature can use GPU acceleration. When enabled, 
        GPU execution is triggered based on input size or backend settings.

    Methods
    -------
    `get(image: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> Image | list[Image]`
        Abstract method that defines how the feature transforms the input.
    `__call__(image_list: np.ndarray | list[np.ndarray] | Image | list[Image] | None = None, _ID: tuple[int, ...] = (), **kwargs: Any) -> Any`
        Executes the feature or pipeline on the input and applies property 
        overrides from `kwargs`.
    `store_properties(x: bool = True, recursive: bool = True) -> None`
        Controls whether the properties are stored in the output `Image` object.
    `torch(dtype: torch.dtype | None = None, device: torch.device | None = None, permute_mode: str = "never") -> 'Feature'`
        Converts the feature into a PyTorch-compatible feature.
    `batch(batch_size: int = 32) -> tuple | list[Image]`
        Batches the feature for repeated execution.
    `action(_ID: tuple[int, ...] = ()) -> Image | list[Image]`
        Core logic to create or transform the image.
    `__use_gpu__(inp: np.ndarrary | Image, **_: Any) -> bool`
        Determines if the feature should use the GPU.
    `update(**global_arguments: Any) -> Feature`
        Refreshes the feature to create a new image.
    `add_feature(feature: Feature) -> Feature`
        Adds a feature to the dependency graph of this one.
    `seed(_ID: tuple[int, ...] = ()) -> None`
        Sets the random seed for the feature, ensuring deterministic behavior.
    `bind_arguments(arguments: Feature) -> Feature`
        Binds another feature’s properties as arguments to this feature.
    `_normalize(**properties: dict[str, Any]) -> dict[str, Any]`
        Normalizes the properties of the feature.
    `plot(input_image: np.ndarray | list[np.ndarray] | Image | list[Image] | None = None, resolve_kwargs: dict | None = None, interval: float | None = None, **kwargs) -> Any`
        Visualizes the output of the feature.
    `_process_properties(propertydict: dict[str, Any]) -> dict[str, Any]`
        Preprocesses the input properties before calling the `get` method.
    `_activate_sources(x: Any) -> None`
        Activates sources in the input data.
    `__getattr__(key: str) -> Any`
        Custom attribute access for the Feature class.
    `__iter__() -> Iterable`
        Iterates over the feature.
    `__next__() -> Any`
        Returns the next element in the feature.
    `__rshift__(other: Any) -> Feature`
        Allows chaining of features.
    `__rrshift__(other: Any) -> Feature`
        Allows right chaining of features.
    `__add__(other: Any) -> Feature`
        Overrides add operator.
    `__radd__(other: Any) -> Feature`
        Overrides right add operator.
    `__sub__(other: Any) -> Feature`
        Overrides subtraction operator.
    `__rsub__(other: Any) -> Feature`
        Overrides right subtraction operator.
    `__mul__(other: Any) -> Feature`
        Overrides multiplication operator.
    `__rmul__(other: Any) -> Feature`
        Overrides right multiplication operator.
    `__truediv__(other: Any) -> Feature`
        Overrides division operator.
    `__rtruediv__(other: Any) -> Feature`
        Overrides right division operator.
    `__floordiv__(other: Any) -> Feature`
        Overrides floor division operator.
    `__rfloordiv__(other: Any) -> Feature`
        Overrides right floor division operator.
    `__pow__(other: Any) -> Feature`
        Overrides power operator.
    `__rpow__(other: Any) -> Feature`
        Overrides right power operator.
    `__gt__(other: Any) -> Feature`
        Overrides greater than operator.
    `__rgt__(other: Any) -> Feature`
        Overrides right greater than operator.
    `__lt__(other: Any) -> Feature`
        Overrides less than operator.
    `__rlt__(other: Any) -> Feature`
        Overrides right less than operator.
    `__le__(other: Any) -> Feature`
        Overrides less than or equal to operator.
    `__rle__(other: Any) -> Feature`
        Overrides right less than or equal to operator.
    `__ge__(other: Any) -> Feature`
        Overrides greater than or equal to operator.
    `__rge__(other: Any) -> Feature`
        Overrides right greater than or equal to operator.
    `__xor__(other: Any) -> Feature`
        Overrides XOR operator.
    `__and__(other: Feature) -> Feature`
        Overrides AND operator.
    `__rand__(other: Feature) -> Feature`
        Overrides right AND operator.
    `__getitem__(key: Any) -> Feature`
        Allows direct slicing of the data.
    `_format_input(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        Formats the input data for the feature.
    `_process_and_get(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        Calls the `get` method according to the `__distributed__` attribute.
    `_process_output(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> None`
        Processes the output of the feature.
    `_image_wrapped_format_input(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        Ensures the input is a list of Image.
    `_no_wrap_format_input(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        Ensures the input is a list of Image.
    `_no_wrap_process_and_get(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        Calls the `get` method according to the `__distributed__` attribute.
    `_image_wrapped_process_and_get(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        Calls the `get` method according to the `__distributed__` attribute.
    `_image_wrapped_process_output(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> None`
        Processes the output of the feature.
    `_no_wrap_process_output(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> None`
        Processes the output of the feature.
    `_coerce_inputs(image_list: np.ndarray | list[np.ndarray] | Image | list[Image], **kwargs: Any) -> list[Image]`
        Coerces the input to a list of Image.

    """

    properties: PropertyDict
    _input: DeepTrackNode
    _random_seed: DeepTrackNode
    arguments: Feature | None

    __list_merge_strategy__ = MERGE_STRATEGY_OVERRIDE
    __distributed__ = True
    __property_memorability__ = 1
    __conversion_table__ = ConversionTable()
    __gpu_compatible__ = False

    _wrap_array_with_image: bool = False

    def __init__(
        self: Feature,
        _input: Any = [],
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize a new Feature instance.

        Parameters
        ----------
        _input: np.ndarray or list[np.ndarray] or Image or list of Images, optional
            The initial input(s) for the feature, often images or other data. 
            If not provided, defaults to an empty list.
        **kwargs: dict of str to Any
            Keyword arguments that are wrapped into `Property` instances and 
            stored in `self.properties`, allowing for dynamic or parameterized
            behavior.
            If not provided, defaults to an empty list.
        
        """

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
        self._random_seed = DeepTrackNode(lambda: random.randint(
            0, 2147483648)
            )
        self._random_seed = DeepTrackNode(lambda: random.randint(
            0, 2147483648)
            )
        self._random_seed.add_child(self)
        # self.add_dependency(self._random_seed)  # Executed by add_child.

        # Initialize arguments to None.
        self.arguments = None

    def get(
        self: Feature,
        image: np.ndarray | list[np.ndarray] | Image | list[Image],
        **kwargs: dict[str, Any],
    ) -> Image | list[Image]:
        """Transform an image [abstract method].

        Abstract method that defines how the feature transforms the input. The 
        current value of all properties will be passed as keyword arguments.

        Parameters
        ----------
        image: np.ndarray or list of np.ndarray or Image or list of Images
            The image or list of images to transform.
        **kwargs: dict of str to Any
            The current value of all properties in `properties`, as well as any 
            global arguments passed to the feature.

        Returns
        -------
        Image or list of Images
            The transformed image or list of images.

        Raises
        ------
        NotImplementedError
            Raised if this method is not overridden by subclasses.

        """

        raise NotImplementedError

    def __call__(
        self: Feature,
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image] = None,
        _ID: tuple[int, ...] = (),
        **kwargs: dict[str, Any],
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
        image_list: np.ndarrray or list[np.ndarrray] or Image or list of Images, optional
            The input to the feature or pipeline. If `None`, the feature uses 
            previously set input values or propagates properties.
        **kwargs: dict of str to Any
            Additional parameters passed to the pipeline. These override 
            properties with matching names. For example, calling 
            `feature(x, value=4)` executes `feature` on the input `x` while 
            setting the property `value` to `4`. All features in a pipeline are 
            affected by these overrides.

        Returns
        -------
        Any
            The output of the feature or pipeline after execution.
        
        """

        # If image_list is as Source, activate it.
        self._activate_sources(image_list)

        # Potentially fragile. Maybe a special variable dt._last_input instead?
        # If the input is not empty, set the value of the input.
        if (
            image_list is not None
            and not (isinstance(image_list, list) and len(image_list) == 0)
            and not (isinstance(image_list, tuple)
                     and any(isinstance(x, SourceItem) for x in image_list))
        ):
            self._input.set_value(image_list, _ID=_ID)

        # A dict to store the values of self.arguments before updating them.
        original_values = {}

        # If there are no self.arguments, instead propagate the values of the
        # kwargs to all properties in the computation graph.
        if kwargs and self.arguments is None:
            propagate_data_to_dependencies(self, **kwargs)

        # If there are self.arguments, update the values of self.arguments to 
        # match kwargs.
        if isinstance(self.arguments, Feature):
            for key, value in kwargs.items():
                if key in self.arguments.properties:
                    original_values[key] = \
                        self.arguments.properties[key](_ID=_ID)
                    self.arguments.properties[key].set_value(value, _ID=_ID)

        # This executes the feature. DeepTrackNode will determine if it needs
        # to be recalculated. If it does, it will call the `action` method.
        output = super().__call__(_ID=_ID)

        # If there are self.arguments, reset the values of self.arguments to
        # their original values.
        for key, value in original_values.items():
            self.arguments.properties[key].set_value(value, _ID=_ID)

        return output


    resolve = __call__


    def store_properties(
        self: Feature,
        toggle: bool = True,
        recursive: bool = True,
    ) -> None:
        """Control whether to return an Image object.
        
        If selected `True`, the output of the evaluation of the feature is an 
        Image object that also contains the properties.

        Parameters
        ----------
        toggle: bool
            If `True`, store properties. If `False`, do not store.
        recursive: bool
            If `True`, also set the same behavior for all dependent features.

        """

        self._wrap_array_with_image = toggle

        if recursive:
            for dependency in self.recurse_dependencies():
                if isinstance(dependency, Feature):
                    dependency.store_properties(toggle, recursive=False)

    def torch(
        self: Feature, 
        dtype: torch.dtype = None, 
        device: torch.device = None,
        permute_mode: str = "never",
    ) -> 'Feature':
        """Convert the feature to a PyTorch feature.

        Parameters
        ----------
        dtype: torch.dtype, optional
            The data type of the output.
        device: torch.device, optional
            The target device of the output (e.g., CPU or GPU).
        permute_mode: str
            Controls whether to permute image axes for PyTorch. 
            Defaults to "never".

        Returns
        -------
        Feature
            The transformed, PyTorch-compatible feature.

        """

        from deeptrack.pytorch.features import ToTensor

        tensor_feature = ToTensor(
            dtype=dtype, 
            device=device, 
            permute_mode=permute_mode,
        )
        
        tensor_feature.store_properties(False, recursive=False)
        
        return self >> tensor_feature

    def batch(
        self: Feature,
        batch_size: int = 32
    ) -> tuple | list[Image]:
        """Batch the feature.

        This method produces a batch of outputs by repeatedly calling 
        `update()` and `__call__()`.

        Parameters
        ----------
        batch_size: int
            The number of times to sample or generate data.

        Returns
        -------
        tuple or list of Images
            A tuple of stacked arrays (if the outputs are NumPy arrays or 
            torch tensors) or a list of images if the outputs are not 
            stackable.

        """

        results = [self.update()() for _ in range(batch_size)]
        results = list(zip(*results))

        for idx, r in enumerate(results):

            if isinstance(r[0], np.ndarray):
                results[idx] = np.stack(r)
            else:
                import torch

                if isinstance(r[0], torch.Tensor):
                    results[idx] = torch.stack(r)

        return tuple(results)

    def action(
        self: Feature,
        _ID: tuple[int, ...] = (),
    ) -> Image | list[Image]:
        """Core logic to create or transform the image.

        This method creates or transforms the input image by calling the 
        `get()` method with the correct inputs.

        Parameters
        ----------
        _ID: tuple of int
            The unique identifier for the current execution.

        Returns
        -------
        Image or list of Images
            The resolved image or list of resolved images.

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

    def __use_gpu__(
        self: Feature,
        inp: np.ndarray | Image,
        **_: Any,
    ) -> bool:
        """Determine if the feature should use the GPU.
        
        Parameters
        ----------
        inp: np.ndarray or Image
            The input image to check.
        **_: Any
            Additional arguments (unused).

        Returns
        -------
        bool
            True if GPU acceleration is enabled and beneficial, otherwise 
            False.

        """

        return self.__gpu_compatible__ and np.prod(np.shape(inp)) > (90000)

    def update(
        self: Feature,
        **global_arguments: Any,
    ) -> Feature:
        """Refreshes the feature to generate a new output.

        By default, when a feature is called multiple times, it returns the 
        same value. Calling `update()` forces the feature to recompute and 
        return a new value the next time it is evaluated.

        Parameters
        ----------
        **global_arguments: Any
            Optional global arguments that can be passed to modify the 
            feature update behavior.

        Returns
        -------
        Feature
            The updated feature instance, ensuring the next evaluation produces 
            a fresh result.
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
        """Adds a feature to the dependecy graph of this one.

        Parameters
        ----------
        feature: Feature
            The feature to add as a dependency.

        Returns
        -------
        Feature
            The newly added feature (for chaining).

        """

        feature.add_child(self)
        # self.add_dependency(feature)  # Already done by add_child().

        return feature

    def seed(
        self: Feature,
        _ID: tuple[int, ...] = (),
    ) -> None:
        """Seed the random number generator.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            Unique identifier for parallel evaluations.

        """

        np.random.seed(self._random_seed(_ID=_ID))

    def bind_arguments(
        self: Feature,
        arguments: Feature,
    ) -> Feature:
        """Binds another feature’s properties as arguments to this feature.

        This method allows properties of `arguments` to be dynamically linked 
        to this feature, enabling shared configurations across multiple features.
        It is commonly used in advanced feature pipelines.

        See Also
        --------
        features.Arguments
            A utility that helps manage and propagate feature arguments efficiently.

        Parameters
        ----------
        arguments: Feature
            The feature whose properties will be bound as arguments to this feature.

        Returns
        -------
        Feature
            The current feature instance with bound arguments.
        """

        self.arguments = arguments

        return self

    def _normalize(
        self: Feature,
        **properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalizes the properties.

        This method handles all unit normalizations and conversions. For each class in 
        the method resolution order (MRO), it checks if the class has a 
        `__conversion_table__` attribute. If found, it calls the `convert` method of 
        the conversion table using the properties as arguments.

        Parameters
        ----------
        **properties: dict of str to Any
            The properties to be normalized and converted.

        Returns
        -------
        dict of str to Any
            The normalized and converted properties.

        """

        for cl in type(self).mro():
            if hasattr(cl, "__conversion_table__"):
                properties = cl.__conversion_table__.convert(**properties)

        for key, val in properties.items():
            if isinstance(val, Quantity):
                properties[key] = val.magnitude
        return properties

    def plot(
        self: Feature,
        input_image: np.ndarray | list[np.ndarray] | Image | list[Image] = None,
        resolve_kwargs: dict = None,
        interval: float = None,
        **kwargs
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
        input_image: np.ndarray or list np.ndarray or Image or list of Image, optional
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

        else:
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
                    interval = (1 / 30 * 1000)

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

    def _process_properties(
        self: Feature,
        propertydict: dict[str, Any],
    ) -> dict[str, Any]:
        """Preprocesses the input properties before calling `.get()`.

        This method acts as a preprocessing hook for subclasses, allowing them 
        to modify or normalize input properties before the feature's main 
        computation.

        Parameters
        ----------
        propertydict: dict[str, Any]
            The dictionary of properties to be processed before being passed 
            to the `.get()` method.

        Returns
        -------
        dict[str, Any]
            The processed property dictionary after normalization.

        Notes
        -----
        - Calls `_normalize()` internally to standardize input properties.
        - Subclasses may override this method to implement additional 
          preprocessing steps.
        
        """

        propertydict = self._normalize(**propertydict)
        return propertydict

    def _activate_sources(
        self: Feature,
        x: Any,
    ) -> None:
        """Activates source items within the given input.

        This method checks if `x` or its elements (if `x` is a list) are 
        instances of `SourceItem`, and if so, calls them to trigger their 
        behavior.

        Parameters
        ----------
        x: Any
            The input to process. If `x` is a `SourceItem`, it is activated.
            If `x` is a list, each `SourceItem` within the list is activated.

        Notes
        -----
        - Non-`SourceItem` elements in `x` are ignored.
        - This method is used to ensure that all source-dependent computations
          are properly triggered when required.
        
        """
        
        if isinstance(x, SourceItem):
            x()
        else:
            if isinstance(x, list):
                for source in x:
                    if isinstance(source, SourceItem):
                        source()

    def __getattr__(
        self: Feature,
        key: str,
    ) -> Any:
        """Custom attribute access for the Feature class.

        This method allows the properties of the `Feature` instance to be 
        accessed as if they were attributes. For example, `feature.my_property`
        is equivalent to `feature.properties["my_property"]`.

        If the requested attribute (`key`) exists in the `properties` 
        dictionary, the corresponding value is returned. If the attribute does
        not exist, or if the `properties` attribute is not set, an 
        This method allows the properties of the `Feature` instance to be 
        accessed as if they were attributes. For example, `feature.my_property`
        is equivalent to `feature.properties["my_property"]`.

        If the requested attribute (`key`) exists in the `properties` 
        dictionary, the corresponding value is returned. If the attribute does
        not exist, or if the `properties` attribute is not set, an 
        `AttributeError` is raised.

        Parameters
        ----------
        key: str
            The name of the attribute being accessed.

        Returns
        -------
        Any
            The value of the property corresponding to the given `key` in the 
            `properties` dictionary.

        Raises
        ------
        AttributeError
            If the `properties` attribute is not defined for the instance or 
            if the `key` does not exist in `properties`.

        Examples
        --------
        >>> import deptrack as dt 

        Accessing an attribute as if it were a property:
        >>> feature = dt.DummyFeature(value=42)
        >>> feature.value()
        42

        If the `properties` attribute is not defined for the instance or if the
        `key` does not exist in `properties`, an `AttributeError` is raised:
        >>> feature.nonexistent_property
        AttributeError: 'MyFeature' object has no attribute 
        'nonexistent_property'
        
        """

        if "properties" in self.__dict__:
            properties = self.__dict__["properties"]
            if key in properties:
                return properties[key]

        raise AttributeError(f"'{self.__class__.__name__}' object has "
                             "no attribute '{key}'")

    def __iter__(
        self: Feature,
    ) -> Iterable:
        """ Returns an infinite iterator that continuously yields feature 
        values.

        """

        while True:
            yield from next(self)

    def __next__(
        self: Feature,
    ) -> Any:
        """Returns the next resolved feature in the sequence.
        
        """

        yield self.update().resolve()

    def __rshift__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Chains this feature with another feature or function using '>>'.
        
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
        """Chains another feature or function with this feature using '<<'.
        
        """

        if isinstance(other, Feature):
            return Chain(other, self)

        if isinstance(other, DeepTrackNode):
            return Chain(Value(other), self)

        return NotImplemented

    def __add__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Adds another value or feature using '+'.
        
        """
    
        return self >> Add(other)

    def __radd__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Adds this feature to another value using right '+'.
        
        """
    
        return Value(other) >> Add(self)

    def __sub__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Subtracts another value or feature using '-'.
        
        """
        
        return self >> Subtract(other)

    def __rsub__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Subtracts this feature from another value using right '-'.
        
    """
        return Value(other) >> Subtract(self)

    def __mul__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Multiplies this feature with another value using '*'.
        
        """
    
        return self >> Multiply(other)

    def __rmul__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Multiplies another value with this feature using right '*'.
        
        """

        return Value(other) >> Multiply(self)

    def __truediv__(
        self: Feature, 
        other: Any
        ) -> Feature:
        """Divides this feature by another value using '/'.
        
        """
        
        return self >> Divide(other)

    def __rtruediv__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Divides another value by this feature using right '/'.
        
        """
    
        return Value(other) >> Divide(self)

    def __floordiv__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Performs floor division using '//'.
        
        """
        
        return self >> FloorDivide(other)

    def __rfloordiv__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Performs right floor division using '//'.
        
        """
        
        return Value(other) >> FloorDivide(self)

    def __pow__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Raises this feature to a power using '**'.
        
        """
        
        return self >> Power(other)

    def __rpow__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Raises another value to this feature as a power using right '**'.
        
        """
        
        return Value(other) >> Power(self)

    def __gt__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if this feature is greater than another using '>'."""
        return self >> GreaterThan(other)

    def __rgt__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if another value is greater than this feature using 
        right '>'.
        
        """
        
        return Value(other) >> GreaterThan(self)

    def __lt__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if this feature is less than another using '<'.
        
        """
        
        return self >> LessThan(other)

    def __rlt__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if another value is less than this feature using 
        right '<'.
        
        """
        
        return Value(other) >> LessThan(self)

    def __le__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if this feature is less than or equal to another using '<='.
        
        """
        
        return self >> LessThanOrEquals(other)

    def __rle__(
        self: Feature,
        other: Any
    ) -> Feature:
        """Checks if another value is less than or equal to this feature using 
        right '<='.
        
        """
        
        return Value(other) >> LessThanOrEquals(self)

    def __ge__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if this feature is greater than or equal to another 
        using '>='.
        
        """
        
        return self >> GreaterThanOrEquals(other)

    def __rge__(
        self: Feature, 
        other: Any
    ) -> Feature:
        """Checks if another value is greater than or equal to this feature 
        using right '>='.
        
        """

        return Value(other) >> GreaterThanOrEquals(self)

    def __xor__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Repeats the feature a given number of times using '^'.
        
        """

        return Repeat(self, other)

    def __and__(
        self: Feature,
        other: Any,
    ) -> Feature:
        """Stacks this feature with another using '&'.
        
        """

        return self >> Stack(other)

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
    ) -> 'Feature':
        """Allows direct slicing of the feature's output.
        
        """
        
        if not isinstance(slices, tuple):
            slices = (slices,)

        # We make it a list to ensure that each element is sampled 
        # independently.
        slices = list(slices)

        return self >> Slice(slices)

    # private properties to dispatch based on config
    @property
    def _format_input(self):
        """Selects the appropriate input formatting function based on 
        configuration.
        
        """

        if self._wrap_array_with_image:
            return self._image_wrapped_format_input
        else:
            return self._no_wrap_format_input

    @property
    def _process_and_get(self):
        """Selects the appropriate processing function based on configuration.
        
        """

        if self._wrap_array_with_image:
            return self._image_wrapped_process_and_get
        else:
            return self._no_wrap_process_and_get

    @property
    def _process_output(self):
        """Selects the appropriate output processing function based on 
        configuration.
        
        """

        if self._wrap_array_with_image:
            return self._image_wrapped_process_output
        else:
            return self._no_wrap_process_output

    def _image_wrapped_format_input(
        self: Feature,
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image],
        **kwargs: dict[str, Any],
    ) -> list[Image]:
        """Wraps input data as Image instances before processing.
        
        """

        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]

        inputs = [(Image(image)) for image in image_list]
        return self._coerce_inputs(inputs, **kwargs)

    def _no_wrap_format_input(
        self: Feature, 
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image],
        **kwargs: dict[str, Any],
    ) -> list[Image]:
        """Processes input data without wrapping it as Image instances.
       
        """

        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]

        return image_list

    def _no_wrap_process_and_get(
        self: Feature,
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image],
        **feature_input: dict[str, Any],
    ) -> list[Image]:
        """Processes input data without additional wrapping and retrieves 
        results.
        
        """

        if self.__distributed__:
            # Call get on each image in list, and merge properties from 
            # corresponding image
            return [self.get(x, **feature_input) for x in image_list]

        else:
            # Call get on entire list.
            new_list = self.get(image_list, **feature_input)

            if not isinstance(new_list, list):
                new_list = [new_list]

            return new_list

    def _image_wrapped_process_and_get(
        self: Feature,
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image],
        **feature_input: dict[str, Any],
    ) -> list[Image]:
        """Processes input data while maintaining Image properties.
        
        """

        if self.__distributed__:
            # Call get on each image in list, and merge properties from 
            # corresponding image

            results = []

            for image in image_list:
                output = self.get(image, **feature_input)
                if not isinstance(output, Image):
                    output = Image(output)

                output.merge_properties_from(image)
                results.append(output)

            return results

        else:
            # Call get on entire list.
            new_list = self.get(image_list, **feature_input)

            if not isinstance(new_list, list):
                new_list = [new_list]

            for idx, image in enumerate(new_list):
                if not isinstance(image, Image):
                    new_list[idx] = Image(image)
            return new_list

    def _image_wrapped_process_output(
        self: Feature,
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image], 
        feature_input: dict[str, Any],
    ) -> None:
        """Appends feature properties and input data to each Image.
        
        """

        for index, image in enumerate(image_list):

            if self.arguments:
                image.append(self.arguments.properties())

            image.append(feature_input)

    def _no_wrap_process_output(
        self: Feature,
        image_list: np.ndarray | list[np.ndarray] | Image | list[Image],
        feature_input: dict[str, Any],
    ) -> None:
        """Extracts and updates raw values from Image instances.
        
        """

        for index, image in enumerate(image_list):

            if isinstance(image, Image):
                image_list[index] = image._value

    def _coerce_inputs(
        self: Feature,
        inputs: list[np.ndarray] | list[Image],
        **kwargs: dict[str, Any],
    ) -> list[Image]:
        """Converts inputs to the appropriate data type based on 
        GPU availability.
        
        """

        if config.gpu_enabled:

            return [
                i.to_cupy()
                if (not self.__distributed__) and self.__use_gpu__(i, **kwargs)
                else i.to_numpy()
                for i in inputs
            ]

        else:
            return [i.to_numpy() for i in inputs]

def propagate_data_to_dependencies(
    feature: Feature,
    **kwargs: dict[str, Any]
) -> None:
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
    """Provides the structure of a feature set without input transformations.

    A `StructuralFeature` does not directly transform the input data or add new 
    properties. Instead, it is commonly used as a logical or organizational 
    tool to structure and manage feature sets within a pipeline.

    Since `StructuralFeature` does not override the `__init__` or `get` 
    methods, it inherits the behavior of the base `Feature` class.

    Attributes
    ----------
    __property_verbosity__: int
        Controls whether this feature’s properties are included in the output 
        image’s property list. A value of `2` means that this feature’s 
        properties are not included.
    __distributed__: bool
        Determines whether the feature’s `get` method is applied to each 
        element in the input list (`__distributed__ = True`) or to the entire 
        list as a whole (`__distributed__ = False`).

    Notes
    -----
    Structural features are typically used for tasks like grouping or chaining 
    features, applying sequential or conditional logic, or structuring 
    pipelines without directly modifying the data.

    """

    __property_verbosity__: int = 2  # Hide properties from logs or output.
    __distributed__: bool = False  # Process the entire image list in one call.


class Chain(StructuralFeature):
    """Resolve two features sequentially.

    This feature applies two features sequentially, passing the output of the 
    first feature as the input to the second. It enables building feature 
    chains that execute complex transformations by combining simple operations.

    Parameters
    ----------
    feature_1: Feature
        The first feature in the chain. Its output is passed to `feature_2`.
    feature_2: Feature
        The second feature in the chain, which processes the output from 
        `feature_1`.
    **kwargs: dict of str to Any, optional
        Additional keyword arguments passed to the parent `StructuralFeature` 
        (and, therefore, `Feature`).

    Methods
    -------
    `get(image: np.ndarray | list[np.ndarray] | Image | list[Image], _ID: tuple[int, ...], **kwargs: dict[str, Any]) -> Image | list[Image]`
        Apply the two features in sequence on the given input image.

    Notes
    -----
    This feature is used to combine simple operations into a pipeline without the 
    need for explicit function chaining. It is syntactic sugar for creating 
    sequential feature pipelines.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create a feature chain where the first feature adds a constant offset, and 
    the second feature multiplies the result by a constant:
    
    >>> A = dt.Add(value=10)
    >>> M = dt.Multiply(value=0.5)

    Chain the features:
    >>> chain = A >> M  

    Equivalent to: 
    >>> chain = dt.Chain(A, M)

    Create a dummy image:
    >>> dummy_image = np.ones((2, 4))

    Apply the chained features:
    >>> transformed_image = chain(dummy_image)
    >>> print(transformed_image)
    [[5.5 5.5 5.5 5.5]
    [5.5 5.5 5.5 5.5]]

    """

    def __init__(
        self: Feature,
        feature_1: Feature,
        feature_2: Feature,
        **kwargs: dict[str, Any],
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
            The second feature, applied after `feature_1`.
        **kwargs: dict of str to Any, optional
            Additional keyword arguments passed to the parent constructor (e.g., 
            name, properties).

        """

        super().__init__(**kwargs)

        self.feature_1 = self.add_feature(feature_1)
        self.feature_2 = self.add_feature(feature_2)

    def get(
        self: Feature,
        image: np.ndarray | list[np.ndarray] | Image | list[Image],
        _ID: tuple[int, ...] = (),
        **kwargs: dict[str, Any],
    ) -> Image | list[Image]:
        """Apply the two features sequentially to the given input image(s).

        This method first applies `feature_1` to the input image(s) and then passes 
        the output through `feature_2`.

        Parameters
        ----------
        image: np.ndarray or list np.ndarray or Image or list of Image
            The input data, which can be an `Image` or a list of `Image` objects, 
            to transform sequentially.
        _ID: tuple of int, optional
            A unique identifier for caching or parallel execution. Defaults to an 
            empty tuple.
        **kwargs: dict of str to Any
            Additional parameters passed to or sampled by the features. These are 
            generally unused here, as each sub-feature fetches its required properties 
            internally.

        Returns
        -------
        Image or list of Images
            The final output after `feature_1` and then `feature_2` have processed 
            the input.

        """

        image = self.feature_1(image, _ID=_ID)
        image = self.feature_2(image, _ID=_ID)
        return image


Branch = Chain  # Alias for backwards compatibility.


class DummyFeature(Feature):
    """A no-op feature that simply returns the input unchanged.

    This class can serve as a container for properties that don't directly 
    transform the data but need to be logically grouped. Since it inherits 
    transform the data but need to be logically grouped. Since it inherits 
    from `Feature`, any keyword arguments passed to the constructor are 
    stored as `Property` instances in `self.properties`, enabling dynamic 
    behavior or parameterization without performing any transformations 
    on the input data.

    Parameters
    ----------
    _input: np.ndarray or list np.ndarray or Image or list of Images, optional
        An optional input (image or list of images) that can be set for 
        the feature. By default, an empty list.
    **kwargs: dict of str to Any
        Additional keyword arguments are wrapped as `Property` instances and 
        stored in `self.properties`.

    Methods
    -------
    `get(image: np.ndarray | list np.ndarray | Image | list[Image], **kwargs: dict[str, Any]) -> Image | list[Image]`
        Simply returns the input image(s) unchanged.


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
    >>> print(np.array_equal(dummy_image, output_image))
    True

    Access the properties stored in DummyFeature:
    >>> print(dummy_feature.properties["value"]())
    42

    """

    def get(
        self: Feature,
        image: np.ndarray | list[np.ndarray] | Image | list[Image], 
        **kwargs: Any,
    )-> Image | list[Image]:
        """Return the input image or list of images unchanged.

        This method simply returns the input without applying any transformation. 
        It adheres to the `Feature` interface by accepting additional keyword 
        arguments for consistency, although they are not used in this method.

        Parameters
        ----------
        image: np.ndarray or list np.ndarray or Image or list of Image
            The image or list of images to pass through without modification.
        **kwargs: Any
            Additional properties sampled from `self.properties` or passed 
            externally. These are unused here but provided for consistency 
            with the `Feature` interface.

        Returns
        -------
        Image or list of Images
            The same `image` object that was passed in.

        """

        return image


class Value(Feature):
    """Represents a constant (per evaluation) value in a DeepTrack pipeline.

    This feature holds a constant value (e.g., a scalar or array) and supplies 
    it on demand to other parts of the pipeline. It does not transform the 
    input image but instead returns the stored value.

    Parameters
    ----------
    value: PropertyLike[float], optional
        The numerical value to store. Defaults to 0. If an `Image` is provided,
        a warning is issued recommending conversion to a NumPy array for 
        The numerical value to store. Defaults to 0. If an `Image` is provided,
        a warning is issued recommending conversion to a NumPy array for 
        performance reasons.
    **kwargs: dict of str to Any
        Additional named properties passed to the `Feature` constructor.

    Attributes
    ----------
    __distributed__: bool
        Set to `False`, indicating that this feature’s `get(...)` method 
        processes the entire list of images (or data) at once, rather than 
        distributing calls for each item.

    Methods
    -------
    `get(image: Any, value: float, **kwargs: dict[str, Any]) -> float`
        Returns the stored value, ignoring the input image.


    Examples
    --------
    >>> import deeptrack as dt

    Initialize a constant value and retrieve it:
    >>> value = dt.Value(42)
    >>> print(value())
    42

    Override the value at call time:
    >>> print(value(value=100))
    100

    """

    __distributed__: bool = False  # Process as a single batch.

    def __init__(
        self: Feature, 
        value: PropertyLike[float] = 0, 
        **kwargs: dict[str, Any]
    ):
        """Initialize the `Value` feature to store a constant value.

        This feature holds a constant numerical value and provides it to the 
        pipeline as needed. If an `Image` object is supplied, a warning is 
        issued to encourage converting it to a NumPy array for performance 
        optimization.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The initial value to store. If an `Image` is provided, a warning is
            raised. Defaults to 0.
        **kwargs: dict of str to Any
            Additional keyword arguments passed to the `Feature` constructor, 
            such as custom properties or the feature name.

        """

        if isinstance(value, Image):
            import warnings
            warnings.warn(
                "Setting dt.Value value as an Image object is likely to lead "
                "to performance deterioration. Consider converting it to a "
                "numpy array using np.array."
            )

        super().__init__(value=value, **kwargs)

    def get(
        self: Feature,
        image: Any, 
        value: float, 
        **kwargs: dict[str, Any]
    ) -> float:
        """Return the stored value, ignoring the input image.

        The `get` method simply returns the stored numerical value, allowing 
        for dynamic overrides when the feature is called.

        Parameters
        ----------
        image: Any
            Input data typically processed by features. For `Value`, this is 
            ignored and does not affect the output.
        value: float
            The current value to return. This may be the initial value or an 
            overridden value supplied during the method call.
        **kwargs: dict of str to Any
            Additional keyword arguments, which are ignored but included for 
            consistency with the feature interface.

        Returns
        -------
        float
            The stored or overridden `value`, returned unchanged.

        """

        return value


class ArithmeticOperationFeature(Feature):
    """Applies an arithmetic operation element-wise to inputs.

    This feature performs an arithmetic operation (e.g., addition, subtraction,
    multiplication) on the input data. The inputs can be single values or lists
    of values. If a list is passed, the operation is applied to each element. 
    If both inputs are lists of different lengths, the shorter list is cycled.

    Parameters
    ----------
    op: Callable[[Any, Any], Any]
        The arithmetic operation to apply, such as a built-in operator 
        (`operator.add`, `operator.mul`) or a custom callable.
    value: float or int or list of float or int, optional
        The second operand for the operation. Defaults to 0. If a list is 
        provided, the operation will apply element-wise.
    **kwargs: dict of str to Any
        Additional keyword arguments passed to the parent `Feature`.

    Attributes
    ----------
    __distributed__: bool
        Indicates that this feature’s `get(...)` method processes the input as 
        a whole (`False`) rather than distributing calls for individual items.
    __gpu_compatible__: bool
        Specifies that the feature is compatible with GPU processing (`True`).

    Methods
    -------
    `get(image: Any | list of Any, value: float | int | list[float] | int, **kwargs: dict[str, Any]) -> list[Any]`
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
    __gpu_compatible__: bool = True


    def __init__(
        self: Feature,
        op: Callable[[Any, Any], Any],
        value: float | int | list[float | int] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the ArithmeticOperationFeature.

        Parameters
        ----------
        op: Callable[[Any, Any], Any]
            The arithmetic operation to apply, such as `operator.add`, `operator.mul`, 
            or any custom callable that takes two arguments.
        value: float or int or list of float or int, optional
            The second operand(s) for the operation. If a list is provided, the 
            operation is applied element-wise. Defaults to 0.
        **kwargs: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` constructor.

        """

        super().__init__(value=value, **kwargs)
        self.op = op

    def get(
        self: Feature,
        image: Any | list[Any],
        value: float | int | list[float | int],
        **kwargs: Any,
    ) -> list[Any]:
        """Apply the operation element-wise to the input data.

        Parameters
        ----------
        image: Any or list of Any
            The input data, either a single value or a list of values, to be 
            transformed by the arithmetic operation.
        value: float, int, or list of float or int
            The second operand(s) for the operation. If a single value is 
            provided, it is broadcast to match the input size. If a list is 
            provided, it will be cycled to match the length of the input list.
        **kwargs: dict of str to Any
            Additional parameters or property overrides. These are generally 
            unused in this context but provided for compatibility with the 
            `Feature` interface.

        Returns
        -------
        list of Any
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
    value: PropertyLike[int or float], optional
        The value to add to the input. Defaults to 0.
    **kwargs: dict of str to Any
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
    
    Or:
    >>> pipeline = 5 + dt.Value([1, 2, 3])
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> sum_feature = dt.Add(value=5)
    >>> pipeline = sum_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Add feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to add to the input. Defaults to 0.
        **kwargs: dict of str to Any
            Additional keyword arguments passed to the parent `Feature`.

        """

        super().__init__(operator.add, value=value, **kwargs)


class Subtract(ArithmeticOperationFeature):
    """Subtract a value from the input.

    This feature performs element-wise subtraction (-) from the input.
    
    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to subtract from the input. Defaults to 0.
    **kwargs: dict of str to Any
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
    
    Or:
    >>> pipeline = -2 + dt.Value([1, 2, 3])
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> sub_feature = dt.Subtract(value=2)
    >>> pipeline = sub_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Subtract feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to subtract from the input. Defaults to 0.
        **kwargs: dict of str to Any
            Additional keyword arguments passed to the parent `Feature`.
       
        """

        super().__init__(operator.sub, value=value, **kwargs)


class Multiply(ArithmeticOperationFeature):
    """Multiply the input by a value.

    This feature performs element-wise multiplication (*) of the input.
    
    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to multiply the input. Defaults to 0.
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

    Or:
    >>> pipeline = 5 * dt.Value([1, 2, 3])
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> mul_feature = dt.Multiply(value=5)
    >>> pipeline = mul_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Multiply feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to multiply the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.mul, value=value, **kwargs)


class Divide(ArithmeticOperationFeature):
    """Divide the input with a value.

    This feature performs element-wise division (/) of the input.
    
    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to divide the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Divide`:
    >>> pipeline = Value([1, 2, 3]) >> Divide(value=5)
    >>> pipeline.resolve()
    [0.2 0.4 0.6]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = Value([1, 2, 3]) / 5
    
    Which is not equivalent to:
    >>> pipeline = 5 / Value([1, 2, 3])  # Different result.
    
    Or, more explicitly:
    >>> input_value = Value([1, 2, 3])
    >>> truediv_feature = Divide(value=5)
    >>> pipeline = truediv_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Divide feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to divide the input. Defaults to 0.
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
    value: PropertyLike[int or float], optional
        The value to floor-divide the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `FloorDivide`:
    >>> pipeline = dt.Value([-3, 3, 6]) >> dt.FloorDivide(value=5)
    >>> pipeline.resolve()
    [0.2 0.4 0.6]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([-3, 3, 6]) // 5
    
    Which is not equivalent to:
    >>> pipeline = 5 // dt.Value([-3, 3, 6])  # Different result.
    
    Or, more explicitly:
    >>> input_value = dt.Value([-3, 3, 6])
    >>> floordiv_feature = dt.FloorDivide(value=5)
    >>> pipeline = feature(floordiv_input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the FloorDivide feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to fllor-divide the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.floordiv, value=value, **kwargs)


class Power(ArithmeticOperationFeature):
    """Raise the input to a power.

    This feature performs element-wise power (**) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to take the power of the input. Defaults to 0.
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
    
    Which is not equivalent to:
    >>> pipeline = 3 ** dt.Value([1, 2, 3])  # Different result.
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> pow_feature = Power(value=3)
    >>> pipeline = pow_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Power feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to take the power of the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.pow, value=value, **kwargs)


class LessThan(ArithmeticOperationFeature):
    """Determine whether input is less than value.

    This feature performs element-wise comparison (<) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to compare (<) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `LessThan`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.LessThan(value=2)
    >>> pipeline.resolve()
    [True False False]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) < 2
    
    Which is not equivalent to:
    >>> pipeline = 2 < dt.Value([1, 2, 3])  # Different result.
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> lt_feature = dt.LessThan(value=2)
    >>> pipeline = lt_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the LessThan feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to compare (<) with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.lt, value=value, **kwargs)


class LessThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is less than or equal to value.

    This feature performs element-wise comparison (<=) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to compare (<=) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `LessThanOrEquals`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.LessThanOrEquals(value=2)
    >>> pipeline.resolve()
    [True  True False]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) <= 2
    
    Which is not equivalent to:
    >>> pipeline = 2 <= dt.Value([1, 2, 3])  # Different result.
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> le_feature = dt.LessThanOrEquals(value=2)
    >>> pipeline = le_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the LessThanOrEquals feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to compare (<=) with the input. Defaults to 0.
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
    value: PropertyLike[int or float], optional
        The value to compare (>) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `GreaterThan`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.GreaterThan(value=2)
    >>> pipeline.resolve()
    [False False  True]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) > 2

    Which is not equivalent to:
    >>> pipeline = 2 > dt.Value([1, 2, 3])  # Different result.
    
    Or, most explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> gt_feature = dt.GreaterThan(value=2)
    >>> pipeline = gt_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the GreaterThan feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to compare (>) with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.gt, value=value, **kwargs)


class GreaterThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is greater than or equal to value.

    This feature performs element-wise comparison (>=) of the input.

    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to compare (<=) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `GreaterThanOrEquals`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.GreaterThanOrEquals(value=2)
    >>> pipeline.resolve()
    [False  True  True]
    
    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) >= 2

    Which is not equivalent to:
    >>> pipeline = 2 >= dt.Value([1, 2, 3])  # Different result.
    
    Or, more explicitly:
    >>> input_value = dt.Value([1, 2, 3])
    >>> ge_feature = dt.GreaterThanOrEquals(value=2)
    >>> pipeline = ge_feature(input_value)

    """

    def __init__(
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the GreaterThanOrEquals feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to compare (>=) with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.ge, value=value, **kwargs)


GreaterThanOrEqual = GreaterThanOrEquals


class Equals(ArithmeticOperationFeature):
    """Determine whether input is equal to a given value.

    This feature performs element-wise comparison (==) between the input and a
    specified value.

    Parameters
    ----------
    value: PropertyLike[int or float], optional
        The value to compare (==) with the input. Defaults to 0.
    **kwargs: Any
        Additional keyword arguments passed to the parent constructor.

    Notes
    -----
    - Unlike other arithmetic operators, `Equals` does not define `__eq__` 
      (`==`) and `__req__` (`==`) in `DeepTrackNode` and `Feature`, as this 
      would affect Python’s built-in identity comparison.
    - This means that the standard `==` operator is overloaded only for 
      expressions involving `Feature` instances but not for comparisons 
      involving regular Python objects.
    - Always use `>>` to apply `Equals` correctly in a feature chain.
    
    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Equals`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Equals(value=2)
    >>> pipeline.resolve()
    [False  True  False]
    
    This is the **only correct way** to apply `Equals` in a feature pipeline.
    
    ### Incorrect Approaches
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
        self: Feature,
        value: PropertyLike[float] = 0,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Equals feature.

        Parameters
        ----------
        value: PropertyLike[float], optional
            The value to compare (==) with the input. Defaults to 0.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(operator.eq, value=value, **kwargs)


Equal = Equals


class Stack(Feature):
    """Stacks the input and the value.
    
    This feature combines the output of the input data (`image`) and the 
    value produced by the specified feature (`value`). The resulting output 
    is a list where the elements of the `image` and `value` are concatenated.

    If either the input (`image`) or the `value` is a single `Image` object, 
    it is automatically converted into a list to maintain consistency in the 
    output format.

    If B is a feature, `Stack` can be visualized as::

    >>>   A >> Stack(B) = [*A(), *B()]

    Parameters
    ----------
    value: PropertyLike[Any]
        The feature or data to stack with the input.
    **kwargs: dict of str to Any
        Additional arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs. 
        Always `False` for `Stack`, as it processes all inputs at once.

    Methods
    -------
    `get(image: Any, value: Any, **kwargs: dict[str, Any]) -> list[Any]`
        Concatenate the input with the value.

    Examples
    --------
    >>> import deeptrack as dt

    Start by creating a pipeline using `Stack`:
    >>> pipeline = dt.Value([1, 2, 3]) >> dt.Stack(value=[4, 5])
    >>> print(pipeline.resolve())
    [1, 2, 3, 4, 5]

    Equivalently, this pipeline can be created using:
    >>> pipeline = dt.Value([1, 2, 3]) & [4, 5]

    Or:
    >>> pipeline = [4, 5] & dt.Value([1, 2, 3])  # Different result.

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        value: PropertyLike[Any],
        **kwargs: dict[str, Any],
    ):
        """Initialize the Stack feature.

        Parameters
        ----------
        value: PropertyLike[Any]
            The feature or data to stack with the input.
        **kwargs: dict of str to Any
            Additional arguments passed to the parent `Feature` class.
        
        """

        super().__init__(value=value, **kwargs)

    def get(
        self: Feature,
        image: Any | list[Any],
        value: Any | list[Any],
        **kwargs: dict[str, Any],
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
        **kwargs: dict of str to Any
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
    runtime. This is particularly useful when working with parameterized 
    pipelines, such as toggling behaviors based on whether an image is a label 
    or a raw input.

    Methods
    -------
    `get(image: Any, **kwargs: dict[str, Any]) -> Any`
        Passes the input image through unchanged, while allowing for property 
        overrides.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from tempfile import NamedTemporaryFile
    >>> from PIL import Image as PIL_Image
    >>> import os

    Create a temporary image:
    >>> test_image_array = (np.ones((50, 50)) * 128).astype(np.uint8)
    >>> temp_png = NamedTemporaryFile(suffix=".png", delete=False)
    >>> PIL_Image.fromarray(test_image_array).save(temp_png.name)

    A typical use-case is:
    >>> arguments = dt.Arguments(is_label=False)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name) >>
    ...     dt.Gaussian(sigma = (1 - arguments.is_label) * 5)
    ... )
    >>> image_pipeline.bind_arguments(arguments)

    >>> image = image_pipeline()  # Image with added noise.
    >>> print(image.std())
    5.041072178933536

    Change the argument:
    >>> image = image_pipeline(is_label=True) # Image with no noise.
    >>> print(image.std())
    0.0

    Remove the temporary image:
    >>> os.remove(temp_png.name)

    For a non-mathematical dependence, create a local link to the property as 
    follows:
    >>> arguments = dt.Arguments(is_label=False)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name) >>
    ...     dt.Gaussian(
    ...         is_label=arguments.is_label,
    ...         sigma=lambda is_label: 0 if is_label else 5
    ...     )
    ... )
    >>> image_pipeline.bind_arguments(arguments)

    Keep in mind that, if any dependent property is non-deterministic, they may 
    permanently change:
    >>> arguments = dt.Arguments(noise_max_sigma=5)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name) >>
    ...     dt.Gaussian(
    ...         noise_max_sigma=arguments.noise_max_sigma,
    ...         sigma=lambda noise_max_sigma: np.random.rand()*noise_max_sigma
    ...     )
    ... )
    >>> image_pipeline.bind_arguments(arguments)
    >>> image_pipeline.store_properties()

    >>> image = image_pipeline()
    >>> print(image.get_property("sigma"))
    1.1838819055669947

    >>> image = image_pipeline(noise_max_sigma=0)
    >>> print(image.get_property("sigma"))
    0.0

    As with any feature, all arguments can be passed by deconstructing the 
    properties dict:
    >>> arguments = dt.Arguments(is_label=False, noise_sigma=5)
    >>> image_pipeline = (
    ...     dt.LoadImage(path=temp_png.name) >>
    ...     dt.Gaussian(
    ...         sigma=lambda is_label, noise_sigma: (
    ...             0 if is_label else noise_sigma
    ...         )
    ...         **arguments.properties
    ...     )
    ... )
    >>> image_pipeline.bind_arguments(arguments)

    >>> image = image_pipeline()  # Image with added noise.
    >>> print(image.std())
    5.002151761964336

    >>> image = image_pipeline(is_label=True)  # Raw image with no noise.
    >>> print(image.std())
    0.0

    """

    def get(
        self: Feature,
        image: Any,
        **kwargs: dict[str, Any]
    ) -> Any:

        """Process the input image and allow property overrides.

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

    This feature conditionally applies a given feature to an input image based 
    on a specified probability. A random number is sampled, and if it is less 
    than `probability`, the feature is resolved; otherwise, the input image 
    remains unchanged.

    Parameters
    ----------
    feature: Feature
        The feature to resolve conditionally.
    probability: PropertyLike[float]
        The probability (between 0 and 1) of resolving the feature.
    *args: list[Any], optional
        Positional arguments passed to the parent `StructuralFeature` class.
    **kwargs: dict of str to Any, optional
        Additional keyword arguments passed to the parent `StructuralFeature` 
        class.

    Methods
    -------
    `get(image: np.ndarray, probability: float, random_number: float, **kwargs: dict[str, Any]) -> np.ndarray`
        Resolves the feature if the sampled random number is less than the 
        specified probability.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    
    In this example, the `Add` feature is applied to the input image with 
    a 70% chance. Define a feature and wrap it with `Probability`:
    >>> add_feature = dt.Add(value=2)
    >>> probabilistic_feature = dt.Probability(add_feature, probability=0.7)

    Define an input image:
    >>> input_image = np.ones((5, 5))

    Apply the feature:
    >>> output_image = probabilistic_feature(input_image)

    """

    def __init__(
        self: Feature,
        feature: Feature,
        probability: PropertyLike[float],
        *args: list[Any],
        **kwargs: dict[str, Any],
    ):
        """Initialize the Probability feature.

        Parameters
        ----------
        feature: Feature
            The feature to resolve conditionally.
        probability: PropertyLike[float]
            The probability (between 0 and 1) of resolving the feature.
        *args: list[Any], optional
            Positional arguments passed to the parent `StructuralFeature` class.
        **kwargs: dict of str to Any, optional
            Additional keyword arguments passed to the parent `StructuralFeature` class.

        """
        
        super().__init__(
            *args, 
            probability=probability, 
            random_number=np.random.rand, 
            **kwargs,
        )
        self.feature = self.add_feature(feature) 

    def get(
        self: Feature,
        image: np.ndarray,
        probability: float,
        random_number: float,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Resolve the feature if a random number is less than the probability.

        Parameters
        ----------
        image: np.ndarray
            The input image to process.
        probability: float
            The probability (between 0 and 1) of resolving the feature.
        random_number: float
            A random number sampled to determine whether to resolve the 
            feature.
        **kwargs: dict of str to Any
            Additional arguments passed to the feature's `resolve` method.

        Returns
        -------
        np.ndarray
            The processed image. If the feature is resolved, this is the output of the feature; 
            otherwise, it is the unchanged input image.

        """
                
        if random_number < probability:
            image = self.feature.resolve(image, **kwargs)

        return image


class Repeat(Feature):
    """Applies a feature multiple times in sequence.

    The `Repeat` feature iteratively applies another feature, passing the 
    output of each iteration as the input to the next. This enables chained 
    transformations, where each iteration builds upon the previous one. The 
    number of repetitions is defined by `N`.

    Each iteration operates with its own set of properties, and the index of 
    the current iteration is accessible via `_ID` or `replicate_index`. 
    `_ID` is extended to include the current iteration index, ensuring 
    deterministic behavior when needed.

    Parameters
    ----------
    feature: Feature
        The feature to be repeated.
    N: int
        The number of times to apply the feature in sequence.
    **kwargs: dict of str to Any

    Attributes
    ----------
    __distributed__: bool
        Always `False` for `Repeat`, since it processes sequentially rather 
        than distributing computation across inputs.

    Methods
    -------
    `get(image: Any, N: int, _ID: tuple[int, ...], **kwargs: dict[str, Any]) -> Any`
        Applies the feature `N` times in sequence, passing the output of each 
        iteration as the input to the next.

    Examples
    --------
    >>> import deeptrack as dt
    
    Define an `Add` feature that adds `10` to its input:
    >>> add_ten = dt.Add(value=10)

    Apply this feature **3 times** using `Repeat`:
    >>> pipeline = dt.Repeat(add_ten, N=3)

    Process an input list:
    >>> print(pipeline.resolve([1, 2, 3]))
    [31, 32, 33]

    Step-by-step breakdown:
    - Iteration 1: `[1, 2, 3] + 10 → [11, 12, 13]`
    - Iteration 2: `[11, 12, 13] + 10 → [21, 22, 23]`
    - Iteration 3: `[21, 22, 23] + 10 → [31, 32, 33]`

    Alternative shorthand using `^` operator:
    >>> pipeline = dt.Add(value=10) ^ 3
    >>> print(pipeline.resolve([1, 2, 3]))
    [31, 32, 33]
    
    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        feature: Feature, 
        N: int, 
        **kwargs: dict[str, Any],
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
        **kwargs: dict of str to Any
            Keyword arguments that override properties dynamically at each 
            iteration and are also passed to the parent `Feature` class.

        """

        super().__init__(N = N, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Feature,
        image: Any,
        N: int,
        _ID: tuple[int, ...] = (),
        **kwargs: dict[str, Any],
    ) -> Any:
        """Sequentially apply the feature `N` times.

        This method applies the feature `N` times, passing the output of each 
        iteration as the input to the next. The `_ID` tuple is updated at 
        each iteration, ensuring dynamic property updates and reproducibility.
  
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
        **kwargs: dict of str to Any
            Additional keyword arguments passed to the feature.

        Returns
        -------
        Any
            The output of the final iteration after `N` sequential applications 
            of the feature.

        """
        
        for n in range(N):

            index = _ID + (n,)  # Track iteration index.

            image = self.feature(
                image,
                _ID=index,
                replicate_index=index,  # Pass replicate_index for legacy.
            )

        return image


class Combine(StructuralFeature):
    """Combine multiple features into a single feature.

    This feature sequentially resolves a list of features and returns their 
    results as a list. Each feature in the `features` parameter operates on 
    the same input, and their outputs are aggregated into a single list.

    Parameters
    ----------
    features: list of Features
        A list of features to combine. Each feature will be resolved in the 
        order they appear in the list.
    **kwargs: dict of str to Any, optional
        Additional keyword arguments passed to the parent `StructuralFeature` 
        class.

    Methods
    -------
    `get(image_list: Any, **kwargs: dict[str, Any]) -> list[Any]`
        Resolves each feature in the `features` list on the input image and 
        returns their results as a list.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Define a list of features to combine `GaussianBlur` and `Add`:
    >>> blur_feature = dt.GaussianBlur(sigma=2)
    >>> add_feature = dt.Add(value=10)

    Combine the features:
    >>> combined_feature = dt.Combine([blur_feature, add_feature])

    Define an input image:
    >>> input_image = np.ones((10, 10))

    Apply the combined feature:
    >>> output_list = combined_feature(input_image)

    """

    __distributed__: bool = False

    def __init__(
        self: Feature, 
        features: list[Feature], 
        **kwargs: dict[str, Any]
    ):
        """Initialize the Combine feature.

        Parameters
        ----------
        features: list of Features
            A list of features to combine. Each feature is added as a 
            dependency to ensure proper execution in the computation graph.
        **kwargs: dict of str to Any, optional
            Additional keyword arguments passed to the parent 
            `StructuralFeature` class.

        """

        super().__init__(**kwargs)
        self.features = [self.add_feature(f) for f in features]

    def get(
        self: Feature, 
        image_list: Any,
        **kwargs: dict[str, Any]
    ) -> list[Any]:
        """Resolve each feature in the `features` list on the input image.

        Parameters
        ----------
        image_list: Any
            The input image or list of images to process.
        **kwargs: dict of str to Any
            Additional arguments passed to each feature's `resolve` method.

        Returns
        -------
        list[Any]
            A list containing the outputs of each feature applied to the input.

        """

        return [f(image_list, **kwargs) for f in self.features]


class Slice(Feature):
    """Dynamically applies array indexing to input Image(s).
    
    This feature allows **dynamic slicing** of an image using integer indices, 
    slice objects, or ellipses (`...`). While normal array indexing is preferred 
    for static cases, `Slice` is useful when the slicing parameters **must be 
    computed dynamically** based on other properties.

    Parameters
    ----------
    slices: Iterable[int | slice | ...]
        The slicing instructions for each dimension. Each element corresponds 
        to a dimension in the input image.
    **kwargs: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, slices: tuple[int | slice | ...], **kwargs: dict[str, Any]) -> np.ndarray`
        Applies the specified slices to the input image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    **Recommended Approach: Use Normal Indexing for Static Slicing**
    >>> feature = dt.DummyFeature()
    >>> static_slicing = feature[:, 1:2, ::-2]
    >>> result = static_slicing.resolve(np.arange(27).reshape((3, 3, 3)))
    >>> print(result)

    **Using `Slice` for Dynamic Slicing (when necessary)**
    If slices depend on computed properties, use `Slice`:
    >>> feature = dt.DummyFeature()
    >>> dynamic_slicing = feature >> dt.Slice(
    ...     slices=(slice(None), slice(1, 2), slice(None, None, -2))
    ... )
    >>> result = dynamic_slicing.resolve(np.arange(27).reshape((3, 3, 3)))
    >>> print(result)

    In both cases, slices can be defined dynamically based on feature 
    properties.

    """

    def __init__(
        self: Feature,
        slices: PropertyLike[
            Iterable[
                PropertyLike[int] | PropertyLike[slice] | PropertyLike[...]
            ]
        ],
        **kwargs: dict[str, Any],
    ):
        """Initialize the Slice feature.

        Parameters
        ----------
        slices: list[int | slice | ...] or tuple[int | slice | ...]
            The slicing instructions for each dimension, specified as a 
            list or tuple of integers, slice objects, or ellipses (`...`).
        **kwargs: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(slices=slices, **kwargs)

    def get(
        self: Feature,
        image: np.ndarray,
        slices: tuple[Any, ...] | Any,
        **kwargs: dict[str, Any],
    ):
        """Apply the specified slices to the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to be sliced.
        slices: tuple[int | slice | ellipsis, ...] | int | slice | ellipsis
            The slicing instructions for the input image. Each element in the
            tuple corresponds to a dimension in the input image. If a single
            element is provided, it is converted to a tuple.
        **kwargs: dict of str to Any
            Additional keyword arguments (unused in this implementation).

        Returns
        -------
        np.ndarray
            The sliced image.

        """

        try:
            # Convert slices to a tuple if possible.
            slices = tuple(slices)
        except ValueError:
            # Leave slices as is if conversion fails.
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
    **kwargs: dict of str to Any
        Properties to send to child

    Methods
    -------
    `get(image: Any, **kwargs: dict[str, Any]) -> Any`
        Resolves the child feature with the provided arguments.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Start by creating a `Gaussian` feature: 
    >>> gaussian_noise = dt.Gaussian()

    Dynamically modify the behavior of the feature using `Bind`:
    >>> bound_feature = dt.Bind(gaussian_noise, mu = -5, sigma=2)
    
    >>> input_image = np.zeros((512, 512))
    >>> output_image = bound_feature.resolve(input_image)
    >>> print(np.mean(output_image), np.std(output_image))
    -4.9954959040123152 1.9975296489398942

    """

    __distributed__: bool = False

    def __init__(
        self: Feature, 
        feature: Feature, 
        **kwargs: dict[str, Any]
    ):
        """Initialize the Bind feature.

        Parameters
        ----------
        feature: Feature
            The child feature to bind.
        **kwargs: dict of str to Any
            Properties or arguments to pass to the child feature.

        """

        super().__init__(**kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Feature, 
        image: Any, 
        **kwargs: dict[str, Any]
    ) -> Any:
        """Resolve the child feature with the dynamically provided arguments.

        Parameters
        ----------
        image: Any
            The input data or image to process.
        **kwargs: dict of str to Any
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


class BindUpdate(StructuralFeature):
    """Bind a feature with certain arguments.

    This feature binds a child feature with specific properties (`kwargs`) that 
    are passed to it when it is updated. It is similar to the `Bind` feature 
    but is marked as deprecated in favor of `Bind`.

    Parameters
    ----------
    feature: Feature
        The child feature to bind with specific arguments.
    **kwargs: dict of str to Any
        Properties to send to the child feature during updates.

    Methods
    -------
    `get(image: Any, **kwargs: dict[str, Any]) -> Any`
        Resolves the child feature with the provided arguments.

    Warnings
    --------
    This feature is deprecated and may be removed in a future release. 
    It is recommended to use `Bind` instead for equivalent functionality.

    Notes
    -----
    The current implementation is not guaranteed to be exactly equivalent to 
    prior implementations.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Start by creating a `Gaussian` feature:
    >>> gaussian_noise = dt.Gaussian()

    Dynamically modify the behavior of the feature using `BindUpdate`:
    >>> bound_feature = dt.BindUpdate(gaussian_noise, mu = 5, sigma=3)
    
    >>> input_image = np.zeros((512, 512))
    >>> output_image = bound_feature.resolve(input_image)
    >>> print(np.mean(output_image), np.std(output_image))
    4.998501486851294 3.0020269383538176

    """

    __distributed__: bool = False

    def __init__(
        self: Feature, 
        feature: Feature, 
        **kwargs: dict[str, Any]
    ):
        """Initialize the BindUpdate feature.

        Parameters
        ----------
        feature: Feature
            The child feature to bind with specific arguments.
        **kwargs: dict of str to Any
            Properties to send to the child feature during updates.

        Warnings
        --------
        Emits a deprecation warning, encouraging the use of `Bind` instead.

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
        **kwargs: dict[str, Any]
    ) -> Any:
        """Resolve the child feature with the provided arguments.

        Parameters
        ----------
        image: Any
            The input data or image to process.
        **kwargs: dict of str to Any
            Properties or arguments to pass to the child feature during 
            resolution.

        Returns
        -------
        Any
            The result of resolving the child feature with the provided 
            arguments.

        """

        return self.feature.resolve(image, **kwargs)


class ConditionalSetProperty(StructuralFeature):
    """Conditionally override the properties of a child feature.

    This feature modifies the properties of a child feature only when a 
    specified condition is met. If the condition evaluates to `True`, 
    the given properties are applied; otherwise, the child feature remains 
    unchanged.

    **Note**: It is advisable to use `dt.Arguments` instead when possible, 
    since this feature **overwrites** properties, which may affect future 
    calls to the feature.

    Parameters
    ----------
    feature: Feature
        The child feature whose properties will be modified conditionally.
    condition: PropertyLike[str] or PropertyLike[bool]
        Either a boolean value (`True`/`False`) or the name of a boolean 
        property in the feature’s property dictionary. If the condition 
        evaluates to `True`, the specified properties are applied.
    **kwargs: dict[str, Any]
        The properties to be applied to the child feature if `condition` is 
        `True`.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.

    Methods
    -------
    `get(image: Any, condition: str | bool, **kwargs: dict[str, Any]) -> Any`
        Resolves the child feature, conditionally applying the specified 
        properties.

    Notes
    -----
    - If `condition` is a string, the condition must be explicitly passed when
      resolving.
    - The properties applied **do not persist** unless explicitly stored.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Define a `Gaussian` noise feature:
    >>> gaussian_noise = dt.Gaussian(sigma=0)

    --- Using a boolean condition ---
    Apply `sigma=5` **only if** `condition=True`:
    >>> conditional_feature = dt.ConditionalSetProperty(
    ...     gaussian_noise, sigma=5
    ... )

    Define an image:
    >>> image = np.ones((512, 512))

    Resolve with condition met:
    >>> noisy_image = conditional_feature.update(image, condition=True)
    >>> print(noisy_image.std())  # Should be ~5
    4.987707046984823

    Resolve without condition:
    >>> clean_image = conditional_feature.update(image, condition=False)
    >>> print(clean_image.std())  # Should be 0
    0.0

    --- Using a string-based condition ---
    Define condition as a string:
    >>> conditional_feature = dt.ConditionalSetProperty(
    ...     gaussian_noise, sigma=5, condition="is_noisy"
    ... )

    Resolve with condition met:
    >>> noisy_image = conditional_feature.update(image, is_noisy=True)
    >>> print(noisy_image.std())  # Should be ~5
    5.006310381139811

    Resolve without condition:
    >>> clean_image = conditional_feature.update(image, is_noisy=False)
    >>> print(clean_image.std())  # Should be 0
    0.0
    
    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        feature: Feature,
        condition: PropertyLike[str | bool] | None = None,
        **kwargs: dict[str, Any],
    ):
        """Initialize the ConditionalSetProperty feature.

        Parameters
        ----------
        feature: Feature
            The child feature to conditionally modify.
        condition: PropertyLike[str or bool]
            A boolean value or the name of a boolean property in the feature's 
            property dictionary. If the condition evaluates to `True`, the 
            specified properties are applied.
        **kwargs: dict of str to Any
            Properties to apply to the child feature if the condition is 
            `True`.

        """

        if isinstance(condition, str):
            kwargs.setdefault(condition, True)

        super().__init__(condition=condition, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Feature,
        image: Any,
        condition: str | bool,
        **kwargs: dict[str, Any],
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
        **kwargs:: dict of str to Any
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


class ConditionalSetFeature(StructuralFeature):
    """Conditionally resolves one of two features based on a condition.

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
    **kwargs: dict of str to Any
        Additional keyword arguments passed to the parent `StructuralFeature`.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.

    Methods
    -------
    `get(image: Any, condition: str | bool, **kwargs: dict[str, Any]) -> Any`
        Resolves the appropriate feature based on the condition.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Define two `Gaussian` noise features:
    >>> true_feature = dt.Gaussian(sigma=0)
    >>> false_feature = dt.Gaussian(sigma=5)
    
    --- Using a boolean condition ---
    Combine the features into a conditional set feature. 
    If not provided explicitely, condition is assumed to be True:
    >>> conditional_feature = dt.ConditionalSetFeature(
    ...     on_true=true_feature, 
    ...     on_false=false_feature, 
    ... )

    Define an image:
    >>> image = np.ones((512, 512))

    Resolve based on the condition:
    >>> clean_image = conditional_feature(image) # If not specified, default is True
    >>> print(clean_image.std())  # Should be 0
    0.0
    
    >>> noisy_image = conditional_feature(image, condition=False)
    >>> print(noisy_image.std())  # Should be ~5
    4.987707046984823

    >>> clean_image = conditional_feature(image, condition=True)
    >>> print(clean_image.std())  # Should be 0
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
    >>> print(noisy_image.std())  # Should be ~5
    5.006310381139811

    >>> clean_image = conditional_feature(image, is_noisy=True)
    >>> print(clean_image.std())  # Should be 0
    0.0

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        on_false: Feature | None = None,
        on_true: Feature | None = None,
        condition: PropertyLike[str | bool] = True,
        **kwargs: dict[str, Any],
    ):
        """Initialize the ConditionalSetFeature.

        Parameters
        ----------
        on_false: Feature, optional
            The feature to resolve if the condition evaluates to `False`.
        on_true: Feature, optional
            The feature to resolve if the condition evaluates to `True`.
        condition: str or bool, optional
            The name of the property to listen to, or a boolean value. Defaults 
            to `"is_label"`.
        **kwargs:: dict of str to Any
            Additional keyword arguments for the parent `StructuralFeature`.

        """

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
        self: Feature,
        image: Any,
        *,
        condition: str | bool,
        **kwargs: dict[str, Any],
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
        **kwargs:: dict of str to Any
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
        if _condition:
            if self.on_true:
                return self.on_true(image)
            else:
                return image
        else:
            if self.on_false:
                return self.on_false(image)
            else:
                return image


class Lambda(Feature):
    """Apply a user-defined function to each image in the input.

    This feature allows applying a custom function to individual images in the
    input pipeline. The `function` parameter must be wrapped in an 
    **outer function** that can depend on other properties of the pipeline. 
    The **inner function** processes a single image.

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
    `get(image: np.ndarray | Image, function: Callable[[Image], Image], **kwargs: dict[str, Any]) -> Image`
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

    Apply the feature to an image:
    >>> input_image = np.ones((5, 5))
    >>> output_image = lambda_feature(input_image)
    >>> print(output_image)
    [[5. 5. 5. 5. 5.]
     [5. 5. 5. 5. 5.]
     [5. 5. 5. 5. 5.]
     [5. 5. 5. 5. 5.]
     [5. 5. 5. 5. 5.]]
    
    """

    def __init__(
        self: Feature,
        function: Callable[..., Callable[[Image], Image]],
        **kwargs: dict[str, Any],
    ):
        """Initialize the Lambda feature.

        This feature applies a user-defined function to process an image. The 
        `function` parameter must be a callable that returns another function, 
        where the inner function operates on the image.

        Parameters
        ----------
        function: Callable[..., Callable[[Image], Image]]
            A callable that produces a function. The outer function can accept 
            additional arguments from the pipeline, while the inner function 
            processes a single image.
        **kwargs: dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(function=function, **kwargs)

    def get(
        self: Feature,
        image: np.ndarray | Image,
        function: Callable[[Image], Image],
        **kwargs: dict[str, Any],
    ) -> Image:
        """Apply the custom function to the input image.

        This method applies a user-defined function to transform the input 
        image. The function should be a callable that takes an image as input 
        and returns a modified version of it.

        Parameters
        ----------
        image: np.ndarray or Image
            The input image to be processed.
        function: Callable[[Image], Image]
            A callable function that takes an image and returns a transformed 
            image.
        **kwargs: dict of str to Any
            Additional keyword arguments (unused in this implementation).

        Returns
        -------
        Image
            The transformed image after applying the function.

        """

        return function(image)


class Merge(Feature):
    """Apply a custom function to a list of images.

    This feature allows applying a user-defined function to a list of images. 
    The `function` parameter must be a callable that returns another function, 
    where:
      - The **outer function** can depend on other properties in the pipeline.
      - The **inner function** takes a list of images and returns a single 
      image or a list of images.
    
    **Note:** The function must be wrapped in an **outer layer** to enable 
    dependencies on other properties while ensuring correct execution.

    Parameters
    ----------
    function: Callable[..., Callable[[list[np.ndarray] | list[Image]], np.ndarray | list[np.ndarray] | Image | list[Image]]]
        A callable that produces a function. The **outer function** can depend 
        on other properties of the pipeline, while the **inner function** 
        processes a list of images and returns either a single image or a list 
        of images.
    **kwargs: dict[str, Any]
        Additional parameters passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.

    Methods
    -------
    `get(list_of_images: list[np.ndarray] | list[Image], function: Callable[[list[np.ndarray] | list[Image]], np.ndarray | list[np.ndarray] | Image | list[Image]], **kwargs: dict[str, Any]) -> Image | list[Image]`
        Applies the custom function to the list of images.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Define a merge function that averages multiple images:
    >>> def merge_function_factory():
    ...     def merge_function(images):
    ...         return np.mean(np.stack(images), axis=0)
    ...     return merge_function

    Create a Merge feature:
    >>> merge_feature = dt.Merge(function=merge_function_factory)

    Apply the feature to a list of images:
    >>> image_1 = np.ones((5, 5)) * 2
    >>> image_2 = np.ones((5, 5)) * 4
    >>> output_image = merge_feature([image_1, image_2])
    >>> print(output_image)
    [[3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]]

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
        function: Callable[..., Callable[list[np.ndarray] | [list[Image]], np.ndarray | list[np.ndarray] | Image | list[Image]]]
            A callable that returns a function for processing a list of images.
            - The **outer function** can depend on other properties in the pipeline.
            - The **inner function** takes a list of images as input and 
              returns either a single image or a list of images.
        **kwargs: dict[str, Any]
            Additional parameters passed to the parent `Feature` class.

        """

        super().__init__(function=function, **kwargs)

    def get(
        self: Feature,
        list_of_images: list[np.ndarray] | list[Image],
        function: Callable[[list[np.ndarray] | list[Image]], np.ndarray | list[np.ndarray] | Image | list[Image]],
        **kwargs: dict[str, Any],
    ) -> Image | list[Image]:
        """Apply the custom function to a list of images.

        Parameters
        ----------
        list_of_images: list[np.ndarray] or list[Image]
            A list of images to be processed by the function.
        function: Callable[[list[np.ndarray] | list[Image]], np.ndarray | list[np.ndarray] | Image | list[Image]]
            The function that processes the list of images and returns either:
              - A single transformed image (`Image`)
              - A list of transformed images (`list[Image]`)
        **kwargs: dict[str, Any]
            Additional arguments (unused in this implementation).

        Returns
        -------
        Image | list[Image]
            The processed image(s) after applying the function.

        """

        return function(list_of_images)


class OneOf(Feature):
    """Resolves one feature from a given collection.

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
    **kwargs: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.

    Methods
    -------
    `_process_properties(propertydict: dict) -> dict`
        Processes the properties to determine the selected feature index.
    `get(image: Any, key: int, _ID: tuple[int, ...], **kwargs: dict[str, Any]) -> Any`
        Applies the selected feature to the input image.
  
    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Define multiple features:
    >>> feature_1 = dt.Add(value=10)
    >>> feature_2 = dt.Multiply(value=2)
    
    Create a `OneOf` feature that randomly selects a transformation:
    >>> one_of_feature = dt.OneOf([feature_1, feature_2])

    Apply it to an input image:
    >>> input_image = np.array([1, 2, 3])
    >>> output_image = one_of_feature(input_image)
    >>> print(output_image)  # The output depends on the randomly selected feature.

    Use a `key` to apply a specific feature:
    >>> controlled_feature = dt.OneOf([feature_1, feature_2], key=0)
    >>> output_image = controlled_feature(input_image)
    >>> print(output_image)  # Adds 10 to each element.

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        collection: Iterable[Feature],
        key: int | None = None,
        **kwargs: dict[str, Any],
    ):
        """Initialize the OneOf feature.

        Parameters
        ----------
        collection: Iterable[Feature]
            A collection of features to choose from. It will be stored as a tuple.
        key: int | None, optional
            The index of the feature to resolve from the collection. If not 
            provided, a feature is selected randomly at execution.
        **kwargs: dict of str to Any
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
        **kwargs: dict[str, Any],
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
        **kwargs: dict of str to Any
            Additional parameters passed to the selected feature.

        Returns
        -------
        Any
            The output of the selected feature applied to the input image.

        """

        return self.collection[key](image, _ID=_ID)


class OneOfDict(Feature):
    """Resolve one feature from a dictionary and apply it to an input.

    This feature selects a feature from a dictionary and applies it to an input. 
    The selection is made randomly by default, but it can be controlled using 
    the `key` argument.

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
    **kwargs: dict of str to Any
        Additional parameters passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__: bool
        Indicates whether this feature distributes computation across inputs.

    Methods
    -------
    `_process_properties(propertydict: dict) -> dict`
        Determines which feature to use based on `key`.
    `get(image: Any, key: Any, _ID: tuple[int, ...], **kwargs: dict[str, Any]) -> Any`
        Resolves the selected feature and applies it to the input image.
   
    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Define a dictionary of features:
    >>> features_dict = {
    ...     "add": dt.Add(value=10),
    ...     "multiply": dt.Multiply(value=2),
    ... }
    >>> one_of_dict_feature = dt.OneOfDict(features_dict)

    Apply a randomly selected feature:
    >>> input_image = np.array([1, 2, 3])
    >>> output_image = one_of_dict_feature(input_image)
    >>> print(output_image)

    Use a specific key to apply a predefined feature:
    >>> controlled_feature = dt.OneOfDict(features_dict, key="add")
    >>> output_image = controlled_feature(input_image)
    >>> print(output_image)  # Adds 10 to each element.

    """

    __distributed__: bool = False

    def __init__(
        self: Feature,
        collection: dict[Any, Feature],
        key: Any | None = None,
        **kwargs: dict[str, Any],
    ):
        """Initialize the OneOfDict feature.

        Parameters
        ----------
        collection: dict[Any, Feature]
            A dictionary where keys are identifiers and values are features.
        key: Any | None, optional
            The key of the feature to resolve from the dictionary. If `None`, 
            a random key is selected.
        **kwargs: dict of str to Any
            Additional parameters passed to the parent `Feature` class.

        """

        super().__init__(key=key, **kwargs)
        self.collection = collection

        # Add all features in the dictionary as dependencies.
        for feature in self.collection.values():
            self.add_feature(feature)

    def _process_properties(
        self: Feature, 
        propertydict: dict
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
        **kwargs: dict[str, Any],
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
        **kwargs: dict of str to Any
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
        Indicates whether this feature distributes computation across inputs.

    Methods
    -------
    `get(image: Any, path: str | list[str], load_options: dict[str, Any] | None, ndim: int, to_grayscale: bool, as_list: bool, get_one_random: bool, **kwargs: dict[str, Any]) -> np.ndarray`
        Load the image(s) from disk and process them.

    Raises
    ------
    IOError
        If no file reader could parse the file or the file does not exist.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> from tempfile import NamedTemporaryFile

    Create a temporary image file:
    >>> temp_file = NamedTemporaryFile(suffix=".npy", delete=False)
    >>> np.save(temp_file.name, np.random.rand(100, 100))

    Load the image using `LoadImage`:
    >>> load_image_feature = dt.LoadImage(path=temp_file.name, to_grayscale=True)
    >>> loaded_image = load_image_feature.resolve()

    Print image shape:
    >>> print(loaded_image.shape)

    If `to_grayscale=True`, the image is converted to grayscale (single channel).
    If `ndim=4`, additional dimensions are added if necessary.

    Cleanup the temporary file:
    >>> import os
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
        **kwargs: dict[str, Any],
    ):
        """Initialize the LoadImage feature.

        Parameters
        ----------
        path: PropertyLike[str or list[str]]
            The path(s) to the image(s) to load. Can be a single string or a list 
            of strings.
        load_options: PropertyLike[dict[str, Any]], optional
            Additional options passed to the file reader (e.g., `mode` for OpenCV, 
            `allow_pickle` for NumPy). Defaults to `None`.
        as_list: PropertyLike[bool], optional
            If `True`, treats the first dimension of the image as a list of images. 
            Defaults to `False`.
        ndim: PropertyLike[int], optional
            Ensures the image has at least this many dimensions. If the loaded image 
            has fewer dimensions, extra dimensions are added. Defaults to `3`.
        to_grayscale: PropertyLike[bool], optional
            If `True`, converts the image to grayscale. Defaults to `False`.
        get_one_random: PropertyLike[bool], optional
            If `True`, selects a single random image from a stack when `as_list=True`. 
            Defaults to `False`.
        **kwargs: dict of str to Any
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
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Load and process an image or a list of images from disk.

        This method attempts to load an image using multiple file readers 
        (`imageio`, `numpy`, `Pillow`, and `OpenCV`) until a valid format is 
        found. It supports optional processing steps such as ensuring a minimum
        number of dimensions, grayscale conversion, and treating multi-frame 
        images as lists.

        Parameters
        ----------
        path: str or list of str
            The file path(s) to the image(s) to be loaded. A single string 
            loads one image, while a list of paths loads multiple images.
        load_options: dict of str to Any, optional
            Additional options passed to the file reader (e.g., `allow_pickle` 
            for NumPy, `mode` for OpenCV). Defaults to `None`.
        ndim: int
            Ensures the image has at least this many dimensions. If the loaded 
            image has fewer dimensions, extra dimensions are added.
        to_grayscale: bool
            If `True`, converts the image to grayscale. Defaults to `False`.
        as_list: bool
            If `True`, treats the first dimension as a list of images instead 
            of stacking them into a NumPy array.
        get_one_random: bool
            If `True`, selects a single random image from a multi-frame stack
            when `as_list=True`. Defaults to `False`.
        **kwargs: dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The loaded and processed image(s). If `as_list=True`, returns a 
            list of images; otherwise, returns a single NumPy array.

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

                skimage.color.rgb2gray(image)
            except ValueError:
                import warnings

                warnings.warn("Non-rgb image, ignoring to_grayscale")

        # Ensure the image has at least `ndim` dimensions.
        while ndim and image.ndim < ndim:
            image = np.expand_dims(image, axis=-1)

        return image


class SampleToMasks(Feature):
    """Creates a mask from a list of images.

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
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    Define number of particles:
    >>> n_particles = 12

    Define optics and particles:
    >>> optics = dt.Fluorescence(output_region=(0, 0, 64, 64))
    >>> particle = dt.PointParticle(
    >>>     position=lambda: np.random.uniform(5, 55, size=2)
    >>> )
    >>> particles = particle ^ n_particles

    Define pipelines:
    >>> sim_im_pip = optics(particles)
    >>> sim_mask_pip = particles >> dt.SampleToMasks(
    ...     lambda: lambda particles: particles > 0,
    ...     output_region=optics.output_region,
    ...     merge_method="or"
    ... )
    >>> pipeline = sim_im_pip & sim_mask_pip
    >>> pipeline.store_properties()

    Generate image and mask:
    >>> image, mask = pipeline.update()()

    Get particle positions:
    >>> positions = np.array(image.get_property("position", get_one=False))

    Visualize results:
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(image, cmap="gray")
    >>> plt.title("Original Image")
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(mask, cmap="gray")
    >>> plt.scatter(positions[:,1], positions[:,0], c="r", marker="x", s = 10)
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
        **kwargs: dict[str, Any],
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
        **kwargs: dict[str, Any],
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
    dtype: PropertyLike[Any], optional
        The desired data type for the image. Defaults to `"float64"`.
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, dtype: str, **kwargs: dict[str, Any]) -> np.ndarray`
        Convert the data type of the input image.

    Examples
    --------
    >>> import numpy as np
    >>> from deeptrack.features import AsType

    Create an input array:
    >>> input_image = np.array([1.5, 2.5, 3.5])

    Apply an AsType feature to convert to `int32`:
    >>> astype_feature = AsType(dtype="int32")
    >>> output_image = astype_feature.get(input_image, dtype="int32")
    >>> print(output_image)
    [1 2 3]

    Verify the data type:
    >>> print(output_image.dtype)
    int32

    """

    def __init__(
        self: Feature,
        dtype: PropertyLike[Any] = "float64",
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the AsType feature.

        Parameters
        ----------
        dtype: PropertyLike[Any], optional
            The desired data type for the image. Defaults to `"float64"`.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(dtype=dtype, **kwargs)

    def get(
        self: Feature,
        image: np.ndarray,
        dtype: str,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Convert the data type of the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to process.
        dtype: str
            The desired data type for the image.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray
            The input image converted to the specified data type.

        """

        return image.astype(dtype)


class ChannelFirst2d(Feature):
    """Convert an image to a channel-first format.

    This feature rearranges the axes of a 3D image so that the specified axis 
    (e.g., channel axis) is moved to the first position. If the input image is 
    2D, it adds a new dimension at the front, effectively treating the 2D 
    image as a single-channel image.

    Parameters
    ----------
    axis: int, optional
        The axis to move to the first position. Defaults to `-1` (last axis).
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
        **kwargs: dict[str, Any],
    ):
        """Initialize the ChannelFirst2d feature.

        Parameters
        ----------
        axis: int, optional
            The axis to move to the first position. 
            Defaults to `-1` (last axis).
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Feature,
        image: np.ndarray,
        axis: int,
        **kwargs: dict[str, Any],
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
        integers is provided, each axis is scaled individually. Defaults to 1.
    **kwargs: dict of str to Any
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
        **kwargs: dict[str, Any],
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
            Defaults to `1`.
        **kwargs: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(factor=factor, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Feature,
        image: np.ndarray,
        factor: int | tuple[int, int, int],
        **kwargs: dict[str, Any],
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
        **kwargs: dict of str to Any
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
        The minimum distance between volumes in pixels. Defaults to `1`. 
        It can be negative to allow for partial overlap.
    max_attempts: int, optional
        The maximum number of attempts to place volumes without overlap.
        Defaults to `5`. 
    max_iters: int, optional
        The maximum number of resamplings. If this number is exceeded, a 
            new list of volumes is generated. Defaults to `100`.

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
        **kwargs: dict[str, Any],
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
            pixels. Defaults to `1`. Negative values allow for partial overlap.
        max_attempts: int, optional
            The maximum number of attempts to place the volumes without 
            overlap. Defaults to `5`.
        max_iters: int, optional
            The maximum number of resampling iterations per attempt. If 
            exceeded, a new list of volumes is generated. Defaults to `100`.
        
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
        **kwargs: dict[str, Any],
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
            UserWarning
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
    """Stores the output of a feature for reuse.

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
        If `True`, replaces the stored value with a new computation. Defaults 
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
        **kwargs: dict[str, Any],
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
            Defaults to `False`.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(feature=feature, key=key, replace=replace, **kwargs)
        self.feature = self.add_feature(feature, **kwargs)
        self._store: dict[Any, Image] = {}

    def get(
        self: Store,
        _: Any,
        key: Any,
        replace: bool,
        **kwargs: dict[str, Any],
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
        The axis or axes to squeeze. Defaults to `None`, squeezing all axes.
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, axis: int | tuple[int, ...], **kwargs: dict[str, Any]) -> np.ndarray`
        Squeeze the input image by removing singleton dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from deeptrack.features import Squeeze

    Create an input array with extra dimensions:
    >>> input_image = np.array([[[[1], [2], [3]]]])
    >>> print(input_image.shape)
    (1, 1, 3, 1)

    Create a Squeeze feature:
    >>> squeeze_feature = Squeeze(axis=0)
    >>> output_image = squeeze_feature(input_image)
    >>> print(output_image.shape)
    (1, 3, 1)

    Without specifying an axis:
    >>> squeeze_feature = Squeeze()
    >>> output_image = squeeze_feature(input_image)
    >>> print(output_image.shape)
    (3,)

    """

    def __init__(
        self: Squeeze,
        axis: int | tuple[int, ...] | None = None,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Squeeze feature.

        Parameters
        ----------
        axis: int or tuple[int, ...], optional
            The axis or axes to squeeze. Defaults to `None`, which squeezes 
            all axes.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Squeeze,
        image: np.ndarray,
        axis: int | tuple[int, ...] | None = None,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Squeeze the input image by removing singleton dimensions.

        Parameters
        ----------
        image: np.ndarray
            The input image to process.
        axis: int or tuple[int, ...], optional
            The axis or axes to squeeze. Defaults to `None`, which squeezes 
            all axes.
        **kwargs:: dict of str to Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray
            The squeezed image with reduced dimensions.

        """

        return np.squeeze(image, axis=axis)


class Unsqueeze(Feature):
    """Unsqueezes the input image to the smallest possible dimension.

    This feature adds new singleton dimensions to the input image at the 
    specified axis or axes. If no axis is specified, it defaults to adding 
    a singleton dimension at the last axis.

    Parameters
    ----------
    axis: int or tuple[int, ...], optional
        The axis or axes where new singleton dimensions should be added. 
        Defaults to `None`, which adds a singleton dimension at the last axis.
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, axis: int | tuple[int, ...] | None, **kwargs: dict[str, Any]) -> np.ndarray`
        Add singleton dimensions to the input image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input array:
    >>> input_image = np.array([1, 2, 3])
    >>> print(input_image.shape)
    (3,)

    Apply an Unsqueeze feature:
    >>> unsqueeze_feature = dt.Unsqueeze(axis=0)
    >>> output_image = unsqueeze_feature(input_image)
    >>> print(output_image.shape)
    (1, 3)

    Without specifying an axis:
    >>> unsqueeze_feature = dt.Unsqueeze()
    >>> output_image = unsqueeze_feature(input_image)
    >>> print(output_image.shape)
    (3, 1)

    """

    def __init__(
        self: Unsqueeze,
        axis: int | tuple[int, ...] | None = -1,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Unsqueeze feature.

        Parameters
        ----------
        axis: int or tuple[int, ...], optional
            The axis or axes where new singleton dimensions should be added. 
            Defaults to -1, which adds a singleton dimension at the last axis.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self: Unsqueeze,
        image: np.ndarray,
        axis: int | tuple[int, ...] | None = -1,
        **kwargs: dict[str, Any],

    ) -> np.ndarray:
        """Add singleton dimensions to the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to process.
        axis: int or tuple[int, ...], optional
            The axis or axes where new singleton dimensions should be added. 
            Defaults to -1, which adds a singleton dimension at the last axis.
        **kwargs:: dict of str to Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray
            The input image with the specified singleton dimensions added.

        """

        return np.expand_dims(image, axis=axis)


ExpandDims = Unsqueeze


class MoveAxis(Feature):
    """Moves the axis of the input image.

    This feature rearranges the axes of an input image, moving a specified 
    source axis to a new destination position. All other axes remain in their 
    original order.

    Parameters
    ----------
    source: int
        The axis to move.
    destination: int
        The destination position of the axis.
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, source: int, destination: int, **kwargs: dict[str, Any]) -> np.ndarray`
        Move the specified axis of the input image to a new position.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input array:
    >>> input_image = np.random.rand(2, 3, 4)
    >>> print(input_image.shape)
    (2, 3, 4)

    Apply a MoveAxis feature:
    >>> move_axis_feature = dt.MoveAxis(source=0, destination=2)
    >>> output_image = move_axis_feature(input_image)
    >>> print(output_image.shape)
    (3, 4, 2)

    """

    def __init__(
        self: MoveAxis,
        source: int,
        destination: int,
        **kwargs: dict[str, Any],
    ):
        """Initialize the MoveAxis feature.

        Parameters
        ----------
        source: int
            The axis to move.
        destination: int
            The destination position of the axis.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(source=source, destination=destination, **kwargs)

    def get(
        self: MoveAxis,
        image: np.ndarray,
        source: int,
        destination: int, 
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Move the specified axis of the input image to a new position.

        Parameters
        ----------
        image: np.ndarray
            The input image to process.
        source: int
            The axis to move.
        destination: int
            The destination position of the axis.
        **kwargs:: dict of str to Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray
            The input image with the specified axis moved to the destination.
        """

        return np.moveaxis(image, source, destination)


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
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, axes: tuple[int, ...] | None, **kwargs: dict[str, Any]) -> np.ndarray`
        Transpose the axes of the input image

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input array:
    >>> input_image = np.random.rand(2, 3, 4)
    >>> print(input_image.shape)
    (2, 3, 4)

    Apply a Transpose feature:
    >>> transpose_feature = dt.Transpose(axes=(1, 2, 0))
    >>> output_image = transpose_feature(input_image)
    >>> print(output_image.shape)
    (3, 4, 2)

    Without specifying axes:
    >>> transpose_feature = dt.Transpose()
    >>> output_image = transpose_feature(input_image)
    >>> print(output_image.shape)
    (4, 3, 2)

    """

    def __init__(
        self: Transpose,
        axes: tuple[int, ...] | None = None,
        **kwargs: dict[str, Any],
    ):
        """Initialize the Transpose feature.

        Parameters
        ----------
        axes: tuple[int, ...], optional
            A tuple specifying the permutation of the axes. If `None`, the 
            axes are reversed by default.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.
        
        """

        super().__init__(axes=axes, **kwargs)

    def get(
        self: Transpose,
        image: np.ndarray,
        axes: tuple[int, ...] | None = None,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Transpose the axes of the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to process.
        axes: tuple[int, ...], optional
            A tuple specifying the permutation of the axes. If `None`, the 
            axes are reversed by default.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray
            The transposed image with rearranged axes.

        """

        return np.transpose(image, axes)


Permute = Transpose


class OneHot(Feature):
    """Converts the input to a one-hot encoded array.

    This feature takes an input array of integer class labels and converts it 
    into a one-hot encoded array. The last dimension of the input is replaced 
    by the one-hot encoding.

    Parameters
    ----------
    num_classes: int
        The total number of classes for the one-hot encoding.
    **kwargs:: dict of str to Any
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image: np.ndarray, num_classes: int, **kwargs: dict[str, Any]) -> np.ndarray`
        Convert the input array of class labels into a one-hot encoded array.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    
    Create an input array of class labels:
    >>> input_data = np.array([0, 1, 2])

    Apply a OneHot feature:
    >>> one_hot_feature = dt.OneHot(num_classes=3)
    >>> one_hot_feature = dt.OneHot(num_classes=3)
    >>> one_hot_encoded = one_hot_feature.get(input_data, num_classes=3)
    >>> print(one_hot_encoded)
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    """

    def __init__(
        self: OneHot,
        num_classes: int,
        **kwargs: dict[str, Any],
    ):
        """Initialize the OneHot feature.

        Parameters
        ----------
        num_classes: int
            The total number of classes for the one-hot encoding.
        **kwargs:: dict of str to Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(num_classes=num_classes, **kwargs)

    def get(
        self: OneHot,
        image: np.ndarray,
        num_classes: int,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Convert the input array of labels into a one-hot encoded array.

        Parameters
        ----------
        image: np.ndarray
            The input array of class labels. The last dimension should contain 
            integers representing class indices.
        num_classes: int
            The total number of classes for the one-hot encoding.
        **kwargs: Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray
            The one-hot encoded array. The last dimension is replaced with 
            one-hot vectors of length `num_classes`.

        """

        # Flatten the last dimension if it's singleton.
        if image.shape[-1] == 1:
            image = image[..., 0]

        # Create the one-hot encoded array.
        return np.eye(num_classes)[image]


class TakeProperties(Feature):
    """Extracts all instances of a set of properties from a pipeline.

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
        **kwargs: dict[str, Any],
    ):
        """Initialize the TakeProperties feature.

        Parameters
        ----------
        feature: Feature
            The feature from which to extract properties.
        *names: str
            One or more names of the properties to extract.
=        **kwargs: dict[str, Any], optional
            Additional keyword arguments passed to the parent `Feature` class.
        
        """

        super().__init__(names=names, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: TakeProperties,
        image: Any,
        names: tuple[str, ...],
        _ID: tuple[int, ...] = (),
        **kwargs: dict[str, Any],
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
            dependencies are correctly matched. Defaults to an empty tuple.
        **kwargs: dict[str, Any], optional
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
