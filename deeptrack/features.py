"""Core features for building and processing pipelines in DeepTrack2.

This module defines the core classes and utilities used to create and 
manipulate features in DeepTrack2, enabling users to build sophisticated data 
processing pipelines with modular, reusable, and composable components.

Main Concepts
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

Key Classes
-----------
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

Module Highlights
-----------------
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

Example
-------
Define a simple pipeline with features:

>>> import numpy as np
>>> from deeptrack.features import Feature, Chain, Value

Create a basic addition feature:

>>> class Add(Feature):
...     def get(self, image, value, **kwargs):
...         return image + value

Create two features:

>>> add_five = Add(value=5)
>>> add_ten = Add(value=10)

Chain features together:

>>> pipeline = Chain(add_five, add_ten)

or equivalently:

>>> pipeline = add_five >> add_ten

Process an input image:

>>> input_image = np.array([1, 2, 3])
>>> output_image = pipeline(input_image)
>>> print(output_image)
[16, 17, 18]

"""

from __future__ import annotations
import itertools
import operator
import random
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from pint import Quantity
import skimage
import skimage.measure

from . import units
from .backend import config
from .backend.core import DeepTrackNode
from .backend.units import ConversionTable, create_context
from .image import Image
from .properties import PropertyDict
from .sources import SourceItem
from .types import ArrayLike, PropertyLike


#TODO: for all features check whether image should be Image, np.ndarray, or both.

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
    _input : 'Image' or List['Image'], optional
        Defines a list of DeepTrackNode objects that calculate the input of the 
        feature. In most cases, this can be left empty.
    **kwargs : Dict[str, Any]
        All Keyword arguments will be wrapped as instances of `Property` and
        included in the `properties` attribute.

    Attributes
    ----------
    properties : PropertyDict
        A dict that contains all keyword arguments passed to the constructor 
        wrapped as Distributions. A sampled copy of this dict is sent as input 
        to the get function, and is appended to the properties field of the 
        output image.
    __list_merge_strategy__ : int
        Controls how the output of `.get(image, **kwargs)` is merged with the 
        input list. It can be `MERGE_STRATEGY_OVERRIDE` (0, default), where the 
        input is replaced by the new list, or `MERGE_STRATEGY_APPEND` (1), 
        where the new list is appended to the end of the input list.
    __distributed__ : bool
        Controls whether `.get(image, **kwargs)` is called on each element in 
        the list separately (`__distributed__ = True`), or if it is called on 
        the list as a whole (`__distributed__ = False`).
    __property_memorability__ : int
        Controls whether to store the features properties to the `Image`.
        Values 1 or lower will be included by default.
    __conversion_table__ : ConversionTable
        A ConversionTable that is used to convert properties of the feature to
        the desired units.
    __gpu_compatible__ : bool
        Controls whether to use GPU acceleration for the feature.

    Methods
    -------
    get(
        image: Union['Image', List['Image']], **kwargs: Any
    ) -> Union['Image', List['Image']]
        Abstract method that defines how the feature transforms the input.
    __call__(image_list: Optional[Union[Image, List[Image]]] = None, 
             _ID: Tuple[int, ...] = (), **kwargs: Any) -> Any
        Executes the feature or pipeline on the input and applies property 
        overrides from `kwargs`.
    store_properties(x: bool = True, recursive: bool = True) -> None
        Controls whether the properties are stored in the output `Image` object.
    torch(dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, 
          permute_mode: str = "never") -> 'Feature'
        Converts the feature into a PyTorch-compatible feature.
    batch(batch_size: int = 32) -> Union[tuple, List[Image]]
        Batches the feature for repeated execution.
    seed(_ID: Tuple[int, ...] = ()) -> None
        Sets the random seed for the feature, ensuring deterministic behavior.

    """


    # Attributes.
    properties: 'PropertyDict'
    _input: 'DeepTrackNode'
    _random_seed: 'DeepTrackNode'
    arguments: Optional['Feature']

    __list_merge_strategy__ = MERGE_STRATEGY_OVERRIDE
    __distributed__ = True
    __property_memorability__ = 1
    __conversion_table__ = ConversionTable()
    __gpu_compatible__ = False

    _wrap_array_with_image: bool = False

    def __init__(
        self: Feature,
        _input: Any = [],
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize a new Feature instance.

        Parameters
        ----------
        _input : Any, optional
            The initial input(s) for the feature, often images or other data. 
            Defaults to an empty list.
        **kwargs : Dict[str, Any]
            Keyword arguments, each turned into a `Property` and stored in 
            `self.properties`.
        
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
        self._random_seed = DeepTrackNode(lambda: random.randint(0, 2147483648))
        self._random_seed.add_child(self)
        # self.add_dependency(self._random_seed)  # Executed by add_child.

        # Initialize arguments to None.
        self.arguments = None

    def get(
        self: Feature,
        image: Image | List[Image],
        **kwargs: Dict[str, Any],
    ) -> Image | List[Image]:
        """Transform an image [abstract method].
        
        Abstract method that defines how the feature transforms the input. The 
        current value of all properties will be passed as keyword arguments.

        Parameters
        ---------
        image : Image or List[Image]
            The Image or list of images to transform.
        **kwargs : Dict[str, Any]
            The current value of all properties in `properties` as well as any 
            global arguments.

        Returns
        -------
        Image or List[Image]
            The transformed image or list of images.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.

        """

        raise NotImplementedError

    def __call__(
        self,
        image_list: Union['Image', List['Image']] = None,
        _ID: Tuple[int, ...] = (),
        **kwargs: Dict[str, Any],
    ) -> Any:
        """Execute the feature or pipeline.

        This method executes the feature or pipeline on the provided input and 
        updates the computation graph if necessary. It handles overriding 
        properties using additional keyword arguments.

        The actual computation is performed by calling the parent `__call__` 
        method in the `DeepTrackNode` class, which manages lazy evaluation and 
        caching.

        Arguments
        ---------
        image_list : 'Image' or List['Image'], optional
            The input to the feature or pipeline. If `None`, the feature uses 
            previously set input values or propagates properties.
        **kwargs : Dict[str, Any]
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
        self,
        toggle: bool = True,
        recursive: bool = True,
    ) -> None:
        """Control whether to return an Image object.
        
        If selected `True`, the output of the evaluation of the feature is an 
        Image object that also contains the properties.

        Parameters
        ----------
        toggle : bool
            If `True`, store properties. If `False`, do not store.
        recursive : bool
            If `True`, also set the same behavior for all dependent features.

        """

        self._wrap_array_with_image = toggle

        if recursive:
            for dependency in self.recurse_dependencies():
                if isinstance(dependency, Feature):
                    dependency.store_properties(toggle, recursive=False)

    def torch(
        self, 
        dtype=None, 
        device=None, 
        permute_mode: str = "never",
    ) -> 'Feature':
        """Convert the feature to a PyTorch feature.

        Parameters
        ----------
        dtype : torch.dtype, optional
            The data type of the output.
        device : torch.device, optional
            The target device of the output (e.g., CPU or GPU).
        permute_mode : str
            Controls whether to permute image axes for PyTorch. 
            Defaults to "never".

        Returns
        -------
        Feature
            The transformed, PyTorch-compatible feature.

        """

        from .pytorch import ToTensor

        tensor_feature = ToTensor(
            dtype=dtype, 
            device=device, 
            permute_mode=permute_mode,
        )
        
        tensor_feature.store_properties(False, recursive=False)
        
        return self >> tensor_feature

    def batch(self, batch_size: int = 32) -> Union[tuple, List['Image']]:
        """Batch the feature.

        It produces a batch of outputs by repeatedly calling `update()` and 
        `__call__()`.

        Parameters
        ----------
        batch_size : int
            Number of times to sample or generate data.

        Returns
        -------
        tuple or List['Image']
            A tuple of stacked arrays (if the outputs are numpy arrays or 
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
        self,
        _ID: Tuple[int, ...] = (),
    ) -> Union['Image', List['Image']]:
        """Core logic to create or transform the image.
        
        It creates or transforms the input image by calling the method `get()` 
        with the correct inputs.

        Parameters
        ----------
        _ID : Tuple[int, ...]
            The ID of the current execution.

        Returns
        -------
        Image or List[Image]
            The resolved image or list of images.

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

    def __use_gpu__(self, inp, **_):
        """Determine if the feature should use the GPU.
        
        Parameters
        ----------
        inp : Image
            The input image to check.
        **_ : Any
            Additional arguments (unused).

        Returns
        -------
        bool
            True if GPU acceleration is enabled and beneficial, otherwise 
            False.

        """

        return self.__gpu_compatible__ and np.prod(np.shape(inp)) > (90000)

    def update(self, **global_arguments) -> 'Feature':
        """Refresh the feature to create a new image.

        Per default, when a feature is called multiple times, it will return 
        the same value. To tell the feature to return a new value, first call 
        `update()`.
        
        Returns
        -------
        Feature
            The updated feature.

        """

        if global_arguments:
            # Deptracated, but not necessary to raise hard error.
            warnings.warn(
                "Passing information through .update is no longer supported. "
                "A quick fix is to pass the information when resolving the feature. "
                "The prefered solution is to use dt.Arguments",
                DeprecationWarning,
            )

        super().update()

        return self

    def add_feature(self, feature: 'Feature') -> 'Feature':
        """Adds a feature to the dependecy graph of this one.

        Parameters
        ----------
        feature : Feature
            The feature to add as a dependency.

        Returns
        -------
        Feature
            The newly added feature (for chaining).

        """

        feature.add_child(self)
        # self.add_dependency(feature)  # Already done by add_child().

        return feature

    def seed(self, _ID: Tuple[int, ...] = ()) -> None:
        """Seed the random number generator.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            Unique identifier for parallel evaluations.

        """

        np.random.seed(self._random_seed(_ID=_ID))

    def bind_arguments(self, arguments: 'Feature') -> 'Feature':
        """Bind another feature’s properties as arguments to this feature.

        Often used internally by advanced features or pipelines. 

        See `features.Arguments`.

        """

        self.arguments = arguments

        return self

    def _normalize(self, **properties):
        # Handles all unit normalizations and conversions
        for cl in type(self).mro():
            if hasattr(cl, "__conversion_table__"):
                properties = cl.__conversion_table__.convert(**properties)

        for key, val in properties.items():
            if isinstance(val, Quantity):
                properties[key] = val.magnitude
        return properties

    def plot(
        self,
        input_image: Image or List[Image] = None,
        resolve_kwargs: dict = None,
        interval: float = None,
        **kwargs
    ):
        """Visualizes the output of the feature.

        Resolves the feature and visualizes the result. If the output is an Image,
        show it using `pyplot.imshow`. If the output is a list, create an `Animation`.
        For notebooks, the animation is played inline using `to_jshtml()`. For scripts,
        the animation is played using the matplotlib backend.

        Any parameters in kwargs will be passed to `pyplot.imshow`.

        Parameters
        ----------
        input_image : Image or List[Image], optional
            Passed as argument to `resolve` call
        resolve_kwargs : dict, optional
            Passed as kwarg arguments to `resolve` call
        interval : float
            The time between frames in animation in ms. Default 33.
        kwargs
            keyword arguments passed to the method pyplot.imshow()
        """

        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
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

    def _process_properties(self, propertydict) -> dict:
        # Optional hook for subclasses to preprocess input before calling
        # the method .get()

        propertydict = self._normalize(**propertydict)
        return propertydict

    def _activate_sources(self, x):
        if isinstance(x, SourceItem):
            x()
        else:
            if isinstance(x, list):
                for source in x:
                    if isinstance(source, SourceItem):
                        source()

    def __getattr__(self, key: str) -> Any:
        """Custom attribute access for the Feature class.

        This method allows properties of the `Feature` instance to be accessed 
        as if they were attributes. For example, `feature.my_property` is 
        equivalent to `feature.properties["my_property"]`.
        
        Specifically, it checks if the requested 
        attribute (`key`) exists in the `properties` dictionary of the instance 
        and returns the corresponding value if found. If the `key` does not 
        exist in `properties`, or if the `properties` attribute is not set, an 
        `AttributeError` is raised.

        Parameters
        ----------
        key : str
            The name of the attribute being accessed.

        Returns
        -------
        Any
            The value of the property corresponding to the given `key` in the 
            `properties` dictionary.

        Raises
        ------
        AttributeError
            If `properties` is not defined for the instance, or if the `key` 
            does not exist in `properties`.

        """

        if "properties" in self.__dict__:
            properties = self.__dict__["properties"]
            if key in properties:
                return properties[key]

        raise AttributeError(f"'{self.__class__.__name__}' object has "
                             "no attribute '{key}'")

    def __iter__(self):
        while True:
            yield from next(self)

    def __next__(self):
        yield self.update().resolve()

    def __rshift__(self, other) -> 'Feature':
        # Allows chaining of features. For example,
        # feature1 >> feature2 >> feature3
        # or
        # feature1 >> some_function

        if isinstance(other, DeepTrackNode):
            return Chain(self, other)

        # If other is a function, call it on the output of the feature.
        # For example, feature >> some_function
        if callable(other):
            return self >> Lambda(lambda: other)

        # The operator is not implemented for other inputs.
        return NotImplemented

    def __rrshift__(self, other: 'Feature') -> 'Feature':
        # Allows chaining of features. For example,
        # some_function << feature1 << feature2
        # or
        # some_function << feature1

        if isinstance(other, Feature):
            return Chain(other, self)

        if isinstance(other, DeepTrackNode):
            return Chain(Value(other), self)

        return NotImplemented

    def __add__(self, other) -> 'Feature':
        # Overrides add operator
        return self >> Add(other)

    def __radd__(self, other) -> 'Feature':
        # Overrides add operator
        return Value(other) >> Add(self)

    def __sub__(self, other) -> 'Feature':
        # Overrides add operator
        return self >> Subtract(other)

    def __rsub__(self, other) -> 'Feature':
        # Overrides add operator
        return Value(other) >> Subtract(self)

    def __mul__(self, other) -> 'Feature':
        return self >> Multiply(other)

    def __rmul__(self, other) -> 'Feature':
        return Value(other) >> Multiply(self)

    def __truediv__(self, other) -> 'Feature':
        return self >> Divide(other)

    def __rtruediv__(self, other) -> 'Feature':
        return Value(other) >> Divide(self)

    def __floordiv__(self, other) -> 'Feature':
        return self >> FloorDivide(other)

    def __rfloordiv__(self, other) -> 'Feature':
        return Value(other) >> FloorDivide(self)

    def __pow__(self, other) -> 'Feature':
        return self >> Power(other)

    def __rpow__(self, other) -> 'Feature':
        return Value(other) >> Power(self)

    def __gt__(self, other) -> 'Feature':
        return self >> GreaterThan(other)

    def __rgt__(self, other) -> 'Feature':
        return Value(other) >> GreaterThan(self)

    def __lt__(self, other) -> 'Feature':
        return self >> LessThan(other)

    def __rlt__(self, other) -> 'Feature':
        return Value(other) >> LessThan(self)

    def __le__(self, other) -> 'Feature':
        return self >> LessThanOrEquals(other)

    def __rle__(self, other) -> 'Feature':
        return Value(other) >> LessThanOrEquals(self)

    def __ge__(self, other) -> 'Feature':
        return self >> GreaterThanOrEquals(other)

    def __rge__(self, other) -> 'Feature':
        return Value(other) >> GreaterThanOrEquals(self)

    def __xor__(self, other) -> 'Feature':
        return Repeat(self, other)

    def __and__(self, other) -> 'Feature':
        return self >> Stack(other)

    def __rand__(self, other) -> 'Feature':
        return Value(other) >> Stack(self)

    def __getitem__(self, slices) -> 'Feature':
        # Allows direct slicing of the data.
        if not isinstance(slices, tuple):
            slices = (slices,)

        # We make it a list to ensure that each element is sampled independently.
        slices = list(slices)

        return self >> Slice(slices)

    # private properties to dispatch based on config
    @property
    def _format_input(self):
        if self._wrap_array_with_image:
            return self._image_wrapped_format_input
        else:
            return self._no_wrap_format_input

    @property
    def _process_and_get(self):
        if self._wrap_array_with_image:
            return self._image_wrapped_process_and_get
        else:
            return self._no_wrap_process_and_get

    @property
    def _process_output(self):
        if self._wrap_array_with_image:
            return self._image_wrapped_process_output
        else:
            return self._no_wrap_process_output

    def _image_wrapped_format_input(self, image_list, **kwargs) -> List[Image]:
        # Ensures the input is a list of Image.

        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]

        inputs = [(Image(image)) for image in image_list]
        return self._coerce_inputs(inputs, **kwargs)

    def _no_wrap_format_input(self, image_list, **kwargs) -> list:
        # Ensures the input is a list of Image.

        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]

        return image_list

    def _no_wrap_process_and_get(self, image_list, **feature_input) -> list:
        # Controls how the get function is called

        if self.__distributed__:
            # Call get on each image in list, and merge properties from corresponding image
            return [self.get(x, **feature_input) for x in image_list]

        else:
            # Call get on entire list.
            new_list = self.get(image_list, **feature_input)

            if not isinstance(new_list, list):
                new_list = [new_list]

            return new_list

    def _image_wrapped_process_and_get(self, image_list, **feature_input) -> List[Image]:
        # Controls how the get function is called

        if self.__distributed__:
            # Call get on each image in list, and merge properties from corresponding image

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

    def _image_wrapped_process_output(self, image_list, feature_input):
        for index, image in enumerate(image_list):

            if self.arguments:
                image.append(self.arguments.properties())

            image.append(feature_input)

    def _no_wrap_process_output(self, image_list, feature_input):
        for index, image in enumerate(image_list):

            if isinstance(image, Image):
                image_list[index] = image._value

    def _coerce_inputs(self, inputs, **kwargs):
        # Coerces inputs to the correct type (numpy array or tensor or cupyy array).
        if config.gpu_enabled:

            return [
                i.to_cupy()
                if (not self.__distributed__) and self.__use_gpu__(i, **kwargs)
                else i.to_numpy()
                for i in inputs
            ]

        else:
            return [i.to_numpy() for i in inputs]


def propagate_data_to_dependencies(feature: Feature, **kwargs: Dict[str, Any]):
    """Updates the properties of dependencies in a feature's dependency tree.

    This function iterates over all the dependencies of the given feature and 
    sets the values of their properties based on the provided keyword 
    arguments.

    This function ensures that the properties in the dependency tree are 
    dynamically updated based on the provided data.

    Properties are only updated if the `key` exists in the `PropertyDict` of 
    the dependency.

    Parameters
    ----------
    feature : Feature
        The feature whose dependencies are to be updated. The dependencies are 
        recursively traversed to ensure all relevant nodes in the dependency 
        tree are considered.
    **kwargs : Dict[str, Any]
        Key-value pairs specifying the property names (`key`) and their 
        corresponding values (`value`) to be set in the dependencies. Only 
        properties that exist in the dependencies are updated.


    """

    for dep in feature.recurse_dependencies():
        if isinstance(dep, PropertyDict):
            for key, value in kwargs.items():
                if key in dep:
                    dep[key].set_value(value)


class StructuralFeature(Feature):
    """Provides the structure of a feature set without input transformations.
    
    Because it does not add new properties or affect data distribution, 
    `StructuralFeature` is often used as a logical or organizational tool 
    rather than a direct image transformer.
    
    Since StructuralFeature doesn’t override __init__ or get, it just inherits 
    the behavior of Feature.

    Attributes
    ----------
    __property_verbosity__ : int
        Controls whether this feature’s properties are included in the 
        output image’s property list. A value of 2 means do not include.
    __distributed__ : bool
        If set to False, the feature’s `get` method is called on the 
        entire list rather than each element individually.

    """

    __property_verbosity__: int = 2  # Hide properties from logs or output.
    __distributed__: bool = False  # Process the entire image list in one call.


class Chain(StructuralFeature):
    """Resolve two features sequentially.
    
    This feature resolves two features sequentially, passing the output of the 
    first feature as the input to the second.

    Parameters
    ----------
    feature_1 : Feature
        The first feature in the chain. Its output is passed to `feature_2`.
    feature_2 : Feature
        The second feature in the chain, which processes the output from 
        `feature_1`.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `StructuralFeature` 
        (and thus `Feature`).
    
    Example
    -------
    Let us create a feature chain where the first feature adds an offset to an 
    image, and the second feature multiplies it by a constant.

    >>> import numpy as np
    >>> from deeptrack.features import Chain, Feature
    
    Define the features:
    
    >>> class Addition(Feature):
    ...     '''Simple feature that adds a constant.'''
    ...     def get(self, image, **kwargs):
    ...         # 'addend' property set via self.properties (default: 0).
    ...         return image + self.properties.get("addend", 0)()

    >>> class Multiplication(Feature):
    ...     '''Simple feature that multiplies by a constant.'''
    ...     def get(self, image, **kwargs):
    ...         # 'multiplier' property set via self.properties (default: 1).
    ...         return image * self.properties.get("multiplier", 1)()

    Chain the features:

    >>> A = Addition(addend=10)
    >>> M = Multiplication(multiplier=0.5)
    >>> chain = A >> M  
    
    Equivalent to: 
    
    >>> chain = Chain(A, M)

    Create a dummy image:
    
    >>> dummy_image = np.ones((60, 80))

    Apply the chained features:
    
    >>> transformed_image = chain(dummy_image)

    In this example, the image is first passed through the `Addition` feature 
    to add an offset, and then through the `Multiplication` feature to multiply
    by a constant factor.

    """

    def __init__(
        self,
        feature_1: Feature,
        feature_2: Feature,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the chain with two sub-features.

        Both `feature_1` and `feature_2` are added as dependencies of the 
        chain, ensuring that updates to them propagate correctly through the 
        DeepTrack computation graph.

        Parameters
        ----------
        feature_1 : Feature
            The first feature to be applied.
        feature_2 : Feature
            The second feature, applied after `feature_1`.
        **kwargs : Dict[str, Any]
            Passed to the parent constructor (e.g., name, properties).

        """

        super().__init__(**kwargs)

        self.feature_1 = self.add_feature(feature_1)
        self.feature_2 = self.add_feature(feature_2)

    def get(
        self,
        image: Union['Image', List['Image']],
        _ID: Tuple[int, ...] = (),
        **kwargs: Dict[str, Any],
    ) -> Union['Image', List['Image']]:
        """Apply the two features in sequence on the given input image(s).

        Parameters
        ----------
        image : Image or List[Image]
            The input data (image or list of images) to transform.
        _ID : Tuple[int, ...], optional
            A unique identifier for caching/parallel calls.
        **kwargs : Dict[str, Any]
            Additional parameters passed to or sampled by the features. 
            Generally unused here, since each sub-feature will fetch 
            what it needs from their own properties.

        Returns
        -------
        Image or List[Image]
            The final output after `feature_1` and then `feature_2` have 
            processed the input.

        """

        image = self.feature_1(image, _ID=_ID)
        image = self.feature_2(image, _ID=_ID)
        return image


Branch = Chain  # Alias for backwards compatability.


class DummyFeature(Feature):
    """A no-op feature that simply returns the input unchanged.

    This class can serve as a container for properties that don't directly 
    transform the data, but need to be logically grouped. Since it inherits 
    from `Feature`, any keyword arguments passed to the constructor are 
    stored as `Property` instances in `self.properties`, enabling dynamic 
    behavior or parameterization without performing any transformations 
    on the input data.

    Parameters
    ----------
    _input : Union['Image', List['Image']], optional
        An optional input (image or list of images) that can be set for 
        the feature. By default, an empty list.
    **kwargs : Dict[str, Any]
        Additional keyword arguments are wrapped as `Property` instances and 
        stored in `self.properties`.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import DummyFeature

    Create an image:
    >>> dummy_image = np.ones((60, 80))

    Initialize the DummyFeature:
    
    >>> dummy_feature = DummyFeature(value=42)

    Pass the image through the DummyFeature:
    
    >>> output_image = dummy_feature(dummy_image)

    The output should be identical to the input:
    
    >>> np.array_equal(dummy_image, output_image)
    True

    Access the properties stored in DummyFeature:
    
    >>> dummy_feature.properties["value"]()
    42

    This example illustrates that the `DummyFeature` can act as a container
    for properties, while the data itself remains unaltered.

    """

    def get(
        self, 
        image: Union['Image', List['Image']], 
        **kwargs: Any,
    )-> Union['Image', List['Image']]:
        """Return the input image or list of images unchanged.

        Parameters
        ----------
        image : 'Image' or List['Image']
            The image(s) to pass through without modification.
        **kwargs : Any
            Additional properties sampled from `self.properties` or passed 
            externally. These are unused here but provided for consistency 
            with the `Feature` interface.

        Returns
        -------
        'Image' or List['Image']
            The same `image` object that was passed in.

        """

        return image


class Value(Feature):
    """Represents a constant (per evaluation) value in a DeepTrack pipeline.

    This feature can hold a single value (e.g., a scalar) and supply it 
    on-demand to other parts of the pipeline. It does not transform the input 
    image. Instead, it returns the stored scalar (or array) value.

    Parameters
    ----------
    value : PropertyLike[float], optional
        The numerical value to store. Defaults to 0. If an `Image` is provided, 
        a warning is issued suggesting convertion to a NumPy array for 
        performance reasons.
    **kwargs : Dict[str, Any]
        Additional named properties passed to the `Feature` constructor.

    Attributes
    ----------
    __distributed__ : bool
        Set to False, indicating that this feature’s `get(...)` method
        processes the entire list of images (or data) at once, rather than
        distributing calls for each item.

    Examples
    --------
    >>> from deeptrack.features import Value
    
    >>> value = Value(42)
    >>> print(value())
    42

    Overriding the value at call time:
    
    >>> print(value(value=100))
    100

    >>> print(value())
    100
        
    """

    # Attributes.
    __distributed__: bool = False  # Process as a single batch.


    def __init__(
        self, 
        value: PropertyLike[float] = 0, 
        **kwargs: Dict[str, Any],
    ):
        """
        Create a `Value` feature that stores a constant value.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The initial value for this feature. If it's an Image, a warning 
            will be raised. Defaults to 0.
        **kwargs : dict
            Additional properties passed to `Feature` constructor (e.g., name).
        
        """

        if isinstance(value, Image):
            warnings.warn(
                "Setting dt.Value value as an Image object is likely to lead "
                "to performance deterioation. Consider converting it to a "
                "numpy array using np.array."
            )

        super().__init__(value=value, **kwargs)

    def get(self, image: Any, value: float, **kwargs: Dict[str, Any]) -> float:
        """Return the stored value, ignoring the input image.

        Parameters
        ----------
        image : Any
            Input data that would normally be transformed by a feature, 
            but `Value` does not modify or use it.
        value : float
            The current (possibly overridden) value stored in this feature.
        **kwargs : Dict[str, Any]
            Additional properties or overrides that are not used here.

        Returns
        -------
        float
            The `value` property, unchanged.

        """

        return value


class ArithmeticOperationFeature(Feature):
    """Applies an arithmetic operation element-wise to inputs.

    This feature performs an arithmetic operation (e.g., addition, subtraction, 
    multiplication, etc.) on the input data. The inputs can be single values or 
    lists of values. If a list is passed, the operation is applied to each 
    element in the list. When both inputs are lists of different lengths, the 
    shorter list is cycled.

    Parameters
    ----------
    op : Callable
        The arithmetic operation to apply. This can be a built-in operator or a 
        custom callable.
    value : float or int or List[float or int], optional
        The other value(s) to apply the operation with. Defaults to 0.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature`.

    Attributes
    ----------
    __distributed__ : bool
        Set to `False`, indicating that this feature’s `get(...)` method 
        processes the entire list of images or values at once, rather than 
        distributing calls for each item.
    __gpu_compatible__ : bool
        Set to `True`, indicating compatibility with GPU processing.

    Example
    -------
    >>> import operator
    >>> import numpy as np
    >>> from deeptrack.features import ArithmeticOperationFeature

    Define a simple addition operation:
    
    >>> addition = ArithmeticOperationFeature(operator.add, value=10)

    Create a list of input values:
    
    >>> input_values = [1, 2, 3, 4]

    Apply the operation:
    
    >>> output_values = addition(input_values)
    >>> print(output_values)
    [11, 12, 13, 14]

    In this example, each value in the input list is incremented by 10.
    
    """

    __distributed__: bool = False
    __gpu_compatible__: bool = True


    def __init__(
        self,
        op: Callable[[Any, Any], Any],
        value: Union[float, int, List[Union[float, int]]] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the ArithmeticOperationFeature.

        Parameters
        ----------
        op : Callable[[Any, Any], Any]
            The operation to apply, such as `operator.add` or `operator.mul`.
        value : float or int or List[float or int], optional
            The value(s) to apply the operation with. Defaults to 0.
        **kwargs : Any
            Additional arguments passed to the parent `Feature` constructor.

        """
        self.op = op
        super().__init__(value=value, **kwargs)

    def get(
        self,
        image: Union[Any, List[Any]],
        value: Union[float, int, List[Union[float, int]]],
        **kwargs: Any,
    ) -> List[Any]:
        """Apply the operation element-wise to the input data.

        Parameters
        ----------
        image : Any or List[Any]
            The input data (list or single value) to transform.
        value : float or int or List[float or int]
            The value(s) to apply the operation with. If a single value is 
            provided, it is broadcast to match the input size. If a list is 
            provided, it will be cycled to match the length of the input list.
        **kwargs : Dict[str, Any]
            Additional parameters or overrides (unused here).

        Returns
        -------
        List[Any]
            A list with the result of applying the operation to the input.
            
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
    value : PropertyLike[int or float], optional
        The value to add to the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is incremented by 5.

    Start by creating a pipeline using Add:

    >>> from deeptrack.features import Add, Value
    
    >>> pipeline = Value([1, 2, 3]) >> Add(value=5)
    >>> pipeline.resolve()
    [6, 7, 8]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) + 5
    
    >>> pipeline = 5 + Value([1, 2, 3])
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> add_feature = Add(value=5)
    >>> pipeline = add_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Add feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to add to the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.add, value=value, **kwargs)


class Subtract(ArithmeticOperationFeature):
    """Subtract a value from the input.

    This feature performs element-wise subtraction (-) from the input.
    
    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to subtract from the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is decreased by 2.

    Start by creating a pipeline using Subtract:

    >>> from deeptrack.features import Subtract, Value
    
    >>> pipeline = Value([1, 2, 3]) >> Subtract(value=2)
    >>> pipeline.resolve()
    [-1, 0, 1]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) - 2
    
    >>> pipeline = - 2 + Value([1, 2, 3])
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> sub_feature = Subtract(value=2)
    >>> pipeline = sub_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Subtract feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to subtract from the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.sub, value=value, **kwargs)


class Multiply(ArithmeticOperationFeature):
    """Multiply the input by a value.

    This feature performs element-wise multiplication (*) of the input.
    
    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to multiply the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is multiplied by 5.

    Start by creating a pipeline using Multiply:

    >>> from deeptrack.features import Multiply, Value
    
    >>> pipeline = Value([1, 2, 3]) >> Multiply(value=5)
    >>> pipeline.resolve()
    [5, 10, 15]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) * 5
    
    >>> pipeline = 5 * Value([1, 2, 3])
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> sub_feature = Multiply(value=5)
    >>> pipeline = sub_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Multiply feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to multiply the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.mul, value=value, **kwargs)


class Divide(ArithmeticOperationFeature):
    """Divide the input with a value.

    This feature performs element-wise division (/) of the input.
    
    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to divide the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is divided by 5.

    Start by creating a pipeline using Divide:

    >>> from deeptrack.features import Divide, Value
    
    >>> pipeline = Value([1, 2, 3]) >> Divide(value=5)
    >>> pipeline.resolve()
    [0.2 0.4 0.6]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) / 5
    
    >>> pipeline = 5 / Value([1, 2, 3])  # Different result.
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> truediv_feature = Divide(value=5)
    >>> pipeline = truediv_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Divide feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to divide the input. Defaults to 0.
        **kwargs : Any
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
    value : PropertyLike[int or float], optional
        The value to floor-divide the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is floor-divided by 5.

    Start by creating a pipeline using FloorDivide:

    >>> from deeptrack.features import FloorDivide, Value
    
    >>> pipeline = Value([-3, 3, 6]) >> FloorDivide(value=5)
    >>> pipeline.resolve()
    [0.2 0.4 0.6]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([-3, 3, 6]) // 5
    
    >>> pipeline = 5 // Value([-3, 3, 6])  # Different result.
    
    Or, most explicitly:
    
    >>> input_value = Value([-3, 3, 6])
    >>> floordiv_feature = FloorDivide(value=5)
    >>> pipeline = floordiv_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the FloorDivide feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to fllor-divide the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.floordiv, value=value, **kwargs)


class Power(ArithmeticOperationFeature):
    """Raise the input to a power.

    This feature performs element-wise power (**) of the input.

    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to take the power of the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is elevated to the 3.

    Start by creating a pipeline using Power:

    >>> from deeptrack.features import Power, Value
    
    >>> pipeline = Value([1, 2, 3]) >> Power(value=3)
    >>> pipeline.resolve()
    [1, 8, 27]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) ** 3
    
    >>> pipeline = 3 ** Value([1, 2, 3])  # Different result.
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> pow_feature = Power(value=3)
    >>> pipeline = pow_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Power feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to take the power of the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.pow, value=value, **kwargs)


class LessThan(ArithmeticOperationFeature):
    """Determine whether input is less than value.

    This feature performs element-wise comparison (<) of the input.

    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to compare (<) with the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is compared (<) with 2.

    Start by creating a pipeline using LessThan:

    >>> from deeptrack.features import LessThan, Value
    
    >>> pipeline = Value([1, 2, 3]) >> LessThan(value=2)
    >>> pipeline.resolve()
    [ True False False]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) < 2
    
    >>> pipeline = 2 < Value([1, 2, 3])  # Different result.
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> lt_feature = LessThan(value=2)
    >>> pipeline = lt_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the LessThan feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to compare (<) with the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.lt, value=value, **kwargs)


class LessThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is less than or equal to value.

    This feature performs element-wise comparison (<=) of the input.

    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to compare (<=) with the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is compared (<=) with 2.

    Start by creating a pipeline using LessThanOrEquals:

    >>> from deeptrack.features import LessThanOrEquals, Value
    
    >>> pipeline = Value([1, 2, 3]) >> LessThanOrEquals(value=2)
    >>> pipeline.resolve()
    [ True  True False]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) <= 2
    
    >>> pipeline = 2 <= Value([1, 2, 3])  # Different result.
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> le_feature = LessThanOrEquals(value=2)
    >>> pipeline = le_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the LessThanOrEquals feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to compare (<=) with the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.le, value=value, **kwargs)


LessThanOrEqual = LessThanOrEquals


class GreaterThan(ArithmeticOperationFeature):
    """Determine whether input is greater than value.

    This feature performs element-wise comparison (>) of the input.

    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to compare (>) with the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is compared (>) with 2.

    Start by creating a pipeline using GreaterThan:

    >>> from deeptrack.features import GreaterThan, Value
    
    >>> pipeline = Value([1, 2, 3]) >> GreaterThan(value=2)
    >>> pipeline.resolve()
    [False False  True]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) > 2
    
    >>> pipeline = 2 > Value([1, 2, 3])  # Different result.
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> gt_feature = GreaterThan(value=2)
    >>> pipeline = gt_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the GreaterThan feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to compare (>) with the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.gt, value=value, **kwargs)


class GreaterThanOrEquals(ArithmeticOperationFeature):
    """Determine whether input is greater than or equal to value.

    This feature performs element-wise comparison (>=) of the input.

    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to compare (<=) with the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    Example
    -------
    In this example, each element in the input array is compared (>=) with 2.

    Start by creating a pipeline using GreaterThanOrEquals:

    >>> from deeptrack.features import GreaterThanOrEquals, Value
    
    >>> pipeline = Value([1, 2, 3]) >> GreaterThanOrEquals(value=2)
    >>> pipeline.resolve()
    [False  True  True]
    
    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) >= 2
    
    >>> pipeline = 2 >= Value([1, 2, 3])  # Different result.
    
    Or, most explicitly:
    
    >>> input_value = Value([1, 2, 3])
    >>> ge_feature = GreaterThanOrEquals(value=2)
    >>> pipeline = ge_feature(input_value)

    """

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the GreaterThanOrEquals feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to compare (>=) with the input. Defaults to 0.
        **kwargs : Any
            Additional keyword arguments.

        """

        super().__init__(operator.ge, value=value, **kwargs)


GreaterThanOrEqual = GreaterThanOrEquals


class Equals(ArithmeticOperationFeature):
    """Determine whether input is equal to value.

    This feature performs element-wise comparison (==) of the input.

    Parameters
    ----------
    value : PropertyLike[int or float], optional
        The value to compare (==) with the input. Defaults to 0.
    **kwargs : Any
        Additional keyword arguments passed to the parent constructor.

    """

    #TODO: Example for Equals.
    #TODO: Why Equals behaves differently from the other operators?
    #TODO: Why __eq__ and __req__ are not defined in DeepTrackNode and Feature?

    def __init__(
        self,
        value: PropertyLike[float] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Equals feature.

        Parameters
        ----------
        value : PropertyLike[float], optional
            The value to compare (==) with the input. Defaults to 0.
        **kwargs : Any
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
    value : PropertyLike[Any]
        The feature or data to stack with the input.
    **kwargs : Dict[str, Any]
        Additional arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__ : bool
        Indicates whether this feature distributes computation across inputs. 
        Always `False` for `Stack`, as it processes all inputs at once.

    Example
    -------
    Start by creating a pipeline using Stack:

    >>> from deeptrack.features import Stack, Value
    
    >>> pipeline = Value([1, 2, 3]) >> Stack(value=[4, 5])
    >>> print(pipeline.resolve())
    [1, 2, 3, 4, 5]

    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Value([1, 2, 3]) & [4, 5]

    >>> pipeline = [4, 5] & Value([1, 2, 3])  # Different result.

    """

    __distributed__: bool = False

    def __init__(
        self,
        value: PropertyLike[Any],
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Stack feature.

        Parameters
        ----------
        value : PropertyLike[Any]
            The feature or data to stack with the input.
        **kwargs : Dict[str, Any]
            Additional arguments passed to the parent `Feature` class.
        """

        super().__init__(value=value, **kwargs)

    def get(
        self,
        image: Union[Any, List[Any]],
        value: Union[Any, List[Any]],
        **kwargs: Dict[str, Any],
    ) -> List[Any]:
        """Concatenate the input with the value.

        It ensures that both the input (`image`) and the value (`value`) are 
        treated as lists before concatenation.

        Parameters
        ----------
        image : Any or List[Any]
            The input data to stack. Can be a single element or a list.
        value : Any or List[Any]
            The feature or data to stack with the input. Can be a single 
            element or a list.
        **kwargs : Dict[str, Any]
            Additional keyword arguments (not used here).

        Returns
        -------
        List[Any]
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
    get(image, **kwargs)
        Passes the input image through unchanged, while allowing for property 
        overrides.

    Examples
    --------
    A typical use-case is:

    >>> arguments = Arguments(is_label=False)
    >>> image_loader = (
    ...     LoadImage(path="./image.png") >>
    ...     GaussianNoise(sigma = (1 - arguments.is_label) * 5)
    ...     )
    >>> image_loader.bind_arguments(arguments)

    >>> image_loader()  # Image with added noise.
    >>> image_loader(is_label=True)  # Raw image with no noise.

    For a non-mathematical dependence, create a local link to the property as 
    follows:

    >>> arguments = Arguments(is_label=False)
    >>> image_loader = (
    ...     LoadImage(path="./image.png") >>
    ...     GaussianNoise(
    ...         is_label=arguments.is_label,
    ...         sigma=lambda is_label: 0 if is_label else 5
    ...     )
    ... )
    >>> image_loader.bind_arguments(arguments)

    >>> image_loader()              # Image with added noise
    >>> image_loader(is_label=True) # Raw image with no noise

    Keep in mind that, if any dependent property is non-deterministic, they may 
    permanently change:
    
    >>> arguments = Arguments(noise_max_sigma=5)
    >>> image_loader = (
    ...     LoadImage(path="./image.png") >>
    ...     GaussianNoise(
    ...         noise_max_sigma=5,
    ...         sigma=lambda noise_max_sigma: rand() * noise_max_sigma
    ...     )
    ... )

    >>> image_loader.bind_arguments(arguments)

    >>> image_loader().get_property("sigma") # Example: 3.27...
    >>> image_loader(noise_max_sigma=0) # 0
    >>> image_loader().get_property("sigma") # Example: 1.93...

    As with any feature, all arguments can be passed by deconstructing the 
    properties dict:

    >>> arguments = Arguments(is_label=False, noise_sigma=5)
    >>> image_loader = (
    ...     LoadImage(path="./image.png") >>
    ...     GaussianNoise(
    ...         sigma=lambda is_label, noise_sigma: (
    ...             0 if is_label else noise_sigma
    ...         )
    ...         **arguments.properties
    ...     )
    ... )
    >>> image_loader.bind_arguments(arguments)

    >>> image_loader()  # Image with added noise.
    >>> image_loader(is_label=True)  # Raw image with no noise.

    """

    def get(self, image: Any, **kwargs: Dict[str, Any]) -> Any:
        """Process the input image and allow property overrides.

        This method does not modify the input image but provides a mechanism
        for overriding arguments dynamically during pipeline execution.

        Parameters
        ----------
        image : Any
            The input image to be passed through unchanged.
        **kwargs : Any
            Key-value pairs for overriding pipeline properties.

        Returns
        -------
        Any
            The unchanged input image.

        """

        return image


class Probability(StructuralFeature):
    """Resolve a feature with a certain probability

    This feature conditionally applies a given feature to an input image based 
    on a specified probability. A random number is sampled, and if it is less 
    than the `probability`, the feature is resolved; otherwise, the input 
    image remains unchanged.

    Parameters
    ----------
    feature : Feature
        The feature to resolve conditionally.
    probability : float
        The probability (between 0 and 1) of resolving the feature. A value 
        of 0 ensures the feature is never resolved, while a value of 1 ensures 
        it is always resolved.
    *args : List[Any], optional
        Positional arguments passed to the parent `StructuralFeature` class.
    **kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the parent `StructuralFeature` 
        class.

    Methods
    -------
    get(image, feature, probability, random_number, **kwargs)
        Resolves the feature if the sampled random number is less than the 
        specified probability.

    Example
    -------
    In this example, the `GaussianBlur` is applied to the input image with 
    a 50% chance.

    >>> import numpy as np
    >>> from deeptrack.features import Probability, GaussianBlur

    Define a feature and wrap it with Probability:

    >>> blur_feature = GaussianBlur(sigma=2)
    >>> probabilistic_feature = Probability(blur_feature, probability=0.5)

    Define an input image:

    >>> input_image = np.ones((10, 10))

    Apply the feature:

    >>> output_image = probabilistic_feature(input_image)

    """

    #TODO: verify example + add unit test.

    def __init__(
        self,
        feature: Feature,
        probability: PropertyLike[float],
        *args: List[Any],
        **kwargs: Dict[str, any],
    ):
        """Initialize the Probability feature.

        Parameters
        ----------
        feature : Feature
            The feature to resolve conditionally.
        probability : PropertyLike[float]
            The probability (between 0 and 1) of resolving the feature.
        *args : List[Any], optional
            Positional arguments passed to the parent `StructuralFeature` 
            class.
        **kwargs : Dict[str, Any], optional
            Additional keyword arguments passed to the parent 
            `StructuralFeature` class.

        """

        super().__init__(
            *args,
            feature=feature,
            probability=probability,
            random_number=np.random.rand,
            **kwargs
        )

    def get(
        self,
        image: np.ndarray,
        feature: Feature,
        probability: float,
        random_number: float,
        **kwargs
    ):
        """Resolve the feature if a random number is less than the probability.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        feature : Feature
            The feature to resolve conditionally.
        probability : float
            The probability (between 0 and 1) of resolving the feature.
        random_number : float
            A random number sampled to determine whether to resolve the 
            feature.
        **kwargs : Dict[str, Any]
            Additional arguments passed to the feature's `resolve` method.

        Returns
        -------
        np.ndarray
            The processed image. If the feature is resolved, this is the 
            output of the feature; otherwise, it is the unchanged input image.

        """

        if random_number < probability:
            image = feature.resolve(image, **kwargs)

        return image


class Repeat(Feature):
    """Repeats the evaluation of the input feature a certain number of times.

    The `Repeat` feature allows iterative application of another feature, 
    passing the output of each iteration as the input to the next. Each 
    iteration operates with its own set of properties, and the index of the 
    current iteration is available as `_ID` or `replicate_index`. This enables 
    dynamic behavior across iterations.

    Parameters
    ----------
    feature : Feature
        The feature to be repeated.
    N : int
        The number of times to repeat the feature evaluation.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__ : bool
        Indicates whether this feature distributes computation across inputs. 
        Always `False` for `Repeat`, as it processes sequentially.

    Example
    -------
    Start by creating a pipeline using Repeat:

    >>> from deeptrack.features import Add, Repeat
    
    >>> pipeline = Repeat(Add(value=10), N=3)
    >>> print(pipeline.resolve([1, 2, 3]))
    [31, 32, 33]

    Equivalently, this pipeline can be created using:
    
    >>> pipeline = Add(value=10) ^ 3

    """

    __distributed__: bool = False

    def __init__(
        self, 
        feature: 'Feature', 
        N: int, 
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Repeat feature.

        Parameters
        ----------
        feature : Feature
            The feature to be repeated.
        N : int
            The number of times to repeat the feature evaluation.
        **kwargs : Dict[str, Any]
            Additional arguments passed to the parent `Feature` class.

        """

        super().__init__(N=N, **kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self,
        image: Any,
        N: int,
        _ID: Tuple[int, ...] = (),
        **kwargs: Dict[str, Any],
    ):
        """Apply sequentially the feature a set number of times.

        Sequentially applies the feature `N` times, passing the output of each 
        iteration as the input to the next.

        Parameters
        ----------
        image : Any
            The input data to be transformed by the repeated feature.
        N : int
            The number of repetitions.
        _ID : Tuple[int, ...], optional
            A unique identifier for the current computation, which tracks the 
            iteration index for caching and reproducibility.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the feature.

        Returns
        -------
        Any
            The final output after `N` repetitions of the feature.
        
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
    features : List[Feature]
        A list of features to combine. Each feature will be resolved in the 
        order they appear in the list.
    **kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to the parent `StructuralFeature` 
        class.

    Methods
    -------
    get(image_list, **kwargs)
        Resolves each feature in the `features` list on the input image and 
        returns their results as a list.

    Example
    -------
    The result is a list containing the outputs of the `GaussianBlur` and 
    `Add` features applied to the input image.

    >>> import numpy as np
    >>> from deeptrack.features import Combine, GaussianBlur, Add

    Define a list of features to combine:
    >>> blur_feature = GaussianBlur(sigma=2)
    >>> add_feature = Add(value=10)

    Combine the features:
    >>> combined_feature = Combine([blur_feature, add_feature])

    Define an input image:
    >>> input_image = np.ones((10, 10))

    Apply the combined feature:
    >>> output_list = combined_feature(input_image)

    """

    #TODO: verify example + add unit test.

    __distributed__: bool = False

    def __init__(self, features: List[Feature], **kwargs: Dict[str, Any]):
        """Initialize the Combine feature.

        Parameters
        ----------
        features : List[Feature]
            A list of features to combine. Each feature is added as a 
            dependency to ensure proper execution in the computation graph.
        **kwargs : Dict[str, Any], optional
            Additional keyword arguments passed to the parent 
            `StructuralFeature` class.

        """

        self.features = [self.add_feature(f) for f in features]
        super().__init__(**kwargs)

    def get(self, image_list: Any, **kwargs: Dict[str, Any]) -> List[Any]:
        """Resolve each feature in the `features` list on the input image.

        Parameters
        ----------
        image_list : Any
            The input image or list of images to process.
        **kwargs : Dict[str, Any]
            Additional arguments passed to each feature's `resolve` method.

        Returns
        -------
        List[Any]
            A list containing the outputs of each feature applied to the input.

        """

        return [f(image_list, **kwargs) for f in self.features]


class Slice(Feature):
    """Array indexing for each Image in list.

    This feature applies slicing to the input image(s) based on the specified 
    slices. While this feature can be used directly, it is generally easier to 
    apply normal array indexing on a feature directly.

    Parameters
    ----------
    slices : Iterable of int, slice, or ellipsis
        The slicing instructions for each dimension, specified as an iterable 
        of integers, slices, or ellipses. Each element corresponds to a 
        dimension in the input image.
    **kwargs : dict
        Additional keyword arguments passed to the parent `Feature` class.

    Examples
    --------
    Note, this feature is rarely needed to be used directly. Instead, you can 
    do normal array indexing on a feature directly. 
    
    For example, using `lambda` to demonstrate different ways to interact with 
    the slices. In this case, the `lambda` keyword is redundant.

    >>> feature = dt.DummyFeature()
    >>> sliced_feature = feature[
    ...     lambda: 0 : lambda: 1,  # Slices the first dimension.
    ...     1:2,  # Slices the second dimension.
    ...     lambda: slice(None, None, -2)  # Steps through the third dimension.
    ... ]
    >>> sliced_feature.resolve(np.arange(27).reshape((3, 3, 3)))

    Using `Slice` directly can be required in some cases, however. For example 
    if dependencies between properties are required. In this case, one can 
    replicate the previous example as follows::

    >>> feature = dt.DummyFeature()
    >>> sliced_feature = feature + dt.Slice(
    ...     slices=lambda dim1, dim2: (dim1, dim2),
    ...     dim1=slice(lambda: 0, lambda: 1, 1),
    ...     dim2=slice(1, 2, None),
    ...     dim3=lambda: slice(None, None, -2)
    ... )
    >>> sliced_feature.resolve(np.arange(27).reshape((3, 3, 3)))

    In both examples, slices can depend on other properties or be defined 
    dynamically.

    """

    #TODO: verify examples + add unit tests.

    def __init__(
        self,
        slices: PropertyLike[
            Iterable[
                PropertyLike[int] or PropertyLike[slice] or PropertyLike[...]
            ]
        ],
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Slice feature.

        Parameters
        ----------
        slices : Iterable of int, slice, or ellipsis
            The slicing instructions for each dimension.
        **kwargs : dict
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(slices=slices, **kwargs)

    def get(
        self,
        image: np.ndarray,
        slices: Union[Tuple[Any, ...], Any],
        **kwargs: Dict[str, Any],
    ):
        """Apply the specified slices to the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to be sliced.
        slices : Union[Tuple[Any, ...], Any]
            The slicing instructions for the input image.
        **kwargs : dict
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
    feature : Feature
        The child feature
    **kwargs : Dict[str, Any]
        Properties to send to child

    Example
    -------
    Dynamically modify the behavior of a feature:

    >>> import deeptrack as dt
    >>> gaussian_noise = dt.GaussianNoise()
    >>> bound_feature = dt.Bind(gaussian_noise, sigma=5)
    >>> output_image = bound_feature.resolve(input_image)

    In this example, the `sigma` parameter is dynamically set to 5 when 
    resolving the `gaussian_noise` feature.

    """

    #TODO: Check example and unit test.

    __distributed__: bool = False

    def __init__(self, feature: Feature, **kwargs: Dict[str, Any]):
        """Initialize the Bind feature.

        Parameters
        ----------
        feature : Feature
            The child feature to bind.
        **kwargs : Dict[str, Any]
            Properties or arguments to pass to the child feature.

        """

        super().__init__(**kwargs)
        self.feature = self.add_feature(feature)

    def get(self, image: Any, **kwargs: Dict[str, Any]) -> Any:
        """Resolve the child feature with the dynamically provided arguments.

        Parameters
        ----------
        image : Any
            The input data or image to process.
        **kwargs : Dict[str, Any]
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
    feature : Feature
        The child feature to bind with specific arguments.
    **kwargs : Dict[str, Any]
        Properties to send to the child feature during updates.

    Warnings
    --------
    This feature is deprecated and may be removed in a future release. 
    It is recommended to use `Bind` instead for equivalent functionality.

    Notes
    -----
    The current implementation is not guaranteed to be exactly equivalent to 
    prior implementations.

    Example
    -------
    >>> import deeptrack as dt
    >>> gaussian_noise = dt.GaussianNoise()
    >>> bound_update_feature = dt.BindUpdate(gaussian_noise, sigma=5)
    >>> output_image = bound_update_feature.resolve(input_image)

    In this example, the `sigma` parameter is dynamically set to 5 when 
    resolving the `gaussian_noise` feature.

    """

    #TODO: Check example and unit test.

    __distributed__: bool = False

    def __init__(self, feature: Feature, **kwargs: Dict[str, Any]):
        """Initialize the BindUpdate feature.

        Parameters
        ----------
        feature : Feature
            The child feature to bind with specific arguments.
        **kwargs : Dict[str, Any]
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

    def get(self, image: Any, **kwargs: Dict[str, Any]) -> Any:
        """Resolve the child feature with the provided arguments.

        Parameters
        ----------
        image : Any
            The input data or image to process.
        **kwargs : Dict[str, Any]
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
    """Conditionally override the properties of child features.

    This feature allows selectively modifying the properties of a child feature 
    based on a specified condition. If the condition evaluates to `True`, 
    the specified properties are applied to the child feature. Otherwise, the 
    child feature is resolved without modification.

    **Note**: It is adviceable to use dt.Arguments instead. Note that this 
    overwrites the properties, and as such may affect future calls.

    Parameters
    ----------
    feature : Feature
        The child feature whose properties will be conditionally overridden.
    condition : bool or str
        A boolean value or the name of a boolean property in the feature's 
        property dictionary. If the condition evaluates to `True`, the 
        specified properties are applied.
    **kwargs : Dict[str, Any]
        The properties to be applied to the child feature if `condition` is 
        `True`.

    Example
    -------
    >>> import deeptrack as dt
    >>> gaussian_noise = dt.GaussianNoise()
    >>> conditional_feature = dt.ConditionalSetProperty(
    ...     gaussian_noise, condition="is_noisy", sigma=5
    ... )
    >>> image = conditional_feature.resolve(is_noisy=True)  # Applies sigma=5.
    >>> image = conditional_feature.resolve(is_noisy=False)  # Doesn't apply it.

    """

    #TODO: Verify example and unit test.

    __distributed__: bool = False

    def __init__(
        self,
        feature: Feature,
        condition=PropertyLike[str or bool],
        **kwargs: Dict[str, Any],
    ):
        """Initialize the ConditionalSetProperty feature.

        Parameters
        ----------
        feature : Feature
            The child feature to conditionally modify.
        condition : PropertyLike[str or bool]
            A boolean value or the name of a boolean property in the feature's 
            property dictionary. If the condition evaluates to `True`, the 
            specified properties are applied.
        **kwargs : Dict[str, Any]
            Properties to apply to the child feature if the condition is 
            `True`.

        """

        super().__init__(condition=condition, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self,
        image: Any,
        condition: Union[str, bool],
        **kwargs: Dict[str, Any],
    ):
        """Resolve the child, conditionally applying specified properties.

        Parameters
        ----------
        image : Any
            The input data or image to process.
        condition : Union[str, bool]
            A boolean value or the name of a boolean property in the feature's 
            property dictionary. If the condition evaluates to `True`, the 
            specified properties are applied.
        **kwargs : Dict[str, Any]
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
    
    The `condition` parameter specifies the name of the property to listen to. 
    For example, if the `condition` is `"is_label"`, the selected feature can 
    be toggled by calling:

    >>> feature.resolve(is_label=True)  # Resolves on_true feature.
    >>> feature.resolve(is_label=False)  # Resolves on_false feature.
    >>> feature.update(is_label=True)  # Updates both features.

    Both `on_true` and `on_false` features are updated in either case, even if 
    only one of them is resolved.
    
    Parameters
    ----------
    on_false : Feature, optional
        The feature to resolve if the conditional property evaluates to `False`. 
        If not provided, the input image remains unchanged in this case.
    on_true : Feature, optional
        The feature to resolve if the conditional property evaluates to `True`. 
        If not provided, the input image remains unchanged in this case.
    condition : str or bool, optional
        The name of the conditional property, or a boolean value. Defaults to 
        `"is_label"`.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `StructuralFeature`.

    Example
    -------
    >>> import deeptrack as dt
    >>> true_feature = dt.GaussianNoise(sigma=5)
    >>> false_feature = dt.GaussianNoise(sigma=0)
    >>> conditional_feature = ConditionalSetFeature(
    ...     on_true=true_feature, 
    ...     on_false=false_feature, 
    ...     condition="is_label"
    ... )
    >>> # Resolve based on the condition.
    >>> image_with_noise = conditional_feature.resolve(is_label=False)
    >>> image_without_noise = conditional_feature.resolve(is_label=True)

    """

    #TODO: Verify example and unit test.

    __distributed__: bool = False

    def __init__(
        self,
        on_false: Optional[Feature] = None,
        on_true: Optional[Feature] = None,
        condition: PropertyLike[Union[str, bool]] = "is_label",
        **kwargs: Dict[str, Any],
    ):
        """Initialize the ConditionalSetFeature.

        Parameters
        ----------
        on_false : Feature, optional
            The feature to resolve if the condition evaluates to `False`.
        on_true : Feature, optional
            The feature to resolve if the condition evaluates to `True`.
        condition : str or bool, optional
            The name of the property to listen to, or a boolean value. Defaults 
            to `"is_label"`.
        **kwargs : Dict[str, Any]
            Additional keyword arguments for the parent `StructuralFeature`.

        """

        super().__init__(condition=condition, **kwargs)
        
        # Add the child features to the dependency graph if provided.
        if on_true:
            self.add_feature(on_true)
        if on_false:
            self.add_feature(on_false)

        self.on_true = on_true
        self.on_false = on_false

    def get(
        self,
        image: Any,
        *,
        condition: Union[str, bool],
        **kwargs: Dict[str, Any],
    ):
        """Resolve the appropriate feature based on the condition.

        Parameters
        ----------
        image : Any
            The input image to process.
        condition : str or bool
            The name of the conditional property or a boolean value. If a 
            string is provided, it is looked up in `kwargs` to get the actual 
            boolean value.
        **kwargs : Dict[str, Any]
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
    """Apply a custom function on each image in the input.

    This feature allows applying a user-defined function to individual imagen
    in the input pipeline. The property `function` needs to be wrapped in an
    outer layer function. The outer layer function can depend on other
    properties, while the inner layer function accepts only an image as input.

    Parameters
    ----------
    function : Callable[..., Callable[[Image], Image]]
        Function that takes the current image as first input. A callable that
        produces a function. The outer function can depend on other properties
        of the pipeline, while the inner function processes a single image.
    **kwargs : Dict[str, Any]
        Additional parameters passed to the parent `Feature` class.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import Lambda, Image

    Define a lambda function that scales an image:
    
    >>> def scale_function_factory(scale=2):
    ...     def scale_function(image):
    ...         return image * scale
    ...     return scale_function

    Create a Lambda feature:
    
    >>> lambda_feature = Lambda(function=scale_function_factory(scale=3))

    Apply the feature to an image:
    
    >>> input_image = Image(np.ones((5, 5)))
    >>> output_image = lambda_feature(input_image)
    >>> print(output_image)
    [[3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]]

    """

    #TODO: Check example + add unit test.

    def __init__(
        self,
        function: Callable[..., Callable[[Image], Image]],
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Lambda feature.

        Parameters
        ----------
        function : Callable[..., Callable[[Image], Image]]
            A callable that produces a function for processing an image.
        **kwargs : Dict[str, Any]
            Additional parameters passed to the parent `Feature` class.

        """

        super().__init__(function=function, **kwargs)

    def get(
        self,
        image: Image,
        function: Callable[[Image], Image],
        **kwargs: Dict[str, Any],
    ):
        """Apply the custom function to the image.

        Parameters
        ----------
        image : Image
            The input image to be processed by the function.
        function : Callable[[Image], Image]
            The function to apply to the image.
        **kwargs : Dict[str, Any]
            Additional arguments (unused here).

        Returns
        -------
        Image
            The result of applying the function to the image.

        """

        return function(image)


class Merge(Feature):
    """Apply a custom function to a list of images.

    This feature allows the application of a user-defined function to a list of 
    images. The `function` parameter must be a callable wrapped in an outer 
    layer that can depend on other properties. The inner layer of the callable
    should process a list of images.

    Note that the property `function` needs to be wrapped in an outer layer 
    function. The outer layer function can depend on other properties, while 
    the inner layer function accepts an image as input.

    Parameters
    ----------
    function : Callable[..., Callable[[List[Image]], Image or List[Image]]]
        A callable that produces a function. The outer function can depend on 
        other properties of the pipeline, while the inner function takes a list 
        of images and returns a single image or a list of images.
    **kwargs : Dict[str, Any]
        Additional parameters passed to the parent `Feature` class.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import Merge, Image

    Define a merge function that combines images by averaging:
    
    >>> def merge_function_factory():
    ...     def merge_function(images):
    ...         return np.mean(np.stack(images), axis=0)
    ...     return merge_function

    Create a Merge feature:
    
    >>> merge_feature = Merge(function=merge_function_factory)

    Apply the feature to a list of images:
    
    >>> image_1 = Image(np.ones((5, 5)) * 2)
    >>> image_2 = Image(np.ones((5, 5)) * 4)
    >>> output_image = merge_feature([image_1, image_2])
    >>> print(output_image)
    [[3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3.]]

    """

    #TODO: verify example + add unit test.

    __distributed__: bool = False

    def __init__(
        self,
        function: Callable[..., 
                           Callable[[List[Image]], Union[Image, List[Image]]]],
        **kwargs: Dict[str, Any]
    ):
        """Initialize the Merge feature.

        Parameters
        ----------
        function : Callable[..., Callable[[List[Image]], Image or List[Image]]]
            A callable that returns a function for processing a list of images.
        **kwargs : Dict[str, Any]
            Additional parameters passed to the parent `Feature` class.

        """

        super().__init__(function=function, **kwargs)

    def get(
        self,
        list_of_images: List[Image],
        function: Callable[[List[Image]], Union[Image, List[Image]]],
        **kwargs: Dict[str, Any],
    ) -> Union[Image, List[Image]]:
        """Apply the custom function to the list of images.

        Parameters
        ----------
        list_of_images : List[Image]
            A list of images to be processed by the function.
        function : Callable[[List[Image]], Image or List[Image]]
            The function to apply to the list of images.
        **kwargs : Dict[str, Any]
            Additional arguments (unused here).

        Returns
        -------
        Image or List[Image]
            The result of applying the function to the list of images.

        """

        return function(list_of_images)


class OneOf(Feature):
    """Resolves one feature from a collection on the input.

    Valid collections are any object that can be iterated (such as lists, 
    tuples, and sets). Internally, the collection is converted to a tuple.

    The default behavior is to sample the collection uniformly random. This can
    be controlled by the `key` argument, where the feature resolved is chosen
    as `tuple(collection)[key]`.

    Parameters
    ----------
    collection : Iterable[Feature]
        A collection of features to choose from.
    key : Optional[int], optional
        The index of the feature to resolve from the collection. If not 
        provided, a feature is selected randomly.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    collection : Tuple[Feature, ...]
        The collection of features to choose from, stored as a tuple.

    Methods
    -------
    get(image, key, _ID=(), **kwargs)
        Resolves the selected feature on the input image.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import OneOf, Add, Multiply

    Create a collection of features:
    
    >>> feature_1 = Add(value=10)
    >>> feature_2 = Multiply(value=2)
    >>> one_of_feature = OneOf([feature_1, feature_2])

    Apply the feature randomly to an input image:
    
    >>> input_image = np.array([1, 2, 3])
    >>> output_image = one_of_feature(input_image)
    >>> print(output_image)  # Output depends on randomly selected feature.

    Specify a key to control the selected feature:
    
    >>> controlled_feature = OneOf([feature_1, feature_2], key=0)
    >>> output_image = controlled_feature(input_image)
    >>> print(output_image)  # Adds 10 to each element.

    """

    __distributed__: bool = False

    def __init__(
        self,
        collection: Iterable[Feature],
        key: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the OneOf feature.

        Parameters
        ----------
        collection : Iterable[Feature]
            A collection of features to choose from.
        key : Optional[int], optional
            The index of the feature to resolve from the collection. If not 
            provided, a feature is selected randomly.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        self.collection = tuple(collection)
        super().__init__(key=key, **kwargs)

        # Add all features in the collection as dependencies.
        for feature in self.collection:
            self.add_feature(feature)

    def _process_properties(
        self, 
        propertydict: dict,
    ) -> dict:
        """Process the properties to select the feature index.

        Parameters
        ----------
        propertydict : dict
            The property dictionary for the feature.

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
        self,
        image: Any,
        key: int,
        _ID: Tuple[int, ...] = (),
        **kwargs: Dict[str, Any],
    ):
        """Resolve the selected feature on the input image.

        Parameters
        ----------
        image : Any
            The input image to process.
        key : int
            The index of the feature to apply from the collection.
        _ID : Tuple[int, ...], optional
            A unique identifier for caching and parallel processing.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The output of the selected feature applied to the input image.

        """

        return self.collection[key](image, _ID=_ID)


class OneOfDict(Feature):
    """Resolve one feature from a dictionary.

    This feature selects one feature from a dictionary of features and applies 
    it to the input. By default, the selection is made randomly from the 
    dictionary's values, but it can be controlled by specifying a `key`.

    Its default behaviour is to sample the values diction uniformly random. 
    This can be controlled by the `key` argument, where the feature resolved is 
    chosen as `collection[key]`.

    Parameters
    ----------
    collection : Dict[Any, Feature]
        A dictionary where keys are identifiers and values are features to 
        choose from.
    key : Optional[Any], optional
        The key of the feature to resolve from the dictionary. If not provided, 
        a key is selected randomly.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    collection : Dict[Any, Feature]
        The dictionary of features to choose from.

    Methods
    -------
    get(image, key, _ID=(), **kwargs)
        Resolves the selected feature from the dictionary and applies it to the 
        input image.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import OneOfDict, Add, Multiply

    Create a dictionary of features:
    
    >>> features_dict = {
    ...     "add": Add(value=10),
    ...     "multiply": Multiply(value=2),
    ... }
    >>> one_of_dict_feature = OneOfDict(features_dict)

    Apply the feature randomly to an input image:
    
    >>> input_image = np.array([1, 2, 3])
    >>> output_image = one_of_dict_feature(input_image)
    >>> print(output_image)  # Output depends on randomly selected feature.

    Specify a key to control the selected feature:
    
    >>> controlled_feature = OneOfDict(features_dict, key="add")
    >>> output_image = controlled_feature(input_image)
    >>> print(output_image)  # Adds 10 to each element.

    """

    __distributed__: bool = False

    def __init__(
        self,
        collection: Dict[Any, Feature],
        key: Optional[Any] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the OneOfDict feature.

        Parameters
        ----------
        collection : Dict[Any, Feature]
            A dictionary where keys are identifiers and values are features to 
            choose from.
        key : Optional[Any], optional
            The key of the feature to resolve from the dictionary. If not 
            provided, a key is selected randomly.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        self.collection = collection

        super().__init__(key=key, **kwargs)

        # Add all features in the dictionary as dependencies.
        for feature in self.collection.values():
            self.add_feature(feature)

    def _process_properties(self, propertydict: dict) -> dict:
        """Process the properties to select the feature key.

        Parameters
        ----------
        propertydict : dict
            The property dictionary for the feature.

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
        self,
        image: Any,
        key: Any,
        _ID: Tuple[int, ...] = (),
        **kwargs: Dict[str, Any],
    )-> Any:
        """Resolve selected feature and applies it to the input image.

        This method resolves the selected feature from the dictionary and 
        applies it to the input image.

        Parameters
        ----------
        image : Any
            The input image to process.
        key : Any
            The key of the feature to apply from the dictionary.
        _ID : Tuple[int, ...], optional
            A unique identifier for caching and parallel processing.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The output of the selected feature applied to the input image.

        """

        return self.collection[key](image, _ID=_ID)


class Label(Feature):
    """Output the properties of this feature.

    This feature can be used to extract properties in a feature set and combine 
    them into a numpy array. Specifically, it extracts specified properties 
    from a feature set and combines them into a NumPy array. Optionally, the 
    output array can be reshaped to a specified shape.

    Parameters
    ----------
    output_shape : Optional[PropertyLike[Tuple[int, ...]]], optional
        Specifies the desired shape of the output array. If `None`, the output 
        array will be one-dimensional.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    get(image, output_shape=None, **kwargs)
        Extracts and combines properties into a NumPy array, reshaping it 
        if `output_shape` is specified.

    """

    #TODO: add example.

    __distributed__: bool = False

    def __init__(
        self,
        output_shape: PropertyLike[int] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Label feature.

        Parameters
        ----------
        output_shape : PropertyLike[Tuple[int, ...]], optional
            Specifies the desired shape of the output array. If `None`, the 
            output array will be one-dimensional.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(output_shape=output_shape, **kwargs)

    def get(
        self,
        image: Any,
        output_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Extract and combine properties into a NumPy array.

        Parameters
        ----------
        image : Any
            The input image (not used in this feature).
        output_shape : Tuple[int, ...], optional
            Specifies the desired shape of the output array. If `None`, the 
            output array will be one-dimensional.
        **kwargs : Dict[str, Any]
            Additional properties passed to the feature.

        Returns
        -------
        np.ndarray
            The extracted properties combined into a NumPy array. If 
            `output_shape` is specified, the array is reshaped accordingly.

        """

        result = []
        for key in self.properties.keys():
            if key in kwargs:
                result.append(kwargs[key])

        if output_shape:
            result = np.reshape(np.array(result), output_shape)

        return np.array(result)


class LoadImage(Feature):
    """Load an image from disk.

    This feature attempts to load an image file using a series of file readers 
    (`imageio`, `numpy`, `Pillow`, and `OpenCV`) until a suitable reader is 
    found. Additional options allow for converting the image to grayscale, 
    reshaping it to a specified number of dimensions, or treating the first 
    dimension as a list of images.

    Parameters
    ----------
    path : PropertyLike[Union[str, List[str]]]
        The path(s) to the image(s) to load. Can be a single string or a list 
        of strings.
    load_options : PropertyLike[Dict[str, Any]], optional
        Options passed to the file reader. Defaults to `None`.
    as_list : PropertyLike[bool], optional
        If `True`, the first dimension of the image will be treated as a list. 
        Defaults to `False`.
    ndim : PropertyLike[int], optional
        Ensures the image has at least this many dimensions. Defaults to `3`.
    to_grayscale : PropertyLike[bool], optional
        If `True`, converts the image to grayscale. Defaults to `False`.
    get_one_random : PropertyLike[bool], optional
        If `True`, extracts a single random image from a stack of images. Only 
        used when `as_list` is `True`. Defaults to `False`.

    Raises
    ------
    IOError
        If no file reader could parse the file or the file does not exist.

    """

    #TODO: add example.

    __distributed__: bool = False

    def __init__(
        self,
        path: PropertyLike[Union[str, List[str]]],
        load_options: PropertyLike[dict] = None,
        as_list: PropertyLike[bool] = False,
        ndim: PropertyLike[int] = 3,
        to_grayscale: PropertyLike[bool] = False,
        get_one_random: PropertyLike[bool] = False,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the LoadImage feature.

        Parameters
        ----------
        path : PropertyLike[str or List[str]]
            The path(s) to the image(s) to load.
        load_options : Optional[PropertyLike[Dict[str, Any]]], optional
            Options passed to the file reader.
        as_list : PropertyLike[bool], optional
            Whether to treat the first dimension of the image as a list.
        ndim : PropertyLike[int], optional
            Ensures the image has at least this many dimensions.
        to_grayscale : PropertyLike[bool], optional
            Whether to convert the image to grayscale.
        get_one_random : PropertyLike[bool], optional
            Whether to extract a single random image from a stack.
        **kwargs : Any
            Additional keyword arguments passed to the parent `Feature` class.

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
        self,
        *ign: Any,
        path: Union[str, List[str]],
        load_options: Optional[Dict[str, Any]],
        ndim: int,
        to_grayscale: bool,
        as_list: bool,
        get_one_random: bool,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Load the image(s) from disk and process them.

        Parameters
        ----------
        path : Union[str, List[str]]
            The path(s) to the image(s) to load.
        load_options : Optional[Dict[str, Any]]
            Options passed to the file reader.
        ndim : int
            Ensures the image has at least this many dimensions.
        to_grayscale : bool
            Whether to convert the image to grayscale.
        as_list : bool
            Whether to treat the first dimension as a list.
        get_one_random : bool
            Whether to extract a single random image from a stack.
        **kwargs : Dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The loaded and processed image(s).

        Raises
        ------
        IOError
            If no file reader could parse the file or the file does not exist.

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
        except (IOError, ImportError, AttributeError):
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

    Calls `transformation_function` for each input image, and merges the 
    outputs to a single image with `number_of_masks` layers. Each input image 
    needs to have a defined property `position` to place it within the image. 
    If used with scatterers, note that the scatterers need to be passed the 
    property `voxel_size` to correctly size the objects.

    Parameters
    ----------
    transformation_function : Callable[[Image], Image]
        Function that takes an image as input, and outputs another image with 
        `number_of_masks` layers.
    number_of_masks : PropertyLike[int], optional
        The number of masks to create.
    output_region : PropertyLike[Tuple[int, int, int, int]], optional
        Size and relative position of the mask. Should generally be the same as
        `optics.output_region`.
    merge_method : PropertyLike[str or Callable or List[str or Callable]]
        How to merge the individual masks to a single image. If a list, the 
        merge_metod is per mask. Can be:
        - "add": Adds the masks together.
        - "overwrite": later masks overwrite earlier masks.
        - "or": 1 if either any mask is non-zero at that pixel.
        - function: a function that accepts two images. The first is the
            current value of the output image where a new mask will be places, 
            and the second is the mask to merge with the output image.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    """

    def __init__(
        self,
        transformation_function: Callable[[Image], Image],
        number_of_masks: PropertyLike[int] = 1,
        output_region: PropertyLike[Tuple[int, int, int, int]] = None,
        merge_method: PropertyLike[Union[str, Callable, List[Union[str, Callable]]]] = "add",
        **kwargs: Any,
    ):
        """Initialize the SampleToMasks feature.

        Parameters
        ----------
        transformation_function : Callable[[Image], Image]
            The function used to transform input images into masks.
        number_of_masks : PropertyLike[int], optional
            Number of masks to generate. Defaults to 1.
        output_region : PropertyLike[Tuple[int, int, int, int]], optional
            Defines the output mask region. Defaults to None.
        merge_method : PropertyLike[str or Callable or List[str or Callable]], optional
            Specifies the method to merge individual masks into a single image. 
            Defaults to "add".
        **kwargs : Dict[str, Any]
            Additional parameters passed to the parent `Feature` class.

        """

        super().__init__(
            transformation_function=transformation_function,
            number_of_masks=number_of_masks,
            output_region=output_region,
            merge_method=merge_method,
            **kwargs,
        )

    def get(
        self,
        image: Image,
        transformation_function: Callable[[Image], Image],
        **kwargs: Dict[str, Any],
    ) -> Image:
        """Apply the transformation function to the input image.

        Parameters
        ----------
        image : Image
            The input image to transform.
        transformation_function : Callable[[Image], Image]
            The function used to transform the input image.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        Image
            The transformed image.

        """

        return transformation_function(image)

    def _process_and_get(
        self,
        images: Union[List[Image], Image],
        **kwargs: Dict[str, Any],
    ) -> Union[Image, np.ndarray]:
        """Process a list of images and generate a multi-layer mask.

        Parameters
        ----------
        images : Image or List[Image]
            A list of input images or a single image.
        **kwargs : Any
            Additional parameters including `output_region`, `number_of_masks`, 
            and `merge_method`.

        Returns
        -------
        Image or np.ndarray
            The generated mask image with the specified number of layers.
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

        # Merge masks into the output.
        for label in list_of_labels:
            positions = _get_position(label)
            for position in positions:
                p0 = np.round(position - output_region[0:2])

                if np.any(p0 > output.shape[0:2]) or np.any(p0 + label.shape[0:2] < 0):
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
                        ] = labelarg[labelarg[..., label_index] != 0, label_index]
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


def _get_position(
    image: Any,
    mode: str = "corner",
    return_z: bool = False,
) -> List[np.ndarray]:
    """Extracts the position of the upper left corner of a scatterer.
    
    This function calculates the position of scatterers in an image, 
    adjusting for the specified mode. It can also include the z-coordinate 
    if `return_z` is `True`.

    The `image` must have a `get_property` method that retrieves scatterer
    positions as a list of 2D or 3D coordinates.

    If `mode` is "corner", positions are adjusted by subtracting half the
    image dimensions (shift).

    Parameters
    ----------
    image : Any
        The input image containing scatterer information. This image is 
        expected to have a `get_property` method to retrieve properties like 
        "position".
    mode : str, optional
        The calculation mode. Defaults to "corner".
        - "corner": Adjusts the position based on the upper-left corner.
        - Any other value: Does not adjust for the corner; assumes positions 
          are given directly.
    return_z : bool, optional
        If `True`, includes the z-coordinate in the output. Defaults to 
        `False`.

    Returns
    -------
    List[np.ndarray]
        A list of position arrays, each representing the (x, y) or (x, y, z) 
        coordinates of a scatterer. Adjustments are made based on the mode.

    """

    #TODO: should this function be moved?
    #TODO: add example + unit test.

    # Determine the shift based on the mode.
    if mode == "corner":
        # Shift corresponds to the half dimensions of the image.
        shift = (np.array(image.shape) - 1) / 2
    else:
        # No shift if mode is not "corner".
        shift = np.zeros((3 if return_z else 2))

    # Retrieve positions from the image properties.
    positions = image.get_property("position", False, [])
    positions_out = []
    
    # Process each position in the list.
    for position in positions:
        if len(position) == 3:  # 3D position.
            if return_z:
                # Adjust for shift and include z-coordinate.
                return positions_out.append(position - shift)
            else:
                # Adjust for shift and exclude z-coordinate.
                return positions_out.append(position[0:2] - shift[0:2])

        elif len(position) == 2:  # 2D position.
            if return_z:
                # Construct a 3D position by adding a z-coordinate 
                # (default to 0).
                outp = (
                    np.array(
                        [
                            position[0],
                            position[1],
                            image.get_property("z", default=0),
                        ]
                    )
                    - shift
                )
                positions_out.append(outp)
            else:
                # Adjust for shift without adding a z-coordinate.
                positions_out.append(position - shift[0:2])

    return positions_out


class AsType(Feature):
    """Convert the data type of images.

    This feature changes the data type (`dtype`) of input images to a specified 
    type. The accepted types are the same as those used by NumPy arrays, such 
    as `float64`, `int32`, `uint16`, `int16`, `uint8`, and `int8`.

    Parameters
    ----------
    dtype : PropertyLike[Any], optional
        The desired data type for the image. Defaults to `"float64"`.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Example
    -------
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
        self,
        dtype: PropertyLike[Any] = "float64",
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize the AsType feature.

        Parameters
        ----------
        dtype : PropertyLike[Any], optional
            The desired data type for the image. Defaults to `"float64"`.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(dtype=dtype, **kwargs)

    def get(
        self,
        image: np.ndarray,
        dtype: str,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Convert the data type of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        dtype : str
            The desired data type for the image.
        **kwargs : Any
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
    axis : int, optional
        The axis to move to the first position. Defaults to `-1` (last axis).
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Example
    -------
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
        self,
        axis: int = -1,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the ChannelFirst2d feature.

        Parameters
        ----------
        axis : int, optional
            The axis to move to the first position. 
            Defaults to `-1` (last axis).
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self,
        image: np.ndarray,
        axis: int,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Rearrange the axes of an image to channel-first format.

        Rearrange the axes of a 3D image to channel-first format or add a 
        channel dimension to a 2D image.

        Parameters
        ----------
        image : np.ndarray
            The input image to process. Can be 2D or 3D.
        axis : int
            The axis to move to the first position (for 3D images).
        **kwargs : Any
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
    """Perform the simulation at a higher resolution.

    This feature scales up the resolution of the input pipeline by a specified 
    factor, performs computations at the higher resolution, and then 
    downsamples the result back to the original size. This is useful for 
    simulating effects at a finer resolution while preserving compatibility 
    with lower-resolution pipelines.
    
    It redefines the sizes of internal units to scale up the simulation. 
    The resulting image is then downscaled back to the original size.

    Parameters
    ----------
    feature : Feature
        The pipeline or feature to resolve at a higher resolution.
    factor : int or Tuple[int, int, int], optional
        The factor by which to upscale the simulation. If a single integer is 
        provided, it is applied uniformly across all axes. If a tuple of three 
        integers is provided, each axis is scaled individually. Defaults to 1.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__ : bool
        Indicates whether this feature distributes computation across inputs.
        Always `False` for `Upscale`.

    Methods
    -------
    get(image, factor, **kwargs)
        Scales up the pipeline, performs computations, and scales down result.

    Example
    -------
    >>> import deeptrack as dt
    >>> optics = dt.Fluorescence()
    >>> particle = dt.Sphere()
    >>> pipeline = optics(particle)
    >>> upscaled_pipeline = dt.Upscale(pipeline, factor=4)

    """

    __distributed__: bool = False

    def __init__(
        self,
        feature: Feature,
        factor: Union[int, Tuple[int, int, int]] = 1,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Upscale feature.

        Parameters
        ----------
        feature : Feature
            The pipeline or feature to resolve at a higher resolution.
        factor : Union[int, Tuple[int, int, int]], optional
            The factor by which to upscale the simulation. If a single integer 
            is provided, it is applied uniformly across all axes. If a tuple of 
            three integers is provided, each axis is scaled individually. 
            Defaults to `1`.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(factor=factor, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self,
        image: np.ndarray,
        factor: Union[int, Tuple[int, int, int]],
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Scale up resolution of feature pipeline and scale down result.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        factor : int or Tuple[int, int, int]
            The factor by which to upscale the simulation. If a single integer 
            is provided, it is applied uniformly across all axes. If a tuple of 
            three integers is provided, each axis is scaled individually.
        **kwargs : Dict[str, Any]
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
        image = skimage.measure.block_reduce(
            image, (factor[0], factor[1]) + (1,) * (image.ndim - 2), np.mean
        )

        return image


class NonOverlapping(Feature):
    """Ensure volumes are placed non-overlapping in a 3D space.

    This feature ensures that a list of 3D volumes are positioned such that 
    their non-zero voxels do not overlap. If volumes overlap, their positions 
    are resampled until they are non-overlapping. If the maximum number of 
    attempts is exceeded, the feature regenerates the list of volumes.
    
    This feature is incompatible with non-volumetric scatterers such as 
    `MieScatterers`.
    
    Parameters
    ----------
    feature : Feature
        The feature that generates the list of volumes to place non-overlapping.
    min_distance : float, optional
        The minimum distance between volumes in pixels. Defaults to `1`.
    max_attempts : int, optional
        The maximum number of attempts to place volumes without overlap. If 
        exceeded, a new list of volumes is generated. Defaults to `100`.

    """

    #TODO: Add example.

    __distributed__: bool = False

    def __init__(
        self,
        feature: Feature,
        min_distance: float = 1,
        max_attempts: int = 100,
        **kwargs: Dict[str, Any],
    ):
        """Places a list of volumes non-overlapping.

        Ensures that the volumes are placed non-overlapping by resampling the
        position of the volumes until they are non-overlapping. If the maximum 
        number of attempts is exceeded, a new list of volumes is generated by 
        updating feature.

        This feature does not work with non-volumetric scatterers, such as 
        `MieScatterers`.

        Parameters
        ----------
        feature : Feature
            The feature that creates the list of volumes to be placed 
            non-overlapping.
        min_distance : float, optional
            The minimum distance between volumes in pixels, by default 1.
        max_attempts : int, optional
            The maximum number of attempts to place the volumes
            non-overlapping. If this number is exceeded, a new list of volumes 
            is generated, by default 100.

        """

        super().__init__(min_distance=min_distance, max_attempts=max_attempts, **kwargs)
        self.feature = self.add_feature(feature, **kwargs)

    def get(
        self,
        _: Any,
        min_distance: float,
        max_attempts: int,
        **kwargs: Dict[str, Any],
    ) -> List[np.ndarray]:
        """
        Parameters
        ----------
        _ : Any
            Placeholder for unused input image.
        min_distance : float
            The minimum distance between volumes in pixels.
        max_attempts : int
            The maximum number of attempts to place the volumes 
            non-overlapping. If this number is exceeded, a new list of volumes 
            is generated.

        Returns
        -------
        List[np.ndarray]
            A list of non-overlapping 3D volumes.

        """

        while True:
            list_of_volumes = self.feature()

            if not isinstance(list_of_volumes, list):
                list_of_volumes = [list_of_volumes]

            for _ in range(max_attempts):

                list_of_volumes = [
                    self._resample_volume_position(volume) 
                    for volume in list_of_volumes
                ]

                if self._check_non_overlapping(list_of_volumes):
                    return list_of_volumes

            # Generate a new list of volumes if max_attempts is exceeded.
            self.feature.update()

    def _check_non_overlapping(
        self, 
        list_of_volumes: List[np.ndarray],
    ) -> bool:
        """Check if all volumes in the list are non-overlapping.

        Checks that the non-zero voxels of the volumes in list_of_volumes are 
        at least min_distance apart. Each volume is a 3 dimnesional array. The 
        first two dimensions are the x and y dimensions, and the third 
        dimension is the z dimension. The volumes are expected to have a 
        position attribute.

        Parameters
        ----------
        list_of_volumes : list of 3d arrays
            The volumes to be checked for non-overlapping.

        Returns
        -------
        bool
            `True` if all volumes are non-overlapping, otherwise `False`.

        """

        from skimage.morphology import isotropic_erosion

        from .augmentations import CropTight
        from .optics import _get_position

        min_distance = self.min_distance()
        if min_distance < 0:
            crop = CropTight()
            # print([np.sum(volume != 0) for volume in list_of_volumes])
            list_of_volumes = [
                Image(
                    crop(isotropic_erosion(volume != 0, -min_distance/2)),
                    copy=False,
                ).merge_properties_from(volume) 
                for volume in list_of_volumes
            ]
            # print([np.sum(volume != 0) for volume in list_of_volumes])

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
        self,
        bounding_cube_1: List[int],
        bounding_cube_2: List[int], 
        min_distance: float,
    ) -> bool:
        """Checks whether two bounding cubes are non-overlapping.

        This method determines if two 3D bounding cubes, defined by their
        corner coordinates, are separated by at least a minimum distance
        (`min_distance`). The bounding cubes are represented as lists of six
        integers, where:

        - The first three integers (`x1, y1, z1`) represent the coordinates of
        the top-left corner.
        - The last three integers (`x2, y2, z2`) represent the coordinates of
        the bottom-right corner.

        Two bounding cubes are considered non-overlapping if the distance
        between their closest edges is greater than or equal to `min_distance` 
        along any of the three spatial axes.

        Parameters
        ----------
        bounding_cube_1 : List[int]
            The first bounding cube, defined as `[x1, y1, z1, x2, y2, z2]`.
        bounding_cube_2 : List[int]
            The second bounding cube, defined as `[x1, y1, z1, x2, y2, z2]`.
        min_distance : float
            The minimum distance allowed between the two bounding cubes.

        Returns
        -------
        bool
            `True` if the bounding cubes are non-overlapping (separated by at 
            least `min_distance`), otherwise `False`.

        """

        # bounding_cube_1 and bounding_cube_2 are (x1, y1, z1, x2, y2, z2).
        # Check that the bounding cubes are non-overlapping.
        return (
            bounding_cube_1[0] > bounding_cube_2[3] + min_distance
            or bounding_cube_1[1] > bounding_cube_2[4] + min_distance
            or bounding_cube_1[2] > bounding_cube_2[5] + min_distance
            or bounding_cube_1[3] < bounding_cube_2[0] - min_distance
            or bounding_cube_1[4] < bounding_cube_2[1] - min_distance
            or bounding_cube_1[5] < bounding_cube_2[2] - min_distance
        )

    def _get_overlapping_cube(
        self,
        bounding_cube_1: List[int],
        bounding_cube_2: List[int],
    ) -> List[int]:
        """Return the overlapping region of the two bounding cubes.

        This method calculates the coordinates of the overlapping region 
        between two 3D bounding cubes. The bounding cubes are represented as 
        lists of six integers, where:

        - The first three integers (`x1, y1, z1`) represent the coordinates of
        the top-left corner.
        - The last three integers (`x2, y2, z2`) represent the coordinates of
        the bottom-right corner.

        The overlapping region is defined as the maximum of the minimum 
        coordinates and the minimum of the maximum coordinates along each axis. 
        If the cubes do not overlap, the resulting coordinates will not 
        represent a valid cube (i.e., `x1 > x2`, `y1 > y2`, or `z1 > z2`).

        If the two bounding cubes do not overlap, the coordinates in the result
        will not define a valid cube (e.g., `x1 > x2`).

        The method does not validate the input; it assumes the input is
        correctly formatted.

        Parameters
        ----------
        bounding_cube_1 : List[int]
            The first bounding cube, defined as `[x1, y1, z1, x2, y2, z2]`.
        bounding_cube_2 : List[int]
            The second bounding cube, defined as `[x1, y1, z1, x2, y2, z2]`.

        Returns
        -------
        List[int]
            A list of six integers representing the overlapping bounding cube, 
            formatted as `[x1, y1, z1, x2, y2, z2]`.

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
        self,
        volume: np.ndarray,  # 3D array.
        bounding_cube: Tuple[float, float, float, float, float, float],
        overlapping_cube: Tuple[float, float, float, float, float, float],
    ) -> np.ndarray:
        """
        Returns the overlapping region of the volume and the overlapping cube.

        Parameters
        ----------
        volume : np.ndarray
            The volume (3D array) to be checked for non-overlapping.
        bounding_cube : Tuple[float, float, float, float, float, float]
            The bounding cube of the volume (list of 6 floats). The first three
            elements are the position of the top left corner of the volume, and
            the last three elements are the position of the bottom right corner
            of the volume.
        overlapping_cube : Tuple[float, float, float, float, float, float]
            The overlapping cube of the volume and the other volume (list of 6 
            floats).

        Returns
        -------
        np.ndarray
            The region of the volume that lies within the overlapping cube, as a 
            3D NumPy array.

        """

        # The position of the top left corner of the overlapping cube in the volume
        overlapping_cube_position = np.array(overlapping_cube[:3]) - np.array(
            bounding_cube[:3]
        )

        # The position of the bottom right corner of the overlapping cube in the volume
        overlapping_cube_end_position = np.array(overlapping_cube[3:]) - np.array(
            bounding_cube[:3]
        )

        # cast to int
        overlapping_cube_position = overlapping_cube_position.astype(int)
        overlapping_cube_end_position = overlapping_cube_end_position.astype(int)

        return volume[
            overlapping_cube_position[0] : overlapping_cube_end_position[0],
            overlapping_cube_position[1] : overlapping_cube_end_position[1],
            overlapping_cube_position[2] : overlapping_cube_end_position[2],
        ]

    def _check_volumes_non_overlapping(
        self,
        volume_1: np.ndarray,
        volume_2: np.ndarray,
        min_distance: float,
    ) -> bool:
        """Check if non-zero voxels of two volumes are minimum distance apart.

        This method determines whether the non-zero voxels (active regions) in
        two 3D volumes are separated by at least `min_distance`. If the volumes
        are of different sizes, the voxel positions of one volume are scaled to
        match the other's size for comparison.

        Parameters
        ----------
        volume_1 : np.ndarray
            The first 3D volume to check for non-overlapping.
        volume_2 : np.ndarray
            The second 3D volume to check for non-overlapping.
        min_distance : float
            The minimum distance required between any two non-zero voxels in
            the two volumes.

        Returns
        -------
        bool
            `True` if all non-zero voxels in `volume_1` and `volume_2` are at
            least `min_distance` apart. `False` otherwise.

        """

        # Get the positions of the non-zero voxels of each volume.
        positions_1 = np.argwhere(volume_1)
        positions_2 = np.argwhere(volume_2)

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
        import scipy.spatial.distance as distance

        return np.all(
            distance.cdist(positions_1, positions_2) > min_distance
        )

    def _resample_volume_position(
        self,
        volume: Image,
    ) -> Image:
        """Draws a new position for the volume.

        This method updates the position of a 3D volume by sampling a new 
        position using the `_position_sampler` property in the volume's 
        properties. The `position` property of the volume is updated with the 
        newly sampled value.

        Parameters
        ----------
        volume : Image
            The input volume whose position needs to be resampled. The volume
            is expected to have a `properties` attribute containing 
            dictionaries with `position` and `_position_sampler` keys.

        Returns
        -------
        Image
            The input volume with its `position` property updated to the newly 
            sampled value.

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
    feature : Feature
        The feature to evaluate and store.
    key : Any
        The key used to identify the stored output.
    replace : bool, optional
        If `True`, replaces the stored value with a new computation. Defaults 
        to `False`.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__ : bool
        Indicates whether this feature distributes computation across inputs.
        Always `False` for `Store`, as it handles caching locally.
    _store : Dict[Any, Image]
        A dictionary used to store the outputs of the evaluated feature.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import Store, Value

    >>> value_feature = Value(lambda: np.random.rand())

    Create a `Store` feature with a key:

    >>> store_feature = Store(feature=value_feature, key="example")

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
        self,
        feature: Feature,
        key: Any,
        replace: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize the Store feature.

        Parameters
        ----------
        feature : Feature
            The feature to evaluate and store.
        key : Any
            The key used to identify the stored output.
        replace : bool, optional
            If `True`, replaces the stored value with a new computation. 
            Defaults to `False`.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(feature=feature, key=key, replace=replace, **kwargs)

        self.feature = self.add_feature(feature, **kwargs)

        self._store: dict[Any, Image] = {}

    def get(
        self,
        _: Any,
        key: Any,
        replace: bool,
        **kwargs: Dict[str, Any],
    ) -> Any:
        """Evaluate and store the feature output, or return the cached result.

        Parameters
        ----------
        _ : Any
            Placeholder for unused image input.
        key : Any
            The key used to identify the stored output.
        replace : bool
            If `True`, replaces the stored value with a new computation.
        **kwargs : Any
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
    axis : int or Tuple[int, ...], optional
        The axis or axes to squeeze. Defaults to `None`, squeezing all axes.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Example
    -------
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
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Squeeze feature.

        Parameters
        ----------
        axis : int or Tuple[int, ...], optional
            The axis or axes to squeeze. Defaults to `None`, which squeezes 
            all axes.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self,
        image: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Squeeze the input image by removing singleton dimensions.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        axis : int or Tuple[int, ...], optional
            The axis or axes to squeeze. Defaults to `None`, which squeezes 
            all axes.
        **kwargs : Dict[str, Any]
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
    axis : int or Tuple[int, ...], optional
        The axis or axes where new singleton dimensions should be added. 
        Defaults to `None`, which adds a singleton dimension at the last axis.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import Unsqueeze

    Create an input array:
    
    >>> input_image = np.array([1, 2, 3])
    >>> print(input_image.shape)
    (3,)

    Apply an Unsqueeze feature:
    
    >>> unsqueeze_feature = Unsqueeze(axis=0)
    >>> output_image = unsqueeze_feature(input_image)
    >>> print(output_image.shape)
    (1, 3)

    Without specifying an axis:

    >>> unsqueeze_feature = Unsqueeze()
    >>> output_image = unsqueeze_feature(input_image)
    >>> print(output_image.shape)
    (3, 1)

    """

    def __init__(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = -1,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Unsqueeze feature.

        Parameters
        ----------
        axis : int or Tuple[int, ...], optional
            The axis or axes where new singleton dimensions should be added. 
            Defaults to -1, which adds a singleton dimension at the last axis.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(axis=axis, **kwargs)

    def get(
        self,
        image: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = -1,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Add singleton dimensions to the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        axis : int or Tuple[int, ...], optional
            The axis or axes where new singleton dimensions should be added. 
            Defaults to -1, which adds a singleton dimension at the last axis.
        **kwargs : Dict[str, Any]
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
    source : int
        The axis to move.
    destination : int
        The destination position of the axis.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import MoveAxis

    Create an input array:
    
    >>> input_image = np.random.rand(2, 3, 4)
    >>> print(input_image.shape)
    (2, 3, 4)

    Apply a MoveAxis feature:
    
    >>> move_axis_feature = MoveAxis(source=0, destination=2)
    >>> output_image = move_axis_feature(input_image)
    >>> print(output_image.shape)
    (3, 4, 2)

    """

    def __init__(
        self,
        source: int,
        destination: int,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the MoveAxis feature.

        Parameters
        ----------
        source : int
            The axis to move.
        destination : int
            The destination position of the axis.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(source=source, destination=destination, **kwargs)

    def get(
        self,
        image: np.ndarray,
        source: int,
        destination: int, 
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Move the specified axis of the input image to a new position.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        source : int
            The axis to move.
        destination : int
            The destination position of the axis.
        **kwargs : Dict[str, Any]
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
    axes : Tuple[int, ...], optional
        A tuple specifying the permutation of the axes. If `None`, the axes are 
        reversed by default.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import Transpose

    Create an input array:

    >>> input_image = np.random.rand(2, 3, 4)
    >>> print(input_image.shape)
    (2, 3, 4)

    Apply a Transpose feature:
    
    >>> transpose_feature = Transpose(axes=(1, 2, 0))
    >>> output_image = transpose_feature(input_image)
    >>> print(output_image.shape)
    (3, 4, 2)

    Without specifying axes:
    
    >>> transpose_feature = Transpose()
    >>> output_image = transpose_feature(input_image)
    >>> print(output_image.shape)
    (4, 3, 2)

    """

    def __init__(
        self,
        axes: Optional[Tuple[int, ...]] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Transpose feature.

        Parameters
        ----------
        axes : Tuple[int, ...], optional
            A tuple specifying the permutation of the axes. If `None`, the 
            axes are reversed by default.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.
        
        """

        super().__init__(axes=axes, **kwargs)

    def get(
        self,
        image: np.ndarray,
        axes: Optional[Tuple[int, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Transpose the axes of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        axes : Tuple[int, ...], optional
            A tuple specifying the permutation of the axes. If `None`, the 
            axes are reversed by default.
        **kwargs : Any
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
    num_classes : int
        The total number of classes for the one-hot encoding.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Example
    -------
    >>> import numpy as np
    >>> from deeptrack.features import OneHot

    Create an input array of class labels:

    >>> input_data = np.array([0, 1, 2])

    Apply a OneHot feature:

    >>> one_hot_feature = OneHot(num_classes=3)
    >>> one_hot_encoded = one_hot_feature.get(input_data, num_classes=3)
    >>> print(one_hot_encoded)
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    """

    def __init__(
        self,
        num_classes: int,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the OneHot feature.

        Parameters
        ----------
        num_classes : int
            The total number of classes for the one-hot encoding.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(num_classes=num_classes, **kwargs)

    def get(
        self,
        image: np.ndarray,
        num_classes: int,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Convert the input array of class labels into a one-hot encoded array.

        Parameters
        ----------
        image : np.ndarray
            The input array of class labels. The last dimension should contain 
            integers representing class indices.
        num_classes : int
            The total number of classes for the one-hot encoding.
        **kwargs : Any
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
    
    If there are multiple property names, this will return a tuple of lists of 
    the property values.

    Parameters
    ----------
    feature : Feature
        The feature from which to extract properties.
    names : List[str]
        The names of the properties to extract

    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Attributes
    ----------
    __distributed__ : bool
        Indicates whether this feature distributes computation across inputs.
        Always `False` for `TakeProperties`, as it processes sequentially.
    __list_merge_strategy__ : int
        Specifies how lists of properties are merged. Set to 
        `MERGE_STRATEGY_APPEND` to append values to the result list.

    Example
    -------
    >>> from deeptrack.features import Feature, TakeProperties
    >>> from deeptrack.properties import Property
    
    >>> class ExampleFeature(Feature):
    ...     def __init__(self, my_property, **kwargs):
    ...         super().__init__(my_property=my_property, **kwargs)

    Create an example feature with a property:
    
    >>> feature = ExampleFeature(my_property=Property(42))

    Use `TakeProperties` to extract the property:
    
    >>> take_properties = TakeProperties(feature, "my_property")
    >>> output = take_properties.get(image=None, names=["my_property"])
    >>> print(output)
    [42]
    
    """

    __distributed__: bool = False
    __list_merge_strategy__: int = MERGE_STRATEGY_APPEND

    def __init__(
        self,
        feature,
        *names,
        **kwargs,
    ):
        """
        Initialize the TakeProperties feature.

        Parameters
        ----------
        feature : Feature
            The feature from which to extract properties.
        *names : List[str]
            One or more names of the properties to extract.
        **kwargs : Dict[str, Any]
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(names=names, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self,
        image: Any,
        names: Tuple[str, ...],
        _ID: Tuple[int, ...] = (),
        **kwargs: Dict[str, Any],
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Extract the specified properties from the feature pipeline.

        Parameters
        ----------
        image : Any
            The input image (unused in this method).
        names : Tuple[str, ...]
            The names of the properties to extract.
        _ID : Tuple[int, ...], optional
            A unique identifier for the current computation, used to match 
            dependencies. Defaults to an empty tuple.
        **kwargs : Any
            Additional keyword arguments (unused here).

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, ...]
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
