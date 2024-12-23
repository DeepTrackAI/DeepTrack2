"""Base class Feature and structural features

Provides classes and tools for creating and interacting with features.
"""

import itertools
import operator
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
from pint import Quantity
import skimage
import skimage.measure

from deeptrack.sources import SourceItem

from .backend.core import DeepTrackNode
from .backend.units import ConversionTable, create_context
from .backend import config
from .image import Image
from .properties import PropertyDict
from .types import ArrayLike, PropertyLike
from . import units

MERGE_STRATEGY_OVERRIDE = 0
MERGE_STRATEGY_APPEND = 1


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

    def __init__(self, _input: Any = [], **kwargs: Dict[str, Any]):
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
        self,
        image: Union['Image', List['Image']],
        **kwargs: Dict[str, Any],
    ) -> Union['Image', List['Image']]:
        """Transform an image [abstract method].
        
        Abstract method that defines how the feature transforms the input. The 
        current value of all properties will be passed as keyword arguments.

        Parameters
        ---------
        image : 'Image' or List['Image']
            The Image or list of images to transform.
        **kwargs : Dict[str, Any]
            The current value of all properties in `properties` as well as any 
            global arguments.

        Returns
        -------
        'Image' or List['Image']
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


def propagate_data_to_dependencies(X, **kwargs):
    """Iterates the dependencies of a feature and sets the value of their properties to the values in kwargs.

    Parameters
    ----------
    X : features.Feature
        The feature whose dependencies are to be updated
    kwargs : dict
        The values to be set for the properties of the dependencies.
    """
    for dep in X.recurse_dependencies():
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

    __distributed__ = False

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

    A typical use-case is::

       arguments = Arguments(is_label=False)
       image_loader = (
           LoadImage(path="./image.png") >>
           GaussianNoise(sigma = (1 - arguments.is_label) * 5)
       )
       image_loader.bind_arguments(arguments)

       image_loader()              # Image with added noise
       image_loader(is_label=True) # Raw image with no noise

    For non-mathematical dependence, create a local link to the property as follows::

       arguments = Arguments(is_label=False)
       image_loader = (
           LoadImage(path="./image.png") >>
           GaussianNoise(
              is_label=arguments.is_label,
              sigma=lambda is_label: 0 if is_label else 5
           )
       )
       image_loader.bind_arguments(arguments)

       image_loader()              # Image with added noise
       image_loader(is_label=True) # Raw image with no noise

    Keep in mind that if any dependent property is non-deterministic,
    they may permanently change::
       arguments = Arguments(noise_max_sigma=5)
       image_loader = (
           LoadImage(path="./image.png") >>
           GaussianNoise(
              noise_max_sigma=5,
              sigma=lambda noise_max_sigma: rand() * noise_max_sigma
           )
       )

       image_loader.bind_arguments(arguments)

       image_loader().get_property("sigma") # 3.27...
       image_loader(noise_max_sigma=0) # 0
       image_loader().get_property("sigma") # 1.93...

    As with any feature, all arguments can be passed by deconstructing the properties dict::

       arguments = Arguments(is_label=False, noise_sigma=5)
       image_loader = (
           LoadImage(path="./image.png") >>
           GaussianNoise(
              sigma=lambda is_label, noise_sigma: 0 if is_label else noise_sigma
              **arguments.properties
           )
       )
       image_loader.bind_arguments(arguments)

       image_loader()              # Image with added noise
       image_loader(is_label=True) # Raw image with no noise


    """

    def get(self, image, **kwargs):
        return image


class Probability(StructuralFeature):
    """Resolves a feature with a certain probability

    Parameters
    ----------
    feature : Feature
        Feature to resolve
    probability : float
        Probability to resolve
    """

    def __init__(
        self, feature: Feature, probability: PropertyLike[float], *args, **kwargs
    ):
        super().__init__(
            *args,
            feature=feature,
            probability=probability,
            random_number=np.random.rand,
            **kwargs
        )

    def get(
        self,
        image,
        feature: Feature,
        probability: float,
        random_number: float,
        **kwargs
    ):
        """Resolves `feature` if `random_number` is less than `probability`"""
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
    """Combines multiple features into a single feature.

    Resolves each feature in `features` and returns them as a list of features.

    Parameters
    ----------
    features : list of features
        features to combine

    """

    __distributed__ = False

    def __init__(self, features: List[Feature], **kwargs):
        self.features = [self.add_feature(f) for f in features]
        super().__init__(**kwargs)

    def get(self, image_list, **kwargs):
        return [f(image_list, **kwargs) for f in self.features]


class Slice(Feature):
    """Array indexing for each Image in list.

    Note, this feature is rarely needed to be used directly. Instead,
    you can do normal array indexing on a feature directly. For example::

       feature = dt.DummyFeature()
       sliced_feature = feature[
           lambda: 0 : lambda: 1,
           1:2,
           lambda: slice(None, None, -2)
       ]
       sliced_feature.resolve(np.arange(27).reshape((3, 3, 3)))

    In the example above, `lambda` is used to demonstrate different ways
    to interact with the slices. In this case, the `lambda` keyword is
    redundant.

    Using `Slice` directly can be required in some cases, however. For example if
    dependencies between properties are required. In this case, one can replicate
    the previous example as follows::

       feature = dt.DummyFeature()
       sliced_feature = feature + dt.Slice(
           slices=lambda dim1, dim2: (dim1, dim2),
           dim1=slice(lambda: 0, lambda: 1, 1),
           dim2=slice(1, 2, None),
           dim3=lambda: slice(None, None, -2)
       )
       sliced_feature.resolve(np.arange(27).reshape((3, 3, 3)))

    Parameters
    ----------
    slices : iterable of int, slice or ellipsis
        The indexing of each dimension in order.
    """

    def __init__(
        self,
        slices: PropertyLike[
            Iterable[PropertyLike[int] or PropertyLike[slice] or PropertyLike[...]]
        ],
        **kwargs
    ):
        super().__init__(slices=slices, **kwargs)

    def get(self, image, slices, **kwargs):

        try:
            slices = tuple(slices)
        except ValueError:
            pass

        return image[slices]


class Bind(StructuralFeature):
    """Binds a feature with property arguments.

    When the feature is resolved, the kwarg arguments are passed
    to the child feature.

    Parameters
    ----------
    feature : Feature
        The child feature
    **kwargs
        Properties to send to child

    """

    __distributed__ = False

    def __init__(self, feature: Feature, **kwargs):

        super().__init__(**kwargs)
        self.feature = self.add_feature(feature)

    def get(self, image, **kwargs):
        return self.feature.resolve(image, **kwargs)


BindResolve = Bind


class BindUpdate(StructuralFeature):
    """Binds a feature with certain arguments.

    When the feature is updated, the child feature

    Parameters
    ----------
    feature : Feature
        The child feature
    **kwargs
        Properties to send to child

    """

    __distributed__ = False

    def __init__(self, feature: Feature, **kwargs):
        import warnings

        warnings.warn(
            "BindUpdate is deprecated and may be removed in a future release."
            "The current implementation is not guaranteed to be exactly equivalent to prior implementations. "
            "Please use Bind instead.",
            DeprecationWarning,
        )

        super().__init__(**kwargs)
        self.feature = self.add_feature(feature)

    def get(self, image, **kwargs):
        return self.feature.resolve(image, **kwargs)


class ConditionalSetProperty(StructuralFeature):
    """Conditionally overrides the properties of child features.

    It is adviceable to use dt.Arguments instead. Note that this overwrites the properties, and as
    such may affect future calls.

    Parameters
    ----------
    feature : Feature
        The child feature
    condition : bool-like or str
        A boolean or the name a boolean property
    **kwargs
        Properties to be used if `condition` is True

    """

    __distributed__ = False

    def __init__(self, feature: Feature, condition=PropertyLike[str or bool], **kwargs):

        super().__init__(condition=condition, **kwargs)
        self.feature = self.add_feature(feature)

    def get(self, image, condition, **kwargs):

        _condition = condition
        if isinstance(condition, str):
            _condition = kwargs.get(condition, False)

        if _condition:
            propagate_data_to_dependencies(self.feature, **kwargs)

        return self.feature(image)


class ConditionalSetFeature(StructuralFeature):
    """Conditionally resolves one of two features

    Set condition to the value to listen to. Example,
    if condition is "is_label", then conditiona can be toggled
    by calling either

    Feature.resolve(is_label=True) / Feature.resolve(is_label=False)
    Feature.update(is_label=True) / Feature.update(is_label=False)

    Note that both features will be updated in either case.

    Parameters
    ----------
    on_false : Feature
        Feature to resolve if the conditional property is false
    on_true : Feature
        Feature to resolve if the conditional property is true
    condition : str
        The name of the conditional property

    """

    __distributed__ = False

    def __init__(
        self,
        on_false: Feature = None,
        on_true: Feature = None,
        condition: PropertyLike[str or bool] = "is_label",
        **kwargs
    ):

        super().__init__(condition=condition, **kwargs)
        if on_true:
            self.add_feature(on_true)
        if on_false:
            self.add_feature(on_false)

        self.on_true = on_true
        self.on_false = on_false

    def get(self, image, *, condition, **kwargs):

        _condition = condition
        if isinstance(condition, str):
            _condition = kwargs.get(condition, False)

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
    """Calls a custom function on each image in the input.

    Note that the property `function` needs to be wrapped in an
    outer layer function. The outer layer function can depend on
    other properties, while the inner layer function accepts an
    image as input.

    Parameters
    ----------
    function : Callable[Image]
        Function that takes the current image as first input
    """

    def __init__(self, function: Callable[..., Callable[[Image], Image]], **kwargs):
        super().__init__(function=function, **kwargs)

    def get(self, image, function, **kwargs):
        return function(image)


class Merge(Feature):
    """Calls a custom function on the entire input.

    Note that the property `function` needs to be wrapped in an
    outer layer function. The outer layer function can depend on
    other properties, while the inner layer function accepts an
    image as input.

    Parameters
    ----------
    function : Callable[list of Image]
        Function that takes the current image as first input
    """

    __distributed__ = False

    def __init__(
        self,
        function: Callable[..., Callable[[List[Image]], Image or List[Image]]],
        **kwargs
    ):
        super().__init__(function=function, **kwargs)

    def get(self, list_of_images, function, **kwargs):
        return function(list_of_images)


class OneOf(Feature):
    """Resolves one feature from a collection on the input.

    Valid collections are any object that can be iterated (such as lists, tuples and sets).
    Internally, the collection is converted to a tuple.

    Default behaviour is to sample the collection uniformly random. This can be
    controlled by the `key` argument, where the feature resolved is chosen as
    `tuple(collection)[key]`.
    """

    __distributed__ = False

    def __init__(self, collection, key=None, **kwargs):
        self.collection = tuple(collection)
        super().__init__(key=key, **kwargs)

        for feature in self.collection:
            self.add_feature(feature)

    def _process_properties(self, propertydict) -> dict:
        super()._process_properties(propertydict)

        if propertydict["key"] is None:
            propertydict["key"] = np.random.randint(len(self.collection))

        return propertydict

    def get(self, image, key, _ID=(), **kwargs):
        return self.collection[key](image, _ID=_ID)


class OneOfDict(Feature):
    """Resolves one feature from a dictionary.

    Default behaviour is to sample the values diction uniformly random. This can be
    controlled by the `key` argument, where the feature resolved is chosen as
    `collection[key]`.
    """

    __distributed__ = False

    def __init__(self, collection, key=None, **kwargs):

        self.collection = collection

        super().__init__(key=key, **kwargs)

        for feature in self.collection.values():
            self.add_feature(feature)

    def _process_properties(self, propertydict) -> dict:
        super()._process_properties(propertydict)

        if propertydict["key"] is None:
            propertydict["key"] = np.random.choice(list(self.collection.keys()))

        return propertydict

    def get(self, image, key, _ID=(), **kwargs):
        return self.collection[key](image, _ID=_ID)


class Label(Feature):
    """Outputs the properties of this features.

    Can be used to extract properties in a feature set and combine them into
    a numpy array.

    Parameters
    ----------
    output_shape : tuple of ints
        Reshapes the output to this shape

    """

    __distributed__ = False

    def __init__(self, output_shape: PropertyLike[int] = None, **kwargs):
        super().__init__(output_shape=output_shape, **kwargs)

    def get(self, image, output_shape=None, **kwargs):
        result = []
        for key in self.properties.keys():
            if key in kwargs:
                result.append(kwargs[key])

        if output_shape:
            result = np.reshape(np.array(result), output_shape)

        return np.array(result)


class LoadImage(Feature):
    """Loads an image from disk.

    Cycles through file-readers numpy, pillow and opencv2 to open the
    image file.

    Parameters
    ----------
    path : str
        Path to image to load
    load_options : dict
        Options passed to the file reader
    as_list : bool
        If True, the irst dimension will be converted to a list.
    ndim : int
        Adds dimensions until it is at least ndim
    to_grayscale : bool
        Whether to convert the image to grayscale
    get_one_random : bool
        Extracts a single image from a stack. Only used if as_list is true.

    Raises
    ------
    IOError
        If no file reader could parse the file or the file doesn't exist.

    """

    __distributed__ = False

    def __init__(
        self,
        path: PropertyLike[str or List[str]],
        load_options: PropertyLike[dict] = None,
        as_list: PropertyLike[bool] = False,
        ndim: PropertyLike[int] = 3,
        to_grayscale: PropertyLike[bool] = False,
        get_one_random: PropertyLike[bool] = False,
        **kwargs
    ):
        super().__init__(
            path=path,
            load_options=load_options,
            as_list=as_list,
            ndim=ndim,
            to_grayscale=to_grayscale,
            get_one_random=get_one_random,
            **kwargs
        )

    def get(
        self,
        *ign,
        path,
        load_options,
        ndim,
        to_grayscale,
        as_list,
        get_one_random,
        **kwargs
    ):

        path_is_list = isinstance(path, list)
        if not path_is_list:
            path = [path]
        if load_options is None:
            load_options = {}

        
        try:
            import imageio
            image = [imageio.v3.imread(file) for file in path]
        except (IOError, ImportError, AttributeError):
            try:
                image = [np.load(file, **load_options) for file in path]
            except (IOError, ValueError):
                try:
                    import PIL.Image

                    image = [PIL.Image.open(file, **load_options) for file in path]
                except (IOError, ImportError):
                    import cv2

                    image = [cv2.imread(file, **load_options) for file in path]
                    if not image:
                        raise IOError(
                            "No filereader available for file {0}".format(path)
                        )

        if as_list:
            if get_one_random:
                image = image[np.random.randint(len(image))]
            else:
                image = list(image)
        elif path_is_list:
            image = np.stack(image, axis=-1)
        else:
            image = image[0]

        if to_grayscale:
            try:
                import skimage

                skimage.color.rgb2gray(image)
            except ValueError:
                import warnings

                warnings.warn("Non-rgb image, ignoring to_grayscale")

        while ndim and image.ndim < ndim:
            image = np.expand_dims(image, axis=-1)

        return image


class SampleToMasks(Feature):
    """Creates a mask from a list of images.

    Calls `transformation_function` for each input image, and merges the outputs
    to a single image with `number_of_masks` layers. Each input image needs to have
    a defined property `position` to place it within the image. If used with scatterers,
    note that the scatterers need to be passed the property `voxel_size` to correctly
    size the objects.

    Parameters
    ----------
    transformation_function : function
       Function that takes an image as input, and outputs another image with `number_of_masks`
       layers.
    number_of_masks : int
       The number of masks to create.
    output_region : (int, int, int, int)
       Size and relative position of the mask. Should generally be the same as
       `optics.output_region`.
    merge_method : str or function or list
       How to merge the individual masks to a single image. If a list, the merge_metod
       is per mask. Can be
          * "add": Adds the masks together.
          * "overwrite": later masks overwrite earlier masks.
          * "or": 1 if either any mask is non-zero at that pixel
          * function: a function that accepts two images. The first is the current
            value of the output image where a new mask will be places, and
            the second is the mask to merge with the output image.

    """

    def __init__(
        self,
        transformation_function: Callable[..., Callable[[Image], Image]],
        number_of_masks: PropertyLike[int] = 1,
        output_region: PropertyLike[ArrayLike[int]] = None,
        merge_method: PropertyLike[str] = "add",
        **kwargs
    ):
        super().__init__(
            transformation_function=transformation_function,
            number_of_masks=number_of_masks,
            output_region=output_region,
            merge_method=merge_method,
            **kwargs
        )

    def get(self, image, transformation_function, **kwargs):
        return transformation_function(image)

    def _process_and_get(self, images, **kwargs):
        if isinstance(images, list) and len(images) != 1:
            list_of_labels = super()._process_and_get(images, **kwargs)
            if not self._wrap_array_with_image:
                for idx, (label, image) in enumerate(zip(list_of_labels, images)):
                    list_of_labels[idx] = Image(label, copy=False).merge_properties_from(image)
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

        output_region = kwargs["output_region"]
        output = np.zeros(
            (
                output_region[2] - output_region[0],
                output_region[3] - output_region[1],
                kwargs["number_of_masks"],
            )
        )

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


def _get_position(image, mode="corner", return_z=False):
    # Extracts the position of the upper left corner of a scatterer

    if mode == "corner":
        shift = (np.array(image.shape) - 1) / 2
    else:
        shift = np.zeros((3 if return_z else 2))

    positions = image.get_property("position", False, [])

    positions_out = []
    for position in positions:
        if len(position) == 3:
            if return_z:
                return positions_out.append(position - shift)
            else:
                return positions_out.append(position[0:2] - shift[0:2])

        elif len(position) == 2:
            if return_z:
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
                positions_out.append(position - shift[0:2])

    return positions_out


class AsType(Feature):
    """Converts the data type of images

    Accepts same types as numpy arrays. Common types include

    `float64, int32, uint16, int16, uint8, int8`

    Parameters
    ----------
    dtype : str
        dtype string. Same as numpy dtype.

    """

    def __init__(self, dtype: PropertyLike[Any] = "float64", **kwargs):
        super().__init__(dtype=dtype, **kwargs)

    def get(self, image, dtype, **kwargs):
        return image.astype(dtype)


class ChannelFirst2d(Feature):
    """Converts a 3d image to channel first format.

    Parameters
    ----------
    axis : int
        The axis to move to the first position. Defaults to -1.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(axis=axis, **kwargs)
    def get(self, image, axis, **kwargs):
        ndim = image.ndim

        if ndim == 2:
            return image[None]
        elif ndim == 3:
            return np.moveaxis(image, axis, 0)


class Upscale(Feature):
    """Performs the simulation at a higher resolution.

    Redefines the sizes of internal units to scale up the simulation. The resulting image
    is then downscaled back to the original size. Example::

       optics = dt.Fluorescence()
       particle = dt.Sphere()
       pipeline = optics(particle)
       upscaled_pipeline = dt.Upscale(pipeline, factor=4)

    Parameters
    ----------
    feature : Feature
        The pipeline to resolve at a higher resolution
    factor : int or tuple of ints
        The factor to scale up the simulation by. If a tuple of three integers,
        each axis is scaled up individually.

    """

    __distributed__ = False

    def __init__(self, feature, factor=1, **kwargs):
        super().__init__(factor=factor, **kwargs)
        self.feature = self.add_feature(feature)

    def get(self, image, factor, **kwargs):
        if np.size(factor) == 1:
            factor = (factor,) * 3
        ctx = create_context(None, None, None, *factor)
        with units.context(ctx):
            image = self.feature(image)

        image = skimage.measure.block_reduce(
            image, (factor[0], factor[1]) + (1,) * (image.ndim - 2), np.mean
        )

        return image


class NonOverlapping(Feature):

    __distributed__ = False

    def __init__(self, feature, min_distance=1, max_attempts=100, **kwargs):
        """Places a list of volumes non-overlapping.

        Ensures that the volumes are placed non-overlapping by resampling the position of the volumes until they are non-overlapping.
        If the maximum number of attempts is exceeded, a new list of volumes is generated by updating feature.

        Note: This feature does not work with non-volumetric scatterers, such as MieScatterers.

        Parameters
        ----------
        feature : Feature
            The feature that creates the list of volumes to be placed non-overlapping.
        min_distance : float, optional
            The minimum distance between volumes in pixels, by default 1
        max_attempts : int, optional
            The maximum number of attempts to place the volumes non-overlapping. If this number is exceeded, a new list of volumes is generated, by default 100.
        """
        super().__init__(min_distance=min_distance, max_attempts=max_attempts, **kwargs)
        self.feature = self.add_feature(feature, **kwargs)

    def get(self, _, min_distance, max_attempts, **kwargs):
        """
        Parameters
        ----------
        list_of_volumes : list of 3d arrays
            The volumes to be placed non-overlapping
        min_distance : float
            The minimum distance between volumes in pixels.
        max_attempts : int
            The maximum number of attempts to place the volumes non-overlapping. If this number is exceeded, a new list of volumes is generated.
        """
        while True:
            list_of_volumes = self.feature()

            if not isinstance(list_of_volumes, list):
                list_of_volumes = [list_of_volumes]

            for attempt in range(max_attempts):

                list_of_volumes = [
                    self._resample_volume_position(volume) for volume in list_of_volumes
                ]

                if self._check_non_overlapping(list_of_volumes):
                    return list_of_volumes

            self.feature.update()

    def _check_non_overlapping(self, list_of_volumes):
        """
        Checks that the non-zero voxels of the volumes in list_of_volumes are at least min_distance apart.
        Each volume is a 3 dimnesional array. The first two dimensions are the x and y dimensions, and the third dimension is the z dimension.
        The volumes are expected to have a position attribute.

        Parameters
        ----------
        list_of_volumes : list of 3d arrays
            The volumes to be checked for non-overlapping
        """
        from skimage.morphology import isotropic_erosion
        from .optics import _get_position
        from .augmentations import CropTight

        min_distance = self.min_distance()
        if min_distance < 0:
            crop = CropTight()
            # print([np.sum(volume != 0) for volume in list_of_volumes])
            list_of_volumes = [Image(crop(isotropic_erosion(volume != 0, -min_distance/2)), copy=False).merge_properties_from(volume) for volume in list_of_volumes]
            # print([np.sum(volume != 0) for volume in list_of_volumes])

            min_distance = 1

        # The position of the top left corner of each volume (index (0, 0, 0))
        volume_positions_1 = [
            _get_position(volume, mode="corner", return_z=True).astype(int)
            for volume in list_of_volumes
        ]

        # The position of the bottom right corner of each volume (index (-1, -1, -1))
        volume_positions_2 = [
            p0 + np.array(v.shape) for v, p0 in zip(list_of_volumes, volume_positions_1)
        ]

        # (x1, y1, z1, x2, y2, z2) for each volume
        volume_bounding_cube = [
            [*p0, *p1] for p0, p1 in zip(volume_positions_1, volume_positions_2)
        ]

        for i, j in itertools.combinations(range(len(list_of_volumes)), 2):
            # If the bounding cubes do not overlap, the volumes do not overlap
            if self._check_bounding_cubes_non_overlapping(
                volume_bounding_cube[i], volume_bounding_cube[j], min_distance
            ):
                continue

            # If the bounding cubes overlap, get the overlapping region of each volume
            overlapping_cube = self._get_overlapping_cube(
                volume_bounding_cube[i], volume_bounding_cube[j]
            )
            overlapping_volume_1 = self._get_overlapping_volume(
                list_of_volumes[i], volume_bounding_cube[i], overlapping_cube
            )
            overlapping_volume_2 = self._get_overlapping_volume(
                list_of_volumes[j], volume_bounding_cube[j], overlapping_cube
            )

            # If either the overlapping regions are empty, the volumes do not overlap (done for speed)
            if np.all(overlapping_volume_1 == 0) or np.all(overlapping_volume_2 == 0):
                continue

            # If the products of the overlapping regions are non-zero, return False
            # if np.any(overlapping_volume_1 * overlapping_volume_2):
            #     return False

            # Finally, check that the non-zero voxels of the volumes are at least min_distance apart
            if not self._check_volumes_non_overlapping(
                overlapping_volume_1, overlapping_volume_2, min_distance
            ):
                return False

        return True

    def _check_bounding_cubes_non_overlapping(
        self, bounding_cube_1, bounding_cube_2, min_distance
    ):

        # bounding_cube_1 and bounding_cube_2 are (x1, y1, z1, x2, y2, z2)
        # Check that the bounding cubes are non-overlapping
        return (
            bounding_cube_1[0] > bounding_cube_2[3] + min_distance
            or bounding_cube_1[1] > bounding_cube_2[4] + min_distance
            or bounding_cube_1[2] > bounding_cube_2[5] + min_distance
            or bounding_cube_1[3] < bounding_cube_2[0] - min_distance
            or bounding_cube_1[4] < bounding_cube_2[1] - min_distance
            or bounding_cube_1[5] < bounding_cube_2[2] - min_distance
        )

    def _get_overlapping_cube(self, bounding_cube_1, bounding_cube_2):
        """
        Returns the overlapping region of the two bounding cubes.
        """
        return [
            max(bounding_cube_1[0], bounding_cube_2[0]),
            max(bounding_cube_1[1], bounding_cube_2[1]),
            max(bounding_cube_1[2], bounding_cube_2[2]),
            min(bounding_cube_1[3], bounding_cube_2[3]),
            min(bounding_cube_1[4], bounding_cube_2[4]),
            min(bounding_cube_1[5], bounding_cube_2[5]),
        ]

    def _get_overlapping_volume(self, volume, bounding_cube, overlapping_cube):
        """
        Returns the overlapping region of the volume and the overlapping cube.

        Parameters
        ----------
        volume : 3d array
            The volume to be checked for non-overlapping
        bounding_cube : list of 6 floats
            The bounding cube of the volume.
            The first three elements are the position of the top left corner of the volume, and the last three elements are the position of the bottom right corner of the volume.
        overlapping_cube : list of 6 floats
            The overlapping cube of the volume and the other volume.
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

    def _check_volumes_non_overlapping(self, volume_1, volume_2, min_distance):
        """
        Checks that the non-zero voxels of the volumes are at least min_distance apart.
        """
        # Get the positions of the non-zero voxels of each volume
        positions_1 = np.argwhere(volume_1)
        positions_2 = np.argwhere(volume_2)

        # If the volumes are not the same size, the positions of the non-zero voxels of each volume need to be scaled
        if volume_1.shape != volume_2.shape:
            positions_1 = (
                positions_1 * np.array(volume_2.shape) / np.array(volume_1.shape)
            )
            positions_1 = positions_1.astype(int)

        # Check that the non-zero voxels of the volumes are at least min_distance apart
        import scipy.spatial.distance

        return np.all(
            scipy.spatial.distance.cdist(positions_1, positions_2) > min_distance
        )

    def _resample_volume_position(self, volume):
        """Draws a new position for the volume."""

        for pdict in volume.properties:
            if "position" in pdict and "_position_sampler" in pdict:
                new_position = pdict["_position_sampler"]()
                if isinstance(new_position, Quantity):
                    new_position = new_position.to("pixel").magnitude
                pdict["position"] = new_position

        return volume


class Store(Feature):

    __distributed__ = False

    def __init__(self, feature, key, replace=False, **kwargs):
        super().__init__(feature=feature, key=key, replace=replace, **kwargs)

        self.feature = self.add_feature(feature, **kwargs)

        self._store: dict[Any, Image] = {}

    def get(self, _, key, replace, **kwargs):
        if replace or not (key in self._store):
            self._store[key] = self.feature()
        if self._wrap_array_with_image:
            return Image(self._store[key], copy=False)
        else:
            return self._store[key]
        # return self._store[key] 


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
    ):
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
    ):
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

    Parameters
    ----------
    source : int
        The axis to move.
    destination : int
        The destination of the axis.
    """

    def __init__(self, source, destination, **kwargs):
        super().__init__(source=source, destination=destination, **kwargs)

    def get(self, image, source, destination, **kwargs):
        return np.moveaxis(image, source, destination)


class Transpose(Feature):
    """Transposes the input image.

    Parameters
    ----------
    axes : tuple of ints
        The axes to transpose.
    """

    def __init__(self, axes, **kwargs):
        super().__init__(axes=axes, **kwargs)

    def get(self, image, axes, **kwargs):
        return np.transpose(image, axes)


Permute = Transpose


class OneHot(Feature):
    """Converts the input to a one-hot encoded array.

    Parameters
    ----------
    num_classes : int
        The number of classes to encode.
    """
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)

    def get(self, image, num_classes, **kwargs):
        if image.shape[-1] == 1:
            image = image[..., 0]
        return np.eye(num_classes)[image]


class TakeProperties(Feature):
    """Extracts all instances of a set of properties from a pipeline

    Only extracts the properties if the feature contains all given property-names.
    Order of the properties is not guaranteed to be the same as the evaluation order.

    If there is only a single property name, this will return a list of the property values.
    If there are multiple property names, this will return a tuple of lists of the property values.

    Parameters
    ----------
    feature : Feature
        The feature to extract the properties from
    names : list of str
        The names of the properties to extract
    """
    __distributed__ = False
    __list_merge_strategy__ = MERGE_STRATEGY_APPEND

    def __init__(self, feature, *names, **kwargs):
        super().__init__(names=names, **kwargs)
        self.feature = self.add_feature(feature)

    def get(self, image, names, _ID=(), **kwargs):

        if not self.feature.is_valid(_ID=_ID):
            self.feature(_ID=_ID)

        res = {}
        for name in names:
            res[name] = []

        for dep in self.feature.recurse_dependencies():
            if isinstance(dep, PropertyDict) and all(name in dep for name in names):
                # if all names are in dep, 
                
                for name in names:
                    data = dep[name].data.dict
                    for key, value in data.items():
                        
                        if key[:len(_ID)] == _ID:
                            res[name].append(value.current_value())

        res = tuple([np.array(res[name]) for name in names])
        if len(res) == 1:
            res = res[0]
        return res
