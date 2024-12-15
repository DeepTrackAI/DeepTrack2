"""Base class Feature and structural features

Provides classes and tools for creating and interacting with features.
"""

import itertools
import operator
from typing import Any, Callable, Iterable, List, Tuple, Union
import warnings
import random

import numpy as np
from pint import Quantity
# import tensorflow as tf #TBE
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
    _input : Union['Image', List['Image']], optional
        Defines a list of DeepTrackNode objects that calculate the input of the 
        feature. In most cases, this can be left empty.
    **kwargs : Any
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

    __list_merge_strategy__ = MERGE_STRATEGY_OVERRIDE
    __distributed__ = True
    __property_memorability__ = 1
    __conversion_table__ = ConversionTable()
    __gpu_compatible__ = False

    _wrap_array_with_image = False

    def __init__(self, _input=[], **kwargs):

        super().__init__()

        # Add all keyword arguments as properties.
        # In most cases, properties does not yet exist as an attribute.
        properties = getattr(self, "properties", {})
        properties.update(**kwargs)
        properties.setdefault("name", type(self).__name__)

        # Create propertydict and add it to the computation graph.
        self.properties = PropertyDict(**properties)
        self.properties.add_child(self)
        # self.add_dependency(self.properties)  # Executed by add_child.

        # The input of the feature is added as a dependency.
        # This lets the feature know that it needs to be recalculated if the 
        # input changes.
        # _input is set when the feature is called.
        self._input = DeepTrackNode(_input)
        self._input.add_child(self)
        # self.add_dependency(self._input)  # Executed by add_child.

        # A random seed can be set to make the feature deterministic.
        # A non-deterministic feature does not need to be recalculated if the 
        # seed is the same.
        self._random_seed = DeepTrackNode(lambda: random.randint(0, 2147483648))
        self._random_seed.add_child(self)
        # self.add_dependency(self._random_seed)  # Executed by add_child.

        # Initilaize arguments
        self.arguments = None

    def get(
        self,
        image: Union['Image', List['Image']],
        **kwargs: Any,
    ) -> Union['Image', List['Image']]:
        """Transform an image [abstract method].
        
        Abstract method that define how the feature transforms the input. The 
        current value of all properties will be passed as keyword arguments.

        Parameters
        ---------
        image : Union['Image', List['Image']]
            The Image or list of images to transform.
        **kwargs
            The current value of all properties in `properties` as well as any 
            global arguments.

        Returns
        -------
        Union['Image', List['Image']]
            The transformed image or list of images.
        
        """

        raise NotImplementedError

    def __call__(
        self,
        image_list: Union['Image', List['Image']] = None,
        _ID: Tuple[int, ...] = (),
        **kwargs: Any,
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
        image_list : Union['Image', List['Image']], optional
            The input to the feature or pipeline. If `None`, the feature uses 
            previously set input values or propagates properties.
        **kwargs : Any
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
                    original_values[key] = self.arguments.properties[key](_ID=_ID)
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
        x: bool = True, 
        recursive: bool = True,
    ) -> None:
        """Store properties to the Image.

        Parameters
        ----------
        x : bool
            Whether to store properties to the Image. If False, properties will not be stored.
        recursive : bool
            Whether to store properties to all dependencies of the feature.
        """
        self._wrap_array_with_image = x
        if recursive:
            for dep in self.recurse_dependencies():
                if isinstance(dep, Feature):
                    dep.store_properties(x, False)

    def torch(self, dtype=None, device=None, permute_mode="never"):
        """Convert the feature to a PyTorch feature.

        Parameters
        ----------
        dtype : torch.dtype
            The dtype of the output.
        device : torch.device
            The device of the output.

        Returns
        -------
        Feature
            A PyTorch feature.
        """
        from .pytorch import ToTensor
        tensor_feature = ToTensor(dtype=dtype, device=device, permute_mode=permute_mode)
        tensor_feature.store_properties(False, recursive=False)
        return self >> tensor_feature

    def batch(self, batch_size=32):
        """Batch the feature.

        Parameters
        ----------
        batch_size : int
            The size of the batch.

        Returns
        -------
        Feature
            A batched feature.
        """
        res = [self.update()() for _ in range(batch_size)]
        res = list(zip(*res))

        for idx, r in enumerate(res):

            if isinstance(r[0], np.ndarray):
                res[idx] = np.stack(r)
            else:
                import torch
                if isinstance(r[0], torch.Tensor):
                    res[idx] = torch.stack(r)

        return tuple(res)

    def action(self, _ID=()):
        """Creates the image.
        Transforms the input image by calling the method `get()` with the
        correct inputs. The properties of the feature can be overruled by
        passing a different value as a keyword argument.

        Parameters
        ----------
        _ID : tuple
            The ID of the current execution.

        Returns
        -------
        Image or List[Image]
            The resolved image
        """

        image_list = self._input(_ID=_ID)

        # Get the input arguments to the method .get()
        feature_input = self.properties(_ID=_ID).copy()

        # Call the _process_properties hook, default does nothing.
        # Can be used to ensure properties are formatted correctly
        # or to rescale properties.
        feature_input = self._process_properties(feature_input)
        if _ID != ():
            feature_input["_ID"] = _ID

        # Ensure that input is a list
        image_list = self._format_input(image_list, **feature_input)

        # Set the seed from the hash_key. Ensures equal results
        # self.seed(_ID=_ID)

        # _process_and_get calls the get function correctly according
        # to the __distributed__ attribute
        new_list = self._process_and_get(image_list, **feature_input)

        self._process_output(new_list, feature_input)


        # Merge input and new_list
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
        """Determine if the feature should use the GPU."""
        return self.__gpu_compatible__ and np.prod(np.shape(inp)) > (90000)

    def update(self, **_):
        """Refresh the feature to create a new image.

        Per default, when a feature is called multiple times, it will return the same value.
        To tell the feature to return a new value, we first call `update`.
        """
        self._update()
        return self

    def _update(self, **global_arguments):
        
        if global_arguments:
            # Deptracated, but not necessary to raise hard error.
            warnings.warn(
                "Passing information through .update is no longer supported. "
                "A quick fix is to pass the information when resolving the feature. "
                "The prefered solution is to use dt.Arguments",
                DeprecationWarning,
            )
            
        super()._update()
        return self

    def add_feature(self, feature):
        """Adds a feature to the dependecy graph."""
        feature.add_child(self)
        self.add_dependency(feature)
        return feature

    def seed(self, _ID=()):
        """Seed the random number generator."""
        np.random.seed(self._random_seed(_ID=_ID))

    def bind_arguments(self, arguments):
        """See `features.Arguments`"""
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

    def __getattr__(self, key):
        # Allows easier access to properties. For example,
        # feature.my_property is equivalent to feature.properties["my_property"]

        if "properties" in self.__dict__:
            properties = self.__dict__["properties"]
            if key in properties:
                return properties[key]
            else:
                raise AttributeError
        else:
            raise AttributeError

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

        # Import here to avoid circular import.
        

        # If other is a function, call it on the output of the feature.
        # For example, feature >> some_function

        if callable(other):
            return self >> Lambda(lambda: other)

        return NotImplemented

    def __rrshift__(self, other: "Feature") -> 'Feature':
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
    """Provides the structure of a feature-set
    Feature with __property_verbosity__ = 2 to avoid adding it to the list
    of properties, and __distributed__ = False to pass the input as-is.
    """

    __property_verbosity__ = 2
    __distributed__ = False


class Chain(StructuralFeature):
    """Resolves two features sequentially.
    Passes the output of the first to the input of the second.
    Parameters
    ----------
    feature_1 : Feature
    feature_2 : Feature
    """

    def __init__(self, feature_1: Feature, feature_2: Feature, **kwargs):

        super().__init__(**kwargs)

        self.feature_1 = self.add_feature(feature_1)
        self.feature_2 = self.add_feature(feature_2)

    def get(self, image, _ID=(), **kwargs):
        """Resolves `feature_1` and `feature_2` sequentially"""
        image = self.feature_1(image, _ID=_ID)
        image = self.feature_2(image, _ID=_ID)
        return image


# Alias for backwards compatability
Branch = Chain


class Value(Feature):
    """Represents a constant (per evaluation) value.

    Parameters
    ----------
    value : number
        The value of the feature.
    """

    __distributed__ = False

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):

        if isinstance(value, Image):
            warnings.warn(
                "Setting dt.Value value as a Image object is likely to lead to performance deterioation. Consider converting it to a numpy array using np.array"
            )
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return value


class ArithmeticOperationFeature(Feature):
    """Parent feature of arithmetic operation features like +*-/> etc.

    Inputs can be either single values or a lists of values. If a list is passed, the operation is applied to each element in the list.
    If both inputs are lists of different lengths, the shorter list is cycled.

    Parameters
    ----------
    operation : callable
        The operation to apply.
    value : number
        The other value to apply the operation to."""

    __distributed__ = False
    __gpu_compatible__ = True

    def __init__(self, op, value=0, **kwargs):
        self.op = op
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):

        if not isinstance(value, (list, tuple)):
            value = [value]

        if len(image) < len(value):
            image = itertools.cycle(image)
        elif len(value) < len(image):
            value = itertools.cycle(value)

        return [self.op(a, b) for a, b in zip(image, value)]


class Add(ArithmeticOperationFeature):
    """Adds a value to the input.

    Parameters
    ----------
    value : number
        The value to add
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.add, value=value, **kwargs)


class Subtract(ArithmeticOperationFeature):
    """Subtracts a value from the input.

    Parameters
    ----------
    value : number
        The value to subtract
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.sub, value=value, **kwargs)


class Multiply(ArithmeticOperationFeature):
    """Multiplies the input with a value.

    Parameters
    ----------
    value : number
        The value to multiply with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.mul, value=value, **kwargs)


class Divide(ArithmeticOperationFeature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.truediv, value=value, **kwargs)


class FloorDivide(ArithmeticOperationFeature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.floordiv, value=value, **kwargs)


class Power(ArithmeticOperationFeature):
    """Raises the input to a power.

    Parameters
    ----------
    value : number
        The power to raise with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.pow, value=value, **kwargs)


class LessThan(ArithmeticOperationFeature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.lt, value=value, **kwargs)


class LessThanOrEquals(ArithmeticOperationFeature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.le, value=value, **kwargs)


class GreaterThan(ArithmeticOperationFeature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.gt, value=value, **kwargs)


class GreaterThanOrEquals(ArithmeticOperationFeature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.ge, value=value, **kwargs)


class Equals(ArithmeticOperationFeature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value: PropertyLike[float] = 0, **kwargs):
        super().__init__(operator.eq, value=value, **kwargs)


class Stack(Feature):
    """Stacks the input and the value.

    If B is a feature then Stack can be visualized as::

       A >> Stack(B) = [*A(), *B()]

    If either A or B create a single Image, an additional dimension is automatically added.

    This can be

    Parameters
    ----------
    value
       Feature that produces image to stack on input.
    """

    __distributed__ = False

    def __init__(self, value=PropertyLike[Any], **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):

        if not isinstance(image, list):
            image = [image]

        if not isinstance(value, list):
            value = [value]

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

    Each time the feature is evaluated, it receives the output of the previous iteration. Each iteration
    also has its own set of properties. The index of the iteration is available as `_ID` or replicate_index.

    Parameters
    ----------
    feature : Feature
        Feature to repeat
    count : int
        Number of times to repeat
    """

    __distributed__ = False

    def __init__(self, feature, N, **kwargs):
        super().__init__(N=N, **kwargs)
        self.feature = self.add_feature(feature)

    def get(self, image, N, _ID=(), **kwargs):
        for n in range(N):

            index = _ID + (n,)

            image = self.feature(
                image,
                _ID=index,
                replicate_index=index,  # Pass replicate_index for legacy reasons
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


# class Dataset(Feature):
#     """Grabs data from a local set of data.

#     The first argument should be an iterator, function or constant,
#     which provides access to a single sample from a dataset. If it returns
#     a tuple, the first element should be an array-like and the second a
#     dictionary. The array-like will be returned as an image with the dictionary
#     added to the set of properties.

#     Parameters
#     ----------
#     data : tuple or array_like
#         Any property that returns a single image or a tuple of two objects,
#         where the first is an array_like.
#     """

#     __distributed__ = False

#     def __init__(
#         self, data: Iterator or PropertyLike[float or ArrayLike[float]], **kwargs
#     ):
#         super().__init__(data=data, **kwargs)

#     def get(self, *ignore, data, **kwargs):
#         return data

#     def _process_properties(self, properties):
#         properties = super()._process_properties(properties)

#         data = properties["data"]

#         if isinstance(data, tuple):
#             properties["data"] = data[0]
#             if isinstance(data[1], dict):
#                 properties.update(data[1])
#             else:
#                 properties["label"] = data[1]
#         return properties


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


class DummyFeature(Feature):
    """Feature that does nothing

    Can be used as a container for properties to separate the code logically.
    """

    def get(self, image, **kwargs):
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


class TensorflowDataset(Feature):
    """Loads a tensorflow dataset. Requires tensorflow_datasets to be installed.

    This feature loads a tensorflow dataset from its name. Check the
    `tensorflow datasets <https://www.tensorflow.org/datasets/catalog/overview>`_
    for a list of available datasets.

    This feature will download the dataset if it is not already present. Each key of
    the dataset will be added as a property to the feature. As such, separate pipelines
    can be created for each key::

        dataset = dt.TensorflowDataset("mnist")
        image_pipeline = dataset.image
        label_pipeline = dataset.label

    Alternatively, they can be loaded in conjunction::

        dataset = dt.TensorflowDataset("mnist", keys=["image", "label"])
        image, label = dataset()

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load
    split : str
        The split of the dataset to load. Defaults to "train".
        See `tensorflow splits <https://www.tensorflow.org/datasets/splits>`_ for more information on splits.
    shuffle_files : bool
        Whether to shuffle the files. Defaults to True.
    keys : list of str
        The keys to load from the dataset. Only used when calling the feature directly.
        Any key can be accessed as a property of the feature.

    Examples
    --------
    >>> dataset = dt.TensorflowDataset("mnist", split="train")
    >>> image_pipeline = dataset.image
    >>> label_pipeline = dataset.label
    """

    __distributed__ = False

    def __init__(
        self,
        dataset_name: str,
        split="train",
        shuffle_files=True,
        keys=["image", "label"],
        **kwargs
    ):

        self.tfds = None
        try:
            import tensorflow_datasets as tfds

            self.tfds = tfds
        except ImportError:
            raise ImportError(
                "Tensorflow Datasets is not installed. Install it with `pip install tensorflow_datasets`"
            )
        
        try:
            import deeptrack.datasets as dtd
            dtd.register_tfds()
        except:
            pass

        

        dataset = tfds.load(dataset_name, split=split, shuffle_files=shuffle_files)
        dataset_size = tfds.builder(dataset_name).info.splits[split].num_examples

        self.dataset = dataset
        self.split = split
        self.shuffle_files = shuffle_files
        self.size = dataset_size

        # get the keys of the dataset
        keys = list(dataset.element_spec.keys())
        attr_getters = {key: lambda output, key=key: output[key] for key in keys}

        self.dataset_iterator = iter(tfds.as_numpy(self.dataset))

        super().__init__(
            output=self.get_next_output, keys=keys, **attr_getters, **kwargs
        )

    def take(self, n):
        """Takes the n next elements of the dataset. Returns a dictionary of lists."""

        # Prepare output
        keys = self.dataset.element_spec.keys()
        output_dict = {key: [] for key in keys}

        for data in self.dataset.take(n):
            for key in keys:
                output_dict[key].append(data[key])

        return output_dict

    def reset_dataset(self):
        """Resets the dataset iterator to the beginning of the dataset."""
        self.dataset_iterator = iter(self.tfds.as_numpy(self.dataset))

    def get_next_output(self):
        try:
            return next(self.dataset_iterator)
        except StopIteration:
            self.reset_dataset()
            return next(self.dataset_iterator)

    def get(self, _, keys, output, **kwargs):
        return [output[key] for key in keys]


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



# Alias
Dataset = TensorflowDataset


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
    """Squeezes the input image to the smallest possible dimension.

    Parameters
    ----------
    axis : int or tuple of ints
        The axis to squeeze. Defaults to None, which squeezes all axes.
    """

    def __init__(self, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)

    def get(self, image, axis, **kwargs):
        return np.squeeze(image, axis=axis)
    
class Unsqueeze(Feature):
    """Unsqueezes the input image to the smallest possible dimension.

    Parameters
    ----------
    axis : int or tuple of ints
        The axis to unsqueeze. Defaults to None, which unsqueezes all axes.
    """

    def __init__(self, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)

    def get(self, image, axis, **kwargs):
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