"""Base class Feature and structural features

Provides classes and tools for creating and interacting with features.

Classes
-------
Feature
    Base abstract class.
StructuralFeature
    Abstract extension of feature for interactions between features.
Branch
    Implementation of `StructuralFeature` that resolves two features 
    sequentially.
Probability
    Implementation of `StructuralFeature` that randomly resolves a feature 
    with a certain probability.
Duplicate
    Implementation of `StructuralFeature` that sequentially resolves an 
    integer number of deep-copies of a feature.

"""

import copy

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import time
import threading

from deeptrack.image import Image
from deeptrack.properties import Property, PropertyDict
from deeptrack.utils import isiterable, hasmethod, get_kwarg_names, kwarg_has_default


MERGE_STRATEGY_OVERRIDE = 0
MERGE_STRATEGY_APPEND = 1

# Global thread execution lock for update calls.
UPDATE_LOCK = threading.Lock()

# Global update memoization. Ensures that the update is consistent
UPDATE_MEMO = {"user_arguments": {}, "memoization": {}}


class Feature:
    """Base feature class.
    Features define the image generation process. All features operate
    on lists of images. Most features, such as noise, apply some
    tranformation to all images in the list. This transformation can
    be additive, such as adding some Gaussian noise or a background
    illumination, or non-additive, such as introducing Poisson noise
    or performing a low-pass filter. This transformation is defined
    by the method `get(image, **kwargs)`, which all implementations of
    the class `Feature` need to define.

    Whenever a Feature is initiated, all keyword arguments passed to the
    constructor will be wrapped as a Property, and stored in the
    `properties` field as a `PropertyDict`. When a Feature is resolved,
    the current value of each property is sent as input to the get method.

    Parameters
    ----------
    *args : dict, optional
        Dicts passed as nonkeyword arguments will be deconstructed to key-value
        pairs and included in the field `properties` in the same way as keyword
        arguments.
    **kwargs
        All Keyword arguments will be wrapped as instances of ``Property`` and
        included in the field `properties`.


    Attributes
    ----------
    properties : dict
        A dict that contains all keyword arguments passed to the
        constructor wrapped as Distributions. A sampled copy of this
        dict is sent as input to the get function, and is appended
        to the properties field of the output image.
    __list_merge_strategy__ : int
        Controls how the output of `.get(image, **kwargs)` is merged with
        the input list. It can be `MERGE_STRATEGY_OVERRIDE` (0, default),
        where the input is replaced by the new list, or
        `MERGE_STRATEGY_APPEND` (1), where the new list is appended to the
        end of the input list.
    __distributed__ : bool
        Controls whether `.get(image, **kwargs)` is called on each element
        in the list separately (`__distributed__ = True`), or if it is
        called on the list as a whole (`__distributed__ = False`).
    __property_memorability__
        Controls whether to store the features properties to the `Image`.
        Values 1 or lower will be included by default.
    """

    __list_merge_strategy__ = MERGE_STRATEGY_OVERRIDE
    __distributed__ = True
    __property_memorability__ = 1

    def __init__(self, *args: dict, **kwargs):
        super(Feature, self).__init__()
        properties = getattr(self, "properties", {})

        # Create an iterable of kwargs and args
        all_dicts = (kwargs,) + args

        for property_dict in all_dicts:
            for key, value in property_dict.items():
                if not isinstance(value, Property):

                    value = Property(value)

                properties[key] = value

        # hash_key is an inexpensive way to compare dicts of properties
        # The hash here is 4 31 bit integers, for a total of 124 bits.
        if "hash_key" not in properties:
            properties["hash_key"] = Property(
                lambda: list(np.random.randint(2 ** 31, size=(4,)))
            )

        self.properties = PropertyDict(**properties)

    def get(self, image: Image or List[Image], **kwargs) -> Image or List[Image]:
        """Method for altering an image
        Abstract method that define how the feature transforms the input. The current
        value of all properties will be passed as keyword arguments.

        Parameters
        ---------
        image : Image or List[Image]
            The Image or list of images to transform
        **kwargs
            The current value of all properties in `properties` as well as any global
            arguments.

        Returns
        -------
        Image or List[Image]
            The transformed image or list of images
        """

    def resolve(self, image_list: Image or List[Image] = None, **global_kwargs):
        """Creates the image.
        Transforms the input image by calling the method `get()` with the
        correct inputs. The properties of the feature can be overruled by
        passing a different value as a keyword argument.

        Parameters
        ----------
        image_list : Image or List[Image], optional
            The Image or list of images to be transformed.
        **global_kwargs
            Set of arguments that are applied globally. That is, every
            feature in the set of features required to resolve an image
            will receive these keyword arguments.

        Returns
        -------
        Image or List[Image]
            The resolved image
        """

        # Remove hash_key from globals.
        global_kwargs.pop("hash_key", False)

        # Ensure that input is a list
        image_list = self._format_input(image_list, **global_kwargs)

        # Get the input arguments to the method .get()
        feature_input = self.properties.current_value_dict(
            is_resolving=True, **global_kwargs
        )

        # Add global_kwargs to input arguments
        feature_input.update(global_kwargs)

        # Call the _process_properties hook, default does nothing.
        # Can be used to ensure properties are formatted correctly
        # or to rescale properties.
        feature_input = self._process_properties(feature_input)

        # Set the seed from the hash_key. Ensures equal results
        np.random.seed(
            int(feature_input["hash_key"][0])
            * int(feature_input.get("sequence_step", 0) + 1)
            % int(2 ** 32 - 1)
        )

        # _process_and_get calls the get function correctly according
        # to the __distributed__ attribute
        new_list = self._process_and_get(image_list, **feature_input)

        # If tuple, assume return additional properties
        if isinstance(new_list, tuple):
            feature_input = {**feature_input, **new_list[1]}
            new_list = new_list[0]

        # Add feature_input to the image the class attribute __property_memorability__
        # is not larger than the passed property_verbosity keyword
        property_verbosity = global_kwargs.get("property_memorability", 1)
        feature_input["name"] = type(self).__name__
        if self.__property_memorability__ <= property_verbosity:
            for image in new_list:
                image.append(feature_input)

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

    def update(self, **kwargs) -> "Feature":
        """Updates the state of all properties.

        Parameters
        ----------
        **kwargs
            Arguments that will be set globally for the update call.
            For example .update(value=10) will have all properties
            and features that depend on "value" use 10 instead of what
            they otherwise would.

        Returns
        -------
        self
        """

        # This should only be accessed by the user. Call _update directly instead
        with UPDATE_LOCK:
            UPDATE_MEMO["user_arguments"] = kwargs
            UPDATE_MEMO["memoization"] = {}
            self._update(**kwargs)
            return self

    def _update(self, **kwargs):
        self.properties.update(**kwargs)

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

        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from IPython.display import HTML, display

        if input_image is not None:
            input_image = [Image(input_image)]

        output_image = self.resolve(input_image, **(resolve_kwargs or {}))

        # If a list, assume video
        if isinstance(output_image, Image):
            # Single image
            plt.imshow(output_image[:, :, 0], **kwargs)
            plt.show()

        else:
            # Assume video
            fig = plt.figure()
            images = []
            plt.axis("off")
            for image in output_image:
                images.append([plt.imshow(image[:, :, 0], **kwargs)])

            interval = (
                interval or output_image[0].get_property("interval") or (1 / 30 * 1000)
            )

            anim = animation.ArtistAnimation(
                fig, images, interval=interval, blit=True, repeat_delay=0
            )

            try:
                get_ipython  # Throws NameError if not in Notebook
                display(HTML(anim.to_jshtml()))
                return anim

            except NameError as e:
                # Not in an notebook
                plt.show()

            except RuntimeError as e:
                # In notebook, but animation failed
                import ipywidgets as widgets

                Warning(
                    "Javascript animation failed. This is a non-performant fallback."
                )

                def plotter(frame=0):
                    plt.imshow(output_image[frame][:, :, 0], **kwargs)
                    plt.show()

                return widgets.interact(
                    plotter,
                    frame=widgets.IntSlider(
                        value=0, min=0, max=len(images) - 1, step=1
                    ),
                )

    def _process_and_get(self, image_list, **feature_input) -> List[Image]:
        # Controls how the get function is called

        if self.__distributed__:
            # Call get on each image in list, and merge properties from corresponding image
            return [
                Image(self.get(image, **feature_input)).merge_properties_from(image)
                for image in image_list
            ]
        else:
            # Call get on entire list.
            new_list = self.get(image_list, **feature_input)

            if not isinstance(new_list, list):
                new_list = [Image(new_list)]

            new_list = [Image(image) for image in new_list]
            return new_list

    def _format_input(self, image_list, **kwargs) -> List[Image]:
        # Ensures the input is a list of Image.

        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]

        return [Image(image) for image in image_list]

    def _process_properties(self, propertydict) -> dict:
        # Optional hook for subclasses to preprocess input before calling
        # the method .get()
        return propertydict

    def sample(self, **kwargs) -> "Feature":
        """Returns the feature"""

        return self

    def __getattr__(self, key):
        # Allows easier access to properties, while guaranteeing they are updated correctly.
        # Should only every be used from the inside of a property function.
        # Is not compatible with sequential properties.
        if "properties" in self.__dict__:
            properties = self.__dict__["properties"]

            if key in properties:
                return properties[key]
            else:
                raise AttributeError
        else:
            raise AttributeError

    def __add__(self, other: "Feature") -> "Feature":
        # Overrides add operator
        if isinstance(other, list) and all(isinstance(f) for f in other):
            other = Combine(features=other)

        if isinstance(other, Feature):
            return Branch(self, other)

    def __radd__(self, other) -> "Feature":
        # Add when left hand is not a feature
        # If left hand is falesly, return self
        # This allows operations such as sum(list_of_features)
        if isinstance(other, list) and all(isinstance(f) for f in other):
            other = Combine(features=other)

        if isinstance(other, Feature):
            return Branch(other, self)
        elif not other:
            return self
        else:
            return NotImplemented

    def __mul__(self, other: float) -> "Feature":
        # Introduces a probablity of a feature to be resolved.
        if isinstance(other, list) and all(isinstance(f) for f in other):
            other = Combine(features=other)

        return Probability(self, other)

    __rmul__ = __mul__

    def __pow__(self, other) -> "Feature":
        # Duplicate the feature to resolve more items
        if isinstance(other, list) and all(isinstance(f) for f in other):
            other = Combine(features=other)

        return Duplicate(self, other)


class StructuralFeature(Feature):
    """Provides the structure of a feature-set
    Feature with __property_verbosity__ = 2 to avoid adding it to the list
    of properties, and __distributed__ = False to pass the input as-is.
    """

    __property_verbosity__ = 2
    __distributed__ = False


class Branch(StructuralFeature):
    """Resolves two features sequentially.
    Passes the output of the first to the input of the second.
    Parameters
    ----------
    feature_1 : Feature
    feature_2 : Feature
    """

    def __init__(self, feature_1: Feature, feature_2: Feature, *args, **kwargs):
        super().__init__(*args, feature_1=feature_1, feature_2=feature_2, **kwargs)

    def get(self, image, feature_1, feature_2, **kwargs):
        """Resolves `feature_1` and `feature_2` sequentially"""
        image = feature_1.resolve(image, **kwargs)
        image = feature_2.resolve(image, **kwargs)
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

    def __init__(self, feature: Feature, probability: float, *args, **kwargs):
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


class Duplicate(StructuralFeature):
    """Resolves copies of a feature sequentially
    Creates `num_duplicates` copies of the feature and resolves
    them sequentially

    Parameters
    ----------
    feature: Feature
        The feature to duplicate
    num_duplicates: int
        The number of duplicates to create
    """

    def __init__(self, feature: Feature, num_duplicates: int, *args, **kwargs):

        self.feature = feature
        super().__init__(
            *args,
            num_duplicates=num_duplicates,  # py > 3.6 dicts are ordered by insert time.
            features=lambda num_duplicates: [
                copy.deepcopy(self.feature) for _ in range(num_duplicates)
            ],
            **kwargs
        )

    def get(self, image, features: List[Feature], **kwargs):
        """Resolves each feature in `features` sequentially"""
        for index in range(len(features)):
            image = features[index].resolve(image, **kwargs)

        return image

    def _update(self, **kwargs):

        super()._update(**kwargs)
        features = self.properties["features"].current_value
        for index in range(len(features)):
            features[index]._update(**kwargs)
        return self

    def __deepcopy__(self, memo):
        # If this is getting deep-copied, we have
        # nested copies, which needs to be handled
        # separately.

        self.properties.update()
        num_duplicates = self.num_duplicates.current_value
        features = []
        for idx in range(num_duplicates):
            memo_copy = copy.copy(memo)
            new_feature = copy.deepcopy(self.feature, memo_copy)
            features.append(new_feature)
        self.properties["features"].current_value = features

        out = copy.copy(self)
        self.properties = copy.copy(self.properties)
        for key, val in self.properties.items():
            self.properties[key] = copy.copy(val)
        return out


class Combine(StructuralFeature):
    """Combines multiple features into a single feature.

    Resolves each feature in `features` and returns them as a list of features.

    Parameters
    ----------
    features : list of features
        features to combine

    """

    __distribute__ = False

    def __init__(self, features=[], **kwargs):
        super().__init__(features=features, **kwargs)

    def get(self, image_list, features, **kwargs):
        return [feature.resolve(image_list, **kwargs) for feature in features]


class ConditionalSetProperty(StructuralFeature):
    """Conditionally overrides the properties of child features

    Parameters
    ----------
    feature : Feature
        The child feature
    condition : str
        The name of the conditional property
    **kwargs
        Properties to be used if `condition` is True

    """

    __distributed__ = False

    def __init__(self, feature: Feature, condition="is_label", **kwargs):
        super().__init__(feature=feature, condition=condition, **kwargs)

    def get(self, image, feature, condition, **kwargs):
        if kwargs.get(condition, False):
            return feature.resolve(image, **kwargs)
        else:
            for property_key in self.properties.keys():
                kwargs.pop(property_key, None)

            return feature.resolve(image, **kwargs)


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
        condition="is_label",
        **kwargs
    ):

        super().__init__(
            on_true=on_true, on_false=on_false, condition=condition, **kwargs
        )

    def get(self, image, *, condition, on_true, on_false, **kwargs):

        if kwargs.get(condition, False):
            if on_true:
                return on_true.resolve(image, **kwargs)
            else:
                return image
        else:
            if on_false:
                return on_false.resolve(image, **kwargs)
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

    def __init__(self, function, **kwargs):
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

    def __init__(self, function, **kwargs):
        super().__init__(function=function, **kwargs)

    def get(self, list_of_images, function, **kwargs):
        return function(list_of_images)


class Dataset(Feature):
    """Grabs data from a local set of data.

    The first argument should be an iterator, function or constant,
    which provides access to a single sample from a dataset. If it returns
    a tuple, the first element should be an array-like and the second a
    dictionary. The array-like will be returned as an image with the dictionary
    added to the set of properties.

    Parameters
    ----------
    data : tuple or array_like
        Any property that returns a single image or a tuple of two objects,
        where the first is an array_like.
    """

    __distributed__ = False

    def __init__(self, data, **kwargs):
        super().__init__(data=data, **kwargs)

    def get(self, *ignore, data, **kwargs):
        return data

    def _process_properties(self, properties):
        data = properties["data"]

        if isinstance(data, tuple):
            properties["data"] = data[0]
            if isinstance(data[1], dict):
                properties.update(data[1])
            else:
                properties["label"] = data[1]
        return properties


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

    def __init__(self, output_shape=None, **kwargs):
        super().__init__(output_shape=output_shape, **kwargs)

    def get(self, image, output_shape=None, hash_key=None, **kwargs):
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
        path,
        load_options=None,
        as_list=False,
        ndim=None,
        to_grayscale=False,
        get_one_random=False,
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
        if load_options is None:
            load_options = {}
        try:
            image = np.load(path, **load_options)
        except (IOError, ValueError):
            try:
                from skimage import io

                image = io.imread(path)
            except (IOError, ImportError, AttributeError):
                try:
                    import PIL.Image

                    omage = np.array(PIL.Image.open(path, **load_options))
                except (IOError, ImportError):
                    import cv2

                    image = np.array(cv2.imread(path, **load_options))
                    if not image:
                        raise IOError(
                            "No filereader available for file {0}".format(path)
                        )

        image = np.squeeze(image)

        if to_grayscale:
            try:
                import skimage

                skimage.color.rgb2gray(image)
            except ValueError:
                import warnings

                warnings.warn("Non-rgb image, ignoring to_grayscale")

        if ndim and image.ndim < ndim:
            image = np.expand_dims(image, axis=-1)

        elif as_list:
            if get_one_random:
                image = image[np.random.randint(len(image))]
            else:
                image = list(image)

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
        transformation_function,
        number_of_masks=1,
        output_region=None,
        merge_method="add",
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
            (output_region[2], output_region[3], kwargs["number_of_masks"])
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

                p0 = p0.astype(np.int)

                output_slice = output[
                    p0[0] : p0[0] + labelarg.shape[0], p0[1] : p0[1] + labelarg.shape[1]
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
                            output_slice[..., label_index], labelarg[..., label_index]
                        )
        output = Image(output)
        for label in list_of_labels:
            output.merge_properties_from(label)
        return output


def _get_position(image, mode="corner", return_z=False):
    # Extracts the position of the upper left corner of a scatterer

    if mode == "corner":
        shift = (np.array(image.shape) - 1) / 2
    else:
        shift = np.zeros((num_outputs))

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
                        [position[0], position[1], image.get_property("z", default=0)]
                    )
                    - shift
                )
                positions_out.append(outp)
            else:
                positions_out.append(position - shift[0:2])

    return positions_out
