""" Image class and relative functions

Defines the Image class and functions that operate on it.

CLASSES
---------
Image
    Subclass of numpy `ndarray`. Has the additional attribute properties
    which contains the properties used to generate it as a `list` of
    `dicts`.

Functions
---------
pad_image_to_fft(image: Image, axes = (0, 1))
    Pads the image with zeros to optimize the speed of Fast Fourier
    Transforms.
"""

import numpy as np


class Image(np.ndarray):
    """Subclass of numpy ndarray

    The class Image is used by features to resolve images and store
    the current values of the properties of each feature in the feature
    series. These properties are stored in the field `properties`
    as a list of dictionaries, in the same order as that in which
    the features have been evaluated.

    The field `properties` is used to store and extract information
    about how an image has been generated.

    Parameters
    ----------
    input_array : array_like
        An array_like object that is used to instantiate the ndarray.
    properties : list of dicts, optional
        Optional parameter to set as the initial value for the field properties.

    Attributes
    ----------
    properties : list
        List of dictionaries of the current value of all properties of
        the features used to resolve the image.

    """

    # Used by numpy to determine output type of u_funcs.
    # This ensures that the output will always be an Image
    __array_priority__ = 999

    def __new__(cls, input_array, properties=None):
        # Converts input to ndarray, and then to an Image
        # In particular, it creates the properties

        image = np.array(input_array).view(cls)
        if properties is None:
            # If input_array has properties attribute, retrieve a copy of it
            properties = getattr(input_array, "properties", [])[:]
        image.properties = properties

        return image

    def append(self, property_dict: dict):
        """Appends a dictionary to the properties list.

        Parameters
        ----------
        property_dict : dict
            A dictionary to append to the property list. Most commonly
            the current values of the properties of a feature.

        Returns
        -------
        Image
            Returns itself.
        """

        self.properties.append(property_dict)
        return self

    def get_property(
        self, key: str, get_one: bool = True, default: any = None
    ) -> list or any:
        """Retrieve a property.

        If the feature has the property defined by `key`, return
        its current_value. Otherwise, return `default`.
        If `get_one` is True, the first instance is returned;
        otherwise, all instances are returned as a list.

        Parameters
        ----------
        key
            The name of the property.
        get_one: optional
            Whether to return all instances of that property or just the first.
        default : optional
            What is returned if the property is not found.

        Returns
        -------
        any
            The value of the property if found, else `default`.

        """

        if get_one:
            for prop in self.properties:
                if key in prop:
                    return prop[key]
            return default
        else:
            return [prop[key] for prop in self.properties if key in prop] or default

    def merge_properties_from(self, other: "Image") -> "Image":
        """Merge properties with those from another Image.

        Appends properties from another images such that no property is duplicated.
        Uniqueness of a dictionary of properties is determined from the
        property `hash_key`.

        Most functions involving two images should automatically output an image with
        merged properties. However, since each property is guaranteed to be unique,
        it is safe to manually call this function if there is any uncertainty.

        Parameters
        ----------
        other
            The Image to retrieve properties from.

        """

        for new_prop in other.properties:

            # If no hash_key, add it
            if "hash_key" not in new_prop:
                self.append(new_prop)
                continue

            should_append = True
            # Else, see if hash is unique
            for my_prop in self.properties:

                if (
                    "hash_key" in my_prop
                    and my_prop["hash_key"] == new_prop["hash_key"]
                ):

                    # Key is not unique, don't add
                    should_append = False
                    break

            if should_append:
                self.append(new_prop)

        return self

    def __array_wrap__(self, image_after_function, context=None):
        # Called at end when a function is called on an image
        # It might be that the information about properties is lost,
        # this method restores it.
        # This method also correctly concatenate the the properties of two images.

        if image_after_function is self:  # for in-place operations
            image_with_restored_properties = image_after_function
        else:
            image_with_restored_properties = Image(image_after_function)

        if context is not None:
            # context is information about operation

            func, args, _ = context
            input_args = args[: func.nin]

            for arg in input_args:

                if not arg is self and isinstance(arg, Image):
                    self.merge_properties_from(arg)

        return image_with_restored_properties

    def __array_finalize__(self, image):
        # Called when an image is created
        # It might be that the information about properties is lost,
        # this method restores it.

        if image is None:
            return

        # Ensure self has properties defined
        self.properties = getattr(self, "properties", [])

        # Merge from image if image is Image
        if isinstance(image, Image):
            self.merge_properties_from(image)


FASTEST_SIZES = [0]
for n in range(1, 10):
    FASTEST_SIZES += [2 ** a * 3 ** (n - a - 1) for a in range(n)]
FASTEST_SIZES = np.sort(FASTEST_SIZES)


def pad_image_to_fft(image: Image, axes=(0, 1)) -> Image:
    """Pads image to speed up fast fourier transforms.
    Pads image to speed up fast fourier transforms by adding 0s to the
    end of the image.

    Parameters
    ----------
    image
        The image to pad
    axes : iterable of int, optional
        The axes along which to pad.
    """

    def _closest(dim):
        # Returns the smallest value frin FASTEST_SIZES
        # larger than dim
        for size in FASTEST_SIZES:
            if size >= dim:
                return size

    new_shape = np.array(image.shape)
    for axis in axes:
        new_shape[axis] = _closest(new_shape[axis])

    increase = np.array(new_shape) - image.shape
    pad_width = [(0, inc) for inc in increase]

    return np.pad(image, pad_width, mode="constant")
