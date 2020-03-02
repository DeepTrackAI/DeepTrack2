import numpy as np


class Image(np.ndarray):
    '''Subclass of numpy ndarray with additional attribute "properties"

    The Image subclass of numpy ndarrays are used by features to resolve
    images and store the current value of the properties of each feature
    in the feature series. These properties are stored in the field
    `properties` as a list of dictionaries, in the same order as that
    in which the features were evaluated.

    This is used thoughout the deeptrack model to store and extract
    information about how an image was generated.

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

    '''

    # Used by numpy to determine output type of u_funcs.
    # This ensures that the output will always be an Image
    __array_priority__ = 999


    def append(self, property_dict: dict):
        ''' Appends a dictionary to the properties list.

        Parameters
        ----------
        property_dict : dict
            A dictionary to append to the property list. Most commonly
            the current values of the properties of a feature.

        Returns
        -------
        Image
            Returns itself
        '''
        
        self.properties.append(property_dict)
        return self


    def get_property(self,
                     key: str,
                     get_one: bool = True,
                     default: any = None) -> list or any:
        '''Retrieve a property.
        
        If the feature has the property defined by `key`, return
        its current_value. Otherwise, return `default`. If get_one
        is True, the first instance is returned, otherwise, all
        instances are returned as a list.

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

        '''

        if get_one:
            for prop in self.properties:
                if key in prop:
                    return prop[key]
            return default
        else:
            return [prop[key] for prop in self.properties if key in prop] or default
    

    
    def merge_properties_from(self, other: "Image") -> "Image":
        ''' Merges properties with ones from another Image.

        Appends properties from another images such that no property is duplicated.
        Uniqueness of a dictionary of properties is determined from the
        `hash_key` property.

        Most function involving two Images should automatically output an image with
        merged properties. However, since each property is guaranteed to be unique,
        it is safe to manually call this function if there's any uncertainty.

        Parameters
        ----------
        other
            The Image to retrieve properties from.

        '''

        for new_prop in other.properties:

            # If no hash_key, add it
            if "hash_key" not in new_prop:
                self.append(new_prop)
                continue

            should_append = True
            # Else, see if hash is unique
            for my_prop in self.properties:

                if ("hash_key" in my_prop and
                        my_prop["hash_key"] == new_prop["hash_key"]):

                    # Key is not unique, don't add
                    should_append = False
                    break

            if should_append:
                self.append(new_prop)

        return self

    def __new__(cls, input_array, properties=None):
        # Converts input to ndarray, and then to an Image
        obj = np.array(input_array).view(cls)
        if properties is None:
            # If input_array has properties attribute, retrieve a copy of it
            properties = getattr(input_array, "properties", [])[:]
        obj.properties = properties

        return obj


    def __array_wrap__(self, out_arr, context=None):
        # Called at end of function call

        if out_arr is self:  # for in-place operations
            result = out_arr
        else:
            result = Image(out_arr)

        if context is not None:
            # context is information about operation

            func, args, _ = context
            input_args = args[:func.nin]

            for arg in input_args:

                if not arg is self and isinstance(arg, Image):
                    self.merge_properties_from(arg)

        return result


    def __array_finalize__(self, obj):
        # Finalizes the array after a function

        if obj is None:
            return

        # Ensure self has properties defined
        self.properties = getattr(self, "properties", [])

        # Merge from obj if obj is Image
        if isinstance(obj, Image):
            self.merge_properties_from(Image)



FASTEST_SIZES = [0]
for n in range(1, 10):
    FASTEST_SIZES += [2**a * 3**(n - a - 1) for a in range(n)]
FASTEST_SIZES = np.sort(FASTEST_SIZES)


def _closest(dim):
    # Returns the smallest value frin FASTEST_SIZES
    # larger than dim
    for size in FASTEST_SIZES:
        if size >= dim:
            return size


def pad_image_to_fft(image: Image, axes=(0, 1)) -> Image:
    ''' Pads image to speed up fast fourier transforms.
    Pads image to speed up fast fourier transforms by adding 0s to the
    end of the image.

    Parameters
    ----------
    image
        The image to pad
    axes : iterable of int, optional
        The axes along which to pad.
    '''

    new_shape = np.array(image.shape)
    for axis in axes:
        new_shape[axis] = _closest(new_shape[axis])

    increase = np.array(new_shape) - image.shape
    pad_width = [(0, inc) for inc in increase]


    return np.pad(image, pad_width, mode='constant')