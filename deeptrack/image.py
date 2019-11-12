import numpy as np


class Image(np.ndarray):
    '''Subclass of numpy ndarray with additional attribute "properties"

    The Image subclass of numpy ndarrays are used by features to resolve
    images and store the current value of the properties of each feature
    in the feature series. These properties are stored in the field
    `properties` as a list of dictionaries, in the same order that the
    features were evaluated.

    This is used thoughout the deeptrack model to store and extract
    information about how an image was generated. Examples include

    * Features may depend on earlier features in the feature series.

    * The class `Label`, which extract numeric information
    about the image to be sent to machine learning models.  

    * Loading previously generated images from storage without
    losing information about the image.

    Parameters
    ----------
    input_array : array_like
        An array_like object that is used to instantiate the ndarray
    properties : {None, list}
        Optional parameter to set as the initial value for the field properties

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
        ''' Appends a dictionary to the properties list

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


    def __new__(cls, input_array, properties=None):
        # Converts input to ndarray, and then to an Image
        obj = np.array(input_array).view(cls)
        if properties is None:
            # If input_array has properties attribute, retrieve a copy of it
            properties = getattr(input_array, "properties", [])[:]
        obj.properties = properties
        return obj


    def __array_wrap__(self, out_arr, context=None):

        if out_arr is self:  # for in-place operations
            result = out_arr
        else:
            result = Image(out_arr)

        if context is not None:
            func, args, _ = context
            input_args = args[:func.nin]

            for arg in input_args:
                if not arg is self:
                    props = getattr(arg, "properties", [])
                    for p in props:
                        result.append(p)
        return result


    def __array_finalize__(self, obj):

        if obj is None: return

        self.properties = getattr(self, "properties", [])


        props = getattr(obj, "properties", [])
        if not props is self.properties:
            for property in props:
                self.append(property)


    # def __reduce__(self):
    #     # Get the parent's __reduce__ tuple
    #     pickled_state = super(Image, self).__reduce__()
    #     # Create our own tuple to pass to __setstate__, appending properties
    #     new_state = pickled_state[2] + (self.properties,)
    #     # Return a tuple that replaces the parent's __setstate__ tuple with our own
    #     return (pickled_state[0], pickled_state[1], new_state)


    # def __setstate__(self, state):
    #     self.properties = state[-1]  # Set the peroperties attribute
    #     # Call the parent's __setstate__ with the other tuple elements.
    #     super(Image, self).__setstate__(state[0:-1])
