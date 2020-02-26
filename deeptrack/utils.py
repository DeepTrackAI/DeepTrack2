''' Utility functions

Defines a set of utility functions used throughout the code
to make it more readable.

Functions
---------
hasfunction(object:any, function_name:str) -> bool
    returns True if the object has a field named function_name
    that is callable. Otherwise it returns False.
isiterable(object:any)
    returns True if the object is iterable, else, return False.
'''
import inspect
import numpy as np
from deeptrack.image import Image


def hasmethod(obj: any, method_name: str) -> bool:
    ''' Checks if the input has a callable method named method_name
    Parameters
    ----------
    object
        The object upon which to check.
    functmethod_nameion_name : str
        the name of the method

    Returns
    -------
    bool
        True if the object has an attribute method_name, and that
        attribute is callable.
    '''
    return hasattr(obj, method_name) and callable(getattr(obj, method_name, None))


def isiterable(obj: any) -> bool:
    ''' Checks if the input is iterable
    Parameters
    ----------
    object
        The object upon which to check.

    Returns
    -------
    bool
        True if the object either has a __iter__ field or a __getitem__
        field.
    '''
    return hasattr(obj, "__next__")


def as_list(obj): 
    try:
        return list(obj)
    except TypeError:
        return [obj]


def get_property(feature, key, default=None):
    for property in feature.properties:
        if key in property:
            return property[key]
    return default


def get_kwarg_names(function):
    try:
        argspec = inspect.getfullargspec(function)
    except TypeError:
        return []

    if (not argspec.args) or (not argspec.defaults):
        return []

    return argspec.args[-len(argspec.defaults):]


def only_keyword_arguments(function, ignore_self=False):
    argspec = inspect.getfullargspec(function)
    return argspec.args



class cached_function:

    def __init__(self, func):
        assert only_keyword_arguments(func), "Cached function may only receive keyword arguments"

        self.previous_inputs = None
        self.previous_output = None
        self.kwarg_names = get_kwarg_names(func)
        self.function = func

    def __call__(self, **kwargs):

        safe_input = {}
        for key in self.kwarg_names:
            if key in kwarg:
                safe_input[key] = kwargs[key]

        kwargs = safe_input
        
        if self.previous_inputs is None:
            new_value = self.function(self, **kwargs)

        elif self.previous_inputs == kwargs:
            new_value = self.previous_output
        
        else: 
            new_value = self.function(self, **kwargs)
        
        self.previous_inputs = kwargs
        self.previous_output =  new_value

        return new_value


# TODO: More exact
FASTEST_SIZES = [0]
for n in range(1, 10):
    FASTEST_SIZES += [2**a * 3**(n - a - 1) for a in range(n)]
FASTEST_SIZES = np.sort(FASTEST_SIZES)

def closest(dim):
    for size in FASTEST_SIZES:
        if size >= dim:
            return size

def pad_image_to_fft(image, axes=(0, 1)):

    new_shape = np.array(image.shape)
    for axis in axes:
        new_shape[axis] = closest(new_shape[axis])
        
    increase =  np.array(new_shape) - image.shape
    pad_width = [(0, inc) for inc in increase]


    return np.pad(image, pad_width, mode='constant')

import scipy.ndimage as ndimage
def track_segmented_image(image,
                          erosion=0,
                          dilation=0,
                          threshold=0.95,
                          min_area=0):

    binary_image = image > threshold

    if dilation:
        binary_image = ndimage.binary_dilation(binary_image, iterations=dilation) 

    if erosion:
        binary_image = ndimage.binary_erosion(binary_image, iterations=erosion) 

    components, _ = ndimage.measurements.label(binary_image)
    
    area = ndimage.measurements.sum(binary_image, labels=components)
    position = ndimage.measurements.center_of_mass(binary_image, labels=components)

    return position[area > min_area]
