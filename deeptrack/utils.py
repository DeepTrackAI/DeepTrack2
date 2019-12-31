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

import inspect
def get_kwarg_names(function):
    argspec = inspect.getfullargspec(function)

    if (not argspec.args) or (not argspec.defaults):
        return []

    return argspec.args[-len(argspec.defaults):]

# TODO: More exact
fastest_sizes = [0]
for n in range(1, 10):
    fastest_sizes += [2**a * 3**(n - a - 1) for a in range(n)]
fastest_sizes = np.sort(fastest_sizes)

def closest(dim):
    for size in fastest_sizes:
        if size >= dim:
            return size

def pad_image_to_fft(image, axes=(0, 1)):

    new_shape = np.array(image.shape)
    for axis in axes:
        new_shape[axis] = closest(new_shape[axis])
        
    increase =  np.array(new_shape) - image.shape
    pad_width = [(0, inc) for inc in increase]


    return np.pad(image, pad_width, mode='constant')
