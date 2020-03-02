''' Utility functions

Defines a set of utility functions used throughout the code
to make it more readable.

Functions
---------
hasfunction(obj: any, function_name: str) -> bool
    Return True if the object has a field named function_name
    that is callable. Otherwise, return False.
isiterable(obj: any)
    Return True if the object is iterable. Else, return False.
as_list(obj: any)
    If the input is iterable, convert it to list. 
    Otherwise, wrap the input in a list.
get_kwarg_names(function: Callable)
    Return the names of the keyword arguments the function accepts.

'''

import inspect

from typing import Callable, List



def hasmethod(obj: any, method_name: str) -> bool:
    ''' Check if an object has a callable method named method_name.

    Parameters
    ----------
    obj
        The object to be checked.
    method_name
        The name of the method to look for.

    Returns
    -------
    bool
        True if the object has an attribute method_name, and that
        attribute is callable.

    '''
    
    return hasattr(obj, method_name) and callable(getattr(obj, method_name, None))


def isiterable(obj: any) -> bool:
    ''' Check if the input is iterable.
    Note that this function does not capture all possible cases
    and is subject to change in the future if issues arise.

    Parameters
    ----------
    obj
        The object to check.

    Returns
    -------
    bool
        True if the object has __next__ defined. 

    '''
    
    return hasattr(obj, "__next__")


def as_list(obj: any) -> list:
    ''' Ensure the input is a list.
    If the input is iterable, convert it to a list, 
    otherwise wrap the input in a list.

    Parameters
    ----------
    obj
        The object that will be made a list.

    Returns
    -------
    list
        The input as a list.

    '''

    try:
        return list(obj)
    except TypeError:
        return [obj]


def get_kwarg_names(function: Callable) -> List[str]:
    ''' Retrieve the names of the keyword arguments.
    Retrieve the names of the keyword arguments accepted by `function`
    as a list of strings.

    Parameters
    ----------
    function
        The function to retrieve keyword argument names from.

    Returns
    -------
    List[str]
        The accepted keyword arguments as a list of strings.

    '''

    try:
        argspec = inspect.getfullargspec(function)
    except TypeError:
        return []

    kwargs = []

    if argspec.args and argspec.defaults:
        kwargs = argspec.args[-len(argspec.defaults):]

    if argspec.kwonlyargs:
        kwargs = kwargs + argspec.kwonlyargs

    return  kwargs

