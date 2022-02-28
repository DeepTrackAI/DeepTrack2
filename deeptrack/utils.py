""" Utility functions

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
"""

import inspect

from typing import Callable, List


def hasmethod(obj: any, method_name: str) -> bool:
    """Check if an object has a callable method named method_name.

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

    """

    return hasattr(obj, method_name) and callable(getattr(obj, method_name, None))


def isiterable(obj: any) -> bool:
    """Check if the input is iterable.
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

    """

    return hasattr(obj, "__next__")


def as_list(obj: any) -> list:
    """Ensure the input is a list.
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

    """

    try:
        return list(obj)
    except TypeError:
        return [obj]


def get_kwarg_names(function: Callable) -> List[str]:
    """Retrieve the names of the keyword arguments.
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

    """

    try:
        argspec = inspect.getfullargspec(function)
    except TypeError:
        return []

    if argspec.varargs:
        return argspec.kwonlyargs or []
    else:
        return argspec.args or []


def kwarg_has_default(function: Callable, argument: str) -> bool:
    """Returns true if an argument has a default value.

    Parameters
    ----------
    function : Callable
        The function to check.
    argument : str
        Name of the argument

    Returns
    -------
    bool

    """
    args = get_kwarg_names(function)

    if argument not in args:
        return False

    defaults = inspect.getfullargspec(function).defaults or ()

    return len(args) - args.index(argument) <= len(defaults)


def safe_call(function, positional_args=[], **kwargs):
    """Calls a function, using keyword arguments from a dictionary of arguments.

    If the function does not accept one of the argument provided, it will not
    be passed. Does not support non-keyword arguments.

    Parameters
    ----------
    function : Callable
        The function to call
    kwargs
        Key-value pairs to draw input arguments from.
    """

    keys = get_kwarg_names(function)

    input_arguments = {}
    for key in keys:
        if key in kwargs:
            input_arguments[key] = kwargs[key]

    return function(*positional_args, **input_arguments)
