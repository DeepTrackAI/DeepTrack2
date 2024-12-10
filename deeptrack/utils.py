"""Utility functions.

This module defines utility functions that enhance code readability, 
streamline common operations, and ensure type and argument consistency.

Functions
---------
hasmethod(obj: any, method_name: str) -> bool
    Returns True if the object has a field named `function_name` that is 
    callable. Otherwise, returns False.
as_list(obj: any) -> list
    If the input is iterable, converts it to list. 
    Otherwise, wraps the input in a list.
get_kwarg_names(function: Callable) -> List[str]
    Retrieves the names of the keyword arguments accepted by a function.
kwarg_has_default(function: Callable, argument: str) -> bool
    Checks if a specific argument of a function has a default value.
safe_call(function, positional_args=[], **kwargs)
    Calls a function, passing only valid arguments from the provided keyword 
    arguments (kwargs).

"""

import inspect
from typing import Any, Callable, List


def hasmethod(obj: any, method_name: str) -> bool:
    """Check if an object has a callable method named `method_name`.

    Parameters
    ----------
    obj : any
        The object to inspect.
    method_name : str
        The name of the method to look for.

    Returns
    -------
    bool
        True if the object has an attribute named `method_name` that is 
        callable.

    """

    return (hasattr(obj, method_name)
            and callable(getattr(obj, method_name, None)))


def as_list(obj: any) -> list:
    """Ensure that the input is a list.

    Converts the input to a list if it is iterable; otherwise, it wraps it in a 
    list.

    Parameters
    ----------
    obj : any
        The object to be converted or wrapped in a list.

    Returns
    -------
    list
        The input object as a list.

    """

    try:
        return list(obj)
    except TypeError:
        return [obj]


def get_kwarg_names(function: Callable) -> List[str]:
    """Retrieve the names of the keyword arguments accepted by a function.
    
    It retrieves the names of the keyword arguments accepted by `function` as a
    list of strings.

    Parameters
    ----------
    function : Callable
        The function whose keyword argument names are to be retrieved.

    Returns
    -------
    List[str]
        A list of names of keyword arguments the function accepts.

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
    """Check if a specific argument of a function has a default value.

    Parameters
    ----------
    function : Callable
        The function to inspect.
    argument : str
        Name of the argument to check.

    Returns
    -------
    bool
        True if the specified argument has a default value.

    """
    
    args = get_kwarg_names(function)

    if argument not in args:
        return False

    defaults = inspect.getfullargspec(function).defaults or ()

    return len(args) - args.index(argument) <= len(defaults)


def safe_call(function, positional_args=[], **kwargs) -> Any:
    """Calls a function with valid arguments from a dictionary of arguments.
    
    It filters `kwargs` to include only arguments accepted by the function,
    ensuring that no invalid arguments are passed. This function also supports
    positional arguments.

    Parameters
    ----------
    function : Callable
        The function to call.
    positional_args : list, optional
        List of positional arguments to pass to the function.
    kwargs : dict
        Dictionary of keyword arguments to filter and pass.
    
    Returns
    -------
    Any
        The result of calling the function with the filtered arguments.   
     
    """

    keys = get_kwarg_names(function)

    # Filter kwargs to include only keys present in the function's signature.
    input_arguments = {}
    for key in keys:
        if key in kwargs:
            input_arguments[key] = kwargs[key]

    return function(*positional_args, **input_arguments)