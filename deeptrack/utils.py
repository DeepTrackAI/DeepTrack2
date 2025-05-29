"""Utility functions for argument handling and signature inspection.

This module provides utility functions to enhance code readability,
streamline common operations, and ensure type and argument consistency
when working with functions, methods, and callables in Python.

Key Features
------------
- **Method Detection**
  
    Check if an object has a callable method with a given name.

- **List Conversion**

    Ensure that any input is represented as a list.

- **Signature Inspection**

    Retrieve the names of arguments a function accepts, and check for
    default values.

- **Safe Function Calling**

    Call a function by passing only arguments accepted by its signature.

Module Structure
----------------
Functions:

- `hasmethod(obj, method_name)`

    def hasmethod(
        obj: Any,
        method_name: str,
    ) -> bool

    Check if an object has a callable method named `method_name`.

- `as_list(obj)`

    def as_list(obj: Any) -> list[Any]

    Ensure that the input is a list, wrapping if necessary.

- `get_kwarg_names(function)`

    def get_kwarg_names(function: Callable[..., Any]) -> list[str]

    Retrieve the names of the keyword arguments accepted by a function.

- `kwarg_has_default(function, argument)`

    def kwarg_has_default(
          function: Callable[..., Any],
          argument: str,
    ) -> bool

    Check if a specific argument of a function has a default value.

- `safe_call(function, positional_args=None, **kwargs)`

    def safe_call(
        function: Callable[..., Any],
        positional_args: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any

    Call a function, passing only valid arguments from a dictionary.

Examples
--------
TODO

"""

from __future__ import annotations

import inspect
from typing import Any, Callable


__all__ = [
    "hasmethod",
    "as_list",
    "get_kwarg_names",
    "kwarg_has_default",
    "safe_call",
]


def hasmethod(
    obj: Any,
    method_name: str,
) -> bool:
    """Check if an object has a callable method named `method_name`.

    It returns `True` if the object has a field named `method_name` that is 
    callable. Otherwise, returns `False`.

    Parameters
    ----------
    obj: Any
        The object to inspect.
    method_name: str
        The name of the method to look for.

    Returns
    -------
    bool
        True if the object has an attribute named `method_name` that is 
        callable.

    Examples
    --------
    TODO

    """

    return (hasattr(obj, method_name)
            and callable(getattr(obj, method_name, None)))


def as_list(obj: Any) -> list[Any]:
    """Ensure that the input is a list.

    It converts the input to a list if it is iterable and not a string or
    bytes; otherwise, it wraps it in a list.

    Parameters
    ----------
    obj: Any
        The object to be converted or wrapped in a list.

    Returns
    -------
    list[Any]
        The input object as a list.

    Examples
    --------
    TODO

    """

    if isinstance(obj, (str, bytes)):
        return [obj]

    try:
        return list(obj)
    except TypeError:
        return [obj]


def get_kwarg_names(function: Callable[..., Any]) -> list[str]:
    """Retrieve the names of the keyword arguments accepted by a function.
    
    It retrieves the names of the keyword arguments accepted by `function` as a
    list of strings.

    Parameters
    ----------
    function: Callable[..., Any]
        The function whose keyword argument names are to be retrieved.

    Returns
    -------
    list[str]
        A list of names of keyword arguments the function accepts.

    Examples
    --------
    TODO

    """

    try:
        argspec = inspect.getfullargspec(function)
    except TypeError:
        return []

    if argspec.varargs:
        return argspec.kwonlyargs or []
    else:
        return argspec.args or []


def kwarg_has_default(
    function: Callable[..., Any],
    argument: str,
) -> bool:
    """Check if a specific argument of a function has a default value.

    Parameters
    ----------
    function: Callable[..., Any]
        The function to inspect.
    argument: str
        Name of the argument to check.

    Returns
    -------
    bool
        True if the specified argument has a default value.

    Examples
    --------
    TODO

    """

    args = get_kwarg_names(function)

    if argument not in args:
        return False

    defaults = inspect.getfullargspec(function).defaults or ()

    return len(args) - args.index(argument) <= len(defaults)


def safe_call(
    function: Callable[..., Any],
    positional_args: list[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Calls a function with valid arguments from a dictionary of arguments.
    
    It filters `kwargs` to include only arguments accepted by the function,
    ensuring that no invalid arguments are passed. This function also supports
    positional arguments.

    Parameters
    ----------
    function: Callable[..., Any]
        The function to call.
    positional_args: list[Any] | None, optional
        List of positional arguments to pass to the function. Defaults to None.
    **kwargs: dict[str, Any]
        Dictionary of keyword arguments to filter and pass.

    Returns
    -------
    Any
        The result of calling the function with the filtered arguments.   

    Examples
    --------
    TODO

    """

    if positional_args is None:
        positional_args = []

    # Filter kwargs to include only keys present in the function's signature.
    input_arguments = {
        key: kwargs[key] for key in get_kwarg_names(function) if key in kwargs
    }

    return function(*positional_args, **input_arguments)
