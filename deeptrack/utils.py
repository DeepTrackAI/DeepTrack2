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
Check if a method exists in an object:

>>> from deeptrack.utils import hasmethod
>>> class Example:
...     def foo(self): pass
>>> hasmethod(Example(), "foo")
True
>>> hasmethod(Example(), "bar")
False

Convert various objects to lists:

>>> from deeptrack.utils import as_list
>>> as_list(42)
[42]
>>> as_list((1, 2))
[1, 2]
>>> as_list("abc")
['abc']

Retrieve keyword argument names from a function:

>>> from deeptrack.utils import get_kwarg_names
>>> def func(x, y=1, z=2):
...     pass
>>> get_kwarg_names(func)
['x', 'y', 'z']

Check if a function argument has a default value:

>>> from deeptrack.utils import kwarg_has_default
>>> def func(x, y=1):
...     pass
>>> kwarg_has_default(func, "x")
False
>>> kwarg_has_default(func, "y")
True

Safely call a function with extra arguments:

>>> from deeptrack.utils import safe_call
>>> def f(a, b=2, c=3):
...     return a + b + c
>>> safe_call(f, positional_args=[1], b=5, x=100)
9

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
    >>> from deeptrack.utils import hasmethod

    Check if an object has a method called 'foo':

    >>> class MyClass:
    ...     def foo(self):
    ...         return 42
    >>> obj = MyClass()
    >>> hasmethod(obj, "foo")
    True
    >>> hasmethod(obj, "bar")
    False

    Built-in types:

    >>> hasmethod([1, 2, 3], "append")
    True
    >>> hasmethod([1, 2, 3], "not_a_method")
    False

    Modules:

    >>> import math
    >>> hasmethod(math, "sqrt")
    True
    >>> hasmethod(math, "not_existing")
    False

    Edge cases:

    >>> hasmethod(42, "bit_length")
    True
    >>> hasmethod(42, "foo")
    False
    >>> hasmethod(None, "foo")
    False

    """

    return (hasattr(obj, method_name)
            and callable(getattr(obj, method_name, None)))


def as_list(obj: Any) -> list[Any]:
    """Ensure that the input is a list.

    It converts the input to a list if it is iterable and not a string or
    bytes; otherwise, it wraps it in a list.

    Note: If `obj` is a PyTorch Tensor, this function will return a list of its
    elements along the first dimension (e.g., for a 2D tensor, the result
    will be a list of 1D tensors). If you want to wrap the entire tensor in a
    list, use `[obj]` explicitly.

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
    from deeptrack.utils import as_list

    Wrap a scalar in a list:

    >>> as_list(5)
    [5]
    >>> as_list(None)
    [None]

    Pass through a list unchanged:

    >>> as_list([1, 2, 3])
    [1, 2, 3]

    Convert a tuple or set to a list:

    >>> as_list((1, 2, 3))
    [1, 2, 3]
    >>> sorted(as_list({3, 2, 1}))
    [1, 2, 3]

    Convert a generator to a list:

    >>> generator = (x * 2 for x in range(3))
    >>> as_list(generator)
    [0, 2, 4]

    Strings and bytes are treated as atomic (not split):

    >>> as_list("abc")
    ['abc']
    >>> as_list(b"xyz")
    [b'xyz']

    NumPy arrays become lists of elements:

    >>> import numpy as np
    >>> as_list(np.array([1, 2, 3]))
    [1, 2, 3]

    PyTorch tensors become lists of elements along the first dimension
    (if PyTorch is available):

    >>> import torch
    >>> t = torch.tensor([[1, 2], [3, 4]])
    >>> as_list(t)
    [tensor([1, 2]), tensor([3, 4])]

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
    from deeptrack.utils import get_kwarg_names

    Basic usage:

    >>> def f(a, b=1, c=2):
    ...     pass
    >>> get_kwarg_names(f)
    ['a', 'b', 'c']

    Functions with only positional arguments:

    >>> def g(x, y):
    ...     pass
    >>> get_kwarg_names(g)
    ['x', 'y']

    Functions with *args and **kwargs (note: **kwargs are not listed):

    >>> def k(*args, alpha=0.1, beta=0.2, **kwargs):
    ...     pass
    >>> get_kwarg_names(k)
    ['alpha', 'beta']

    Built-in functions (may return an empty list):

    >>> get_kwarg_names(len)
    ['obj']

    Lambda functions:

    >>> get_kwarg_names(lambda x, y=5: x + y)
    ['x', 'y']

    Methods (including 'self'):

    >>> class MyClass:
    ...     def method(self, a, b=2):
    ...         pass
    >>> get_kwarg_names(MyClass.method)
    ['self', 'a', 'b']

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
    from deeptrack.utils import kwarg_has_default

    Check default values for positional and keyword-only arguments:

    >>> def f(a, b=2, c=3):
    ...     pass
    >>> kwarg_has_default(f, "a")
    False
    >>> kwarg_has_default(f, "b")
    True
    >>> kwarg_has_default(f, "c")
    True

    Missing argument:

    >>> kwarg_has_default(f, "not_present")
    False

    Keyword-only arguments without defaults:

    >>> def g(*, flag):
    ...     pass
    >>> kwarg_has_default(g, "flag")
    False

    Method example:

    >>> class MyClass:
    ...     def method(self, x, y=42):
    ...         pass
    >>> kwarg_has_default(MyClass.method, "self")
    False
    >>> kwarg_has_default(MyClass.method, "x")
    False
    >>> kwarg_has_default(MyClass.method, "y")
    True

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
    from deeptrack.utils import safe_call

    Basic usage with positional and keyword arguments:

    >>> def f(a, b=2, c=3):
    ...     return a + b + c
    >>> safe_call(f, positional_args=[1], b=4, x=100)
    8

    All keyword arguments:

    >>> safe_call(f, a=1, b=2, c=3)
    6

    Extra keyword arguments (ignored if not accepted by the function):

    >>> safe_call(f, a=2, extra=42)
    7

    Missing required argument (raises TypeError):

    >>> safe_call(f, b=2, c=3)
    Traceback (most recent call last):
        ...
    TypeError: ...

    Function with *args and **kwargs (the kwargs are not passed):

    >>> def g(a, *args, b=5, **kwargs):
    ...     return a, args, b, kwargs
    >>> safe_call(g, positional_args=[1, 10], b=7, x=3, y=2)
    (1, (10,), 7, {})

    Function with only *args (positional):

    >>> def h(*args):
    ...     return args
    >>> safe_call(h, positional_args=[1, 2, 3])
    (1, 2, 3)

    Function with only **kwargs (the kwargs are not passed):

    >>> def i(**kwargs):
    ...     return sorted(kwargs.items())
    >>> safe_call(i, foo=1, bar=2)
    []

    """

    if positional_args is None:
        positional_args = []

    # Filter kwargs to include only keys present in the function's signature.
    input_arguments = {
        key: kwargs[key] for key in get_kwarg_names(function) if key in kwargs
    }

    return function(*positional_args, **input_arguments)
