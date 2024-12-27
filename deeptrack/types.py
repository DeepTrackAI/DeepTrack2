"""Type declarations for internal use.

This module defines type aliases and utility types to standardize the type 
annotations used throughout the codebase. It enhances code readability, 
maintainability, and reduces redundancy in type annotations. These types are 
particularly useful for properties and array-like structures used within the 
library.

Defined Types
-------------
- `PropertyLike`
    A type alias representing a value of type `T` or a callable returning `T`.
- `ArrayLike`
    A type alias for array-like structures (e.g., tuples, lists, numpy arrays).
- `NumberLike`
    A type alias for numeric types, including scalars and arrays (e.g., numpy 
    arrays, GPU tensors).

Examples
--------
Using `PropertyLike`:

>>> def scale(value: PropertyLike[float]) -> float:
...     if callable(value):
...         return value()
...     return value
>>> scale(3.14)  # 3.14
>>> scale(lambda: 2.71)  # 2.71

Using `ArrayLike`:

>>> import numpy as np
>>> def compute_mean(array: ArrayLike[float]) -> float:
...     return np.mean(array)
>>> compute_mean([1.0, 2.0, 3.0])  # 2.0
>>> compute_mean((4.0, 5.0, 6.0))  # 5.0
>>> compute_mean(np.array([7.0, 8.0, 9.0]))  # 8.0

Using `NumberLike`:

>>> def add_numbers(a: NumberLike, b: NumberLike) -> NumberLike:
...     return a + b
>>> add_numbers(5, 3.2)  # 8.2
>>> add_numbers(np.array([1, 2, 3]), 4)  # array([5, 6, 7])

"""

from typing import Callable, List, Tuple, TypeVar, Union

import numpy as np

# Try importing optional libraries for GPU arrays and tensors.
try:
    import cupy
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# T is a generic type variable defining generic types for reusability.
_T = TypeVar("T")

# PropertyLike is a type alias representing a value of type T 
# or a callable returning type T.
PropertyLike = Union[_T, Callable[..., _T]]

# ArrayLike is a type alias representing any array-like structure.
# It supports tuples, lists, and numpy arrays containing elements of type T.
ArrayLike = Union[Tuple[_T, ...], List[_T], np.ndarray]

# NumberLike is a type alias representing any numeric type including arrays.
NumberLike = Union[np.ndarray, int, float, bool, complex]

if _CUPY_AVAILABLE:
    NumberLike = Union[NumberLike, cupy.ndarray]

if _TORCH_AVAILABLE:
    NumberLike = Union[NumberLike, torch.Tensor]
