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
    A type alias for array-like structures, namely, tuples, lists, NumPy
    arrays, and PyTorch tensors.
- `NumberLike`
    A type alias for numeric types, including scalars and arrays, namely, NumPy
    arrays, PyTorch tensors, bool, int, float, and complex.

Examples
--------
>>> import deeptrack as dt

**Using `PropertyLike`**

>>> def scale(value: PropertyLike[float]) -> float:
...     if callable(value):
...         return value()
...     return value

It works for a given type (in this case, a `float`):

>>> scale(3.14)

It also works for function returning the same type (in this case, a function
returning a `float`):

>>> scale(lambda: 2.71)

`PropertyLike[Type]` is generally used for typing arguments passed to a feature
that are then passed to the constructor of the feature parent, because these
can be intrinsically either `Type` or `Callable[..., Type]`.

**Using `ArrayLike`**

>>> def print_arraylike(array: dt.types.ArrayLike[float]) -> None:
...     print(array)

It works for:

- Lists:

>>> print_arraylike([1.0, 2.0, 3.0])

- Tuples:

>>> print_arraylike((4.0, 5.0, 6.0))

- NumPy arrays:

>>> import numpy as np
>>>
>>> print_arraylike(np.array([7.0, 8.0, 9.0]))

- PyTorch tensors:

>>> import torch
>>>
>>> print_arraylike(torch.tensor([1.0, 2.0, 3.0]))

**Using `NumberLike`**

>>> def add_numbers(a: NumberLike, b: NumberLike) -> NumberLike:
...     return a + b

It works for:

- Scalars (bool, int, float, complex):

>>> add_numbers(5, 3.2)

- NumPy arrays:

>>> import numpy as np
>>>
>>> add_numbers(np.array([1, 2, 3]), 4)

- PyTorch tensors:

>>> import torch
>>>
>>> add_numbers(torch.tensor([1, 2, 3]), 4)

"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, TYPE_CHECKING, Union
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias


from numpy.typing import NDArray


__all__ = [
    "PropertyLike",
    "ArrayLike",
    "NumberLike",
]


if TYPE_CHECKING:
    import torch


# T is a generic type variable defining generic types for reusability.
_T = TypeVar("_T")

# PropertyLike is a type alias representing a value of type T
# or a callable returning type T.
PropertyLike: TypeAlias = Union[_T, Callable[..., _T]]

# ArrayLike is a type alias representing any array-like structure.
# It supports tuples and lists containing elements of type T as well as NumPy
# arrays and PyTorch tensors.
ArrayLike: TypeAlias = Union[
    NDArray[Any],
    "torch.Tensor",
    list[_T],
    tuple[_T, ...],
]

# NumberLike is a type alias representing any numeric type including arrays.
NumberLike: TypeAlias = Union[
    NDArray[Any],
    "torch.Tensor",
    bool,
    int,
    float,
    complex,
]
