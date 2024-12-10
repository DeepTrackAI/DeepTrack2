"""Type declarations for internal use.

This module defines type aliases and utility types to standardize the type 
annotations used throughout the codebase. It enhances code readability, 
maintainability, and reduces redundancy in type annotations. These types are 
particularly useful for properties and array-like structures used within the 
library.

"""

import typing

import numpy as np


# T is a generic type variable defining generic types for reusability.
T = typing.TypeVar("T")

# PropertyLike is a type alias representing a value of type T 
# or a callable returning type T.
PropertyLike = typing.Union[T, typing.Callable[..., T]]

# ArrayLike is a type alias representing any array-like structure.
# It supports tuples, lists, and numpy arrays containing elements of type T.
ArrayLike = typing.Union[typing.Tuple[T], typing.List[T], np.ndarray]
