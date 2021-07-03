""" Type declarations for internal use
"""

import typing

import numpy as np

# Property type declaration
T = typing.TypeVar("T")
PropertyLike = typing.Union[T, typing.Callable[..., T]]

ArrayLike = typing.Union[typing.Tuple[T], typing.List[T], np.ndarray]
