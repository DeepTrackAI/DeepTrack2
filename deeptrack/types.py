""" Type declarations for internal use
"""

import typing

# Property type declaration
T = typing.TypeVar("T")
PropertyLike = typing.Union[T, typing.Callable[..., T]]
