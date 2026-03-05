"""Wrappers for arrays with properties.

This module defines lightweight container classes that wrap arrays together
with associated metadata (properties). A wrapper behaves similarly to a
NumPy or PyTorch array while carrying additional contextual information such
as spatial coordinates or simulation parameters.

Wrappers are designed to preserve metadata during arithmetic operations.
When mathematical or logical operations are applied to wrappers, the
underlying arrays are combined while the associated properties are propagated
to the resulting wrapper.

The module is backend-agnostic and supports both NumPy and PyTorch arrays,
allowing wrappers to be used consistently across DeepTrack pipelines
regardless of the active numerical backend.

Key Features
------------
- **Array container with metadata**

    The `Wrapper` class stores an array together with a dictionary of
    properties describing the array. These properties can include spatial
    coordinates, identifiers, or other contextual metadata.

- **Backend-independent behavior**

    Wrappers support both NumPy and PyTorch arrays. Arithmetic and logical
    operations preserve the backend of the underlying array.

- **Property propagation**

    Arithmetic operations between wrappers return new wrappers that preserve
    the original properties while operating on the underlying arrays.

Module Structure
----------------
Classes:

- `Wrapper`: Container for arrays with associated metadata.

    A lightweight data structure that stores an array together with a
    dictionary of properties. The class provides convenience attributes
    (such as `shape` and `ndim`) and supports arithmetic and logical
    operations while preserving metadata.

Examples
--------
>>> import deeptrack as dt

Create a wrapper from an array:

>>> import numpy as np
>>>
>>> array = np.arange(9).reshape(3, 3)
>>> wrapper = dt.Wrapper(array, properties={"position": (1, 2)})
Wrapper(array=array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]]), properties={'position': (1, 2)})

Access array attributes:

>>> wrapper.shape
(3, 3)

>>> wrapper.ndim
2

Access properties:

>>> wrapper.properties
{'position': (1, 2)}

Perform arithmetic operations:

>>> wrapper2 = wrapper + 2
>>> wrapper2
Wrapper(array=array([[ 2,  3,  4],
       [ 5,  6,  7],
       [ 8,  9, 10]]), properties={'position': (1, 2)})

Wrappers preserve metadata while modifying the underlying array.

"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Callable

import numpy as np

from deeptrack.backend import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


__all__ = ["Wrapper"]


@dataclass
class Wrapper:
    """Base class for any structure needing properties.

    A `Wrapper` stores an array together with a dictionary of properties.
    The wrapper behaves similarly to the underlying array for arithmetic and
    logical operations while preserving the associated metadata.

    When operations are applied to wrappers, a new wrapper is returned where
    the array contains the result of the operation and the properties are
    copied from the left-hand operand.

    Parameters
    ----------
    array: np.ndarray | torch.Tensor
        The array wrapped by this object.
    properties: dict[str, Any], optional
        Dictionary of metadata associated with the array.

    Attributes
    ----------
    array: np.ndarray | torch.Tensor
        The wrapped array.
    properties: dict[str, Any]
        Metadata associated with the array.

    Methods
    -------
    `copy(*, array, properties) -> Wrapper`
        Return a shallow copy of the wrapper.
    `as_array() -> np.ndarray | torch.Tensor`
        Return the wrapped array.
    `get_property(key, default) -> Any`
        Retrieve a value, checking wrapper attributes before `properties`.

    Examples
    --------
    Wrappers can be used with both NumPy and PyTorch backends through the
    DeepTrack backend configuration.

    >>> import deeptrack as dt
    >>> from deeptrack import xp

    Use the NumPy backend:

    >>> dt.config.set_backend("numpy")
    >>> a = xp.arange(9, dtype=xp.float32).reshape(3, 3)
    >>> w = dt.Wrapper(a, properties={"position": (1, 2)})
    >>> w
    Wrapper(array=array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]], dtype=float32), properties={'position': (1, 2)})

    Array attributes are accessible:

    >>> w.shape
    (3, 3)

    >>> w.ndim
    2

    Properties are also accessible:

    >>> w.properties
    {'position': (1, 2)}

    Arithmetic operations return new wrappers and preserve properties:

    >>> w2 = w + 2
    >>> w2
    Wrapper(array=array([[ 2.,  3.,  4.],
           [ 5.,  6.,  7.],
           [ 8.,  9., 10.]], dtype=float32), properties={'position': (1, 2)})

    Wrappers can also be combined:

    >>> b = xp.ones((3, 3), dtype=xp.float32)
    >>> w3 = w + dt.Wrapper(b)
    >>> w3
    Wrapper(array=array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]], dtype=float32), properties={'position': (1, 2)})

    Logical operations return wrappers as well:

    >>> mask = w > 5
    >>> mask
    Wrapper(array=array([[False, False, False],
           [False, False, False],
           [ True,  True,  True]]), properties={'position': (1, 2)})

    Switch to the PyTorch backend:

    >>> dt.config.set_backend("torch")
    >>> a = xp.arange(9, dtype=xp.float32).reshape(3, 3)
    >>> w = dt.Wrapper(a, properties={"position": (1, 2)})
    >>> w
    Wrapper(array=tensor([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]]), properties={'position': (1, 2)})

    Operations behave the same way:

    >>> w2 = 2 + w
    >>> w2
    Wrapper(array=tensor([[ 2.,  3.,  4.],
           [ 5.,  6.,  7.],
           [ 8.,  9., 10.]]), properties={'position': (1, 2)})

    """

    array: np.ndarray | torch.Tensor
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def ndim(self: Wrapper) -> int:
        """Number of dimensions of the wrapped array."""
        return self.array.ndim

    @property
    def shape(self: Wrapper) -> tuple[int, ...]:
        """Shape of the wrapped array."""
        return self.array.shape

    # TODO CM: pos3d and position should be in the subclass used for light
    # microscopy, right?

    # @property
    # def pos3d(self) -> np.ndarray:
    #    return np.array([*self.position, self.z], dtype=float)

    # @property
    # def position(self) -> np.ndarray:
    #    pos = self.properties.get("position", None)
    #    if pos is None:
    #        return None
    #    pos = np.asarray(pos, dtype=float)
    #    if pos.ndim == 2 and pos.shape[0] == 1:
    #        pos = pos[0]
    #    return pos

    def copy(
        self: Wrapper,
        *,
        array: np.ndarray | torch.Tensor | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Wrapper:
        """Return a shallow copy of the Wrapper.

        Parameters
        ----------
        array: np.ndarray | torch.Tensor | None, optional
            Replacement for the wrapped array. If `None`, the existing array
            is reused.
        properties: dict[str, Any] | None, optional
            Replacement for the properties dictionary. If `None`, a shallow
            copy of the current properties is used.

        Returns
        -------
        Wrapper
            A new Wrapper instance.

        """

        return type(self)(
            array=self.array if array is None else array,
            properties=(
                properties
                if properties is not None
                else self.properties.copy()
            ),
        )

    def as_array(self: Wrapper) -> np.ndarray | torch.Tensor:
        """Return the underlying array.

        Notes
        -----
        The raw array is also directly available as `self.array`. This method
        exists mainly for API compatibility and clarity.

        Returns
        -------
        np.ndarray | torch.Tensor
            The wrapped array.

        """

        return self.array

    def get_property(
        self: Wrapper,
        key: str,
        default: Any = None,
    ) -> Any:
        """Return a property value with attribute fallback.

        This method first attempts to retrieve `key` as an attribute of the
        wrapper. If the attribute does not exist, the method looks for `key`
        in the wrapper's `properties` dictionary.

        Parameters
        ----------
        key: str
            Name of the property to retrieve.
        default: Any, optional
            Value returned if the property is not found.

        Returns
        -------
        Any
            The resolved property value.

        Examples
        --------
        >>> import deeptrack as dt

        >>> import numpy as np
        >>>
        >>> w = dt.Wrapper(np.zeros((2, 2)), properties={"id": 1})
        >>> w.get_property("id")
        1

        Attributes take precedence over dictionary properties:

        >>> w.get_property("shape")
        (2, 2)

        """

        return getattr(self, key, self.properties.get(key, default))

    def _binary_op(
        self: Wrapper,
        other: Any,
        op: Callable[[Any, Any], Any],
        reverse: bool = False,
    ) -> Wrapper:
        """Apply a binary operation and return a new wrapper.

        Parameters
        ----------
        other : Any
            Right-hand operand. If it is a `Wrapper`, its array is used.
        op : Callable
            Binary operation applied to the underlying arrays.
        reverse : bool, optional
            If True, apply the operation as `op(other, self.array)`.

        Returns
        -------
        Wrapper
            A new wrapper containing the result of the operation.

        """

        a = self.array
        b = other.array if isinstance(other, Wrapper) else other

        result = op(b, a) if reverse else op(a, b)

        new = self.copy()
        new.array = result
        return new

    def _unary_op(
        self: Wrapper,
        op: Callable[[Any], Any],
    ) -> Wrapper:
        """Apply a unary operation to the wrapped array.

        Parameters
        ----------
        op : Callable
            Unary operation applied to the underlying array.

        Returns
        -------
        Wrapper
            A new wrapper containing the result.

        """

        new = self.copy()
        new.array = op(self.array)
        return new

    # ---------------------------------------------------------------------
    # Arithmetic, comparison, and logical operators
    # ---------------------------------------------------------------------
    # These methods forward the corresponding Python operators to the
    # underlying array using `_binary_op`. The result is returned as a new
    # `Wrapper` instance while preserving the properties of the left-hand
    # operand.

    # Arithmetic operators

    def __add__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.add)

    def __radd__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.add, reverse=True)

    def __sub__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.sub)

    def __rsub__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.sub, reverse=True)

    def __mul__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.mul)

    def __rmul__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.mul, reverse=True)

    def __truediv__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.truediv, reverse=True)

    def __floordiv__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.floordiv)

    def __rfloordiv__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.floordiv, reverse=True)

    def __pow__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.pow)

    def __rpow__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.pow, reverse=True)

    # Comparison operators

    def __gt__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.gt)

    def __lt__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.lt)

    def __ge__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.ge)

    def __le__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.le)

    def __eq__(self: Wrapper, other: Any) -> Wrapper:  # type: ignore[override]
        return self._binary_op(other, operator.eq)

    def __ne__(self: Wrapper, other: Any) -> Wrapper:  # type: ignore[override]
        return self._binary_op(other, operator.ne)

    # Logical operators

    def __and__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.and_)

    def __rand__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.and_, reverse=True)

    def __xor__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.xor)

    def __rxor__(self: Wrapper, other: Any) -> Wrapper:
        return self._binary_op(other, operator.xor, reverse=True)

    # TODO CM: Can we erase this?

    # def _apply(x, y):

    # x_wrapped = hasattr(x, "array")
    # if x_wrapped:
    #     x_obj = x.copy()
    #     x = x_obj.array

    # if hasattr(y, "array"):
    #     y = y.array

    # result = self.op(x, y)

    # if x_wrapped:
    #     x_obj.array = result
    #     return x_obj

    # return result
