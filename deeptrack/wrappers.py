"""Wrappers for arrays with properties.

Wrappers are classes that wrap around arrays (either numpy or torch) and provide additional 
properties and methods.
Wrappers are designed to be flexible and can be used to represent various types of data. 
They allow for easy access to properties and support arithmetic operations while maintaining 
the underlying array structure.

Module Structure
----------------
Classes:

- `Wrapper`: Base class for any structure needing properties. It provides methods for accessing the underlying array, getting properties, and performing arithmetic operations.

Example
-------

>>> 

"""


from __future__ import annotations

import numpy as np
from typing import Any
from dataclasses import dataclass, field

from deeptrack.backend import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

@dataclass
class Wrapper:
    """Base class for any structure needing properties."""

    array: np.ndarray | torch.Tensor
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying array."""
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of dimensions of the underlying array."""
        return self.array.shape

    @property
    def pos3d(self) -> np.ndarray:
        return np.array([*self.position, self.z], dtype=float)

    @property
    def position(self) -> np.ndarray:
        pos = self.properties.get("position", None)
        if pos is None:
            return None
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 2 and pos.shape[0] == 1:
            pos = pos[0]
        return pos

    def copy(
        self,
        *,
        array=None,
        properties=None,
    ) -> Wrapper:
        """Return a shallow copy of the ScatteredBase.

        Parameters
        ----------
        array : np.ndarray | torch.Tensor | None
            Optional replacement for the internal array.
            If None, the existing array is reused.
        properties : dict | None
            Optional replacement for properties.
            If None, a shallow copy of the current properties is used.

        Returns
        -------
        ScatteredBase
            A new ScatteredBase instance.
        """
        return type(self)(
            array=self.array if array is None else array,
            properties=self.properties.copy() if properties is None else properties,
        )


    def as_array(self) -> np.ndarray | torch.Tensor:
        """Return the underlying array.

        Notes
        -----
        The raw array is also directly available as ``scatterer.array``.
        This method exists mainly for API compatibility and clarity.

        """
        
        return self.array

    def get_property(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, self.properties.get(key, default))
    

    def _binary_op(self, other, op, reverse=False):
        a = self.array
        b = other.array if isinstance(other, Wrapper) else other

        result = op(b, a) if reverse else op(a, b)

        new = self.copy()
        new.array = result
        return new

    def _unary_op(self, op):
        new = self.copy()
        new.array = op(self.array)
        return new

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_op(other, lambda a, b: a + b, reverse=True)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: a - b, reverse=True)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, lambda a, b: a * b, reverse=True)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b, reverse=True)

    def __floordiv__(self, other):
        return self._binary_op(other, lambda a, b: a // b)

    def __rfloordiv__(self, other):
        return self._binary_op(other, lambda a, b: a // b, reverse=True)

    def __pow__(self, other):
        return self._binary_op(other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return self._binary_op(other, lambda a, b: a ** b, reverse=True)

    def __gt__(self, other):
        return self._binary_op(other, lambda a, b: a > b)
    
    def __rgt__(self, other):
        return self._binary_op(other, lambda a, b: a > b, reverse=True)

    def __lt__(self, other):
        return self._binary_op(other, lambda a, b: a < b)
    
    def __rlt__(self, other):
        return self._binary_op(other, lambda a, b: a < b, reverse=True)
    
    def __ge__(self, other):
        return self._binary_op(other, lambda a, b: a >= b)
    
    def __rge__(self, other):
        return self._binary_op(other, lambda a, b: a >= b, reverse=True)

    def __le__(self, other):
        return self._binary_op(other, lambda a, b: a <= b)
    
    def __rle__(self, other):
        return self._binary_op(other, lambda a, b: a <= b, reverse=True)

    def __eq__(self, other):
        return self._binary_op(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._binary_op(other, lambda a, b: a != b)

    def __and__(self, other):
        return self._binary_op(other, lambda a, b: a & b)

    def __rand__(self, other):
        return self._binary_op(other, lambda a, b: a & b, reverse=True)

    def __xor__(self, other):
        return self._binary_op(other, lambda a, b: a ^ b)



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