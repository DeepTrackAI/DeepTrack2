"""PyTorch conversion features for DeepTrack2.

This module provides features that convert DeepTrack2 outputs to PyTorch
tensors. It is intended as a lightweight bridge between DeepTrack2 feature
pipelines and PyTorch training workflows.

Key Features
------------
- **Convert arbitrary outputs to `torch.Tensor`**

    Supports NumPy arrays, PyTorch tensors, Python scalars, and array-like
    sequences.

- **Optional channel-last to channel-first permutation**

    Enables converting `(H, W, C)` arrays to `(C, H, W)` tensors for common
    computer-vision conventions.

Module Structure
----------------
Classes:

- `ToTensor`

    Convert an input to a PyTorch tensor, with optional dtype/device casting
    and optional permutation to channel-first layout.

Examples
--------
>>> import numpy as np
>>> import deeptrack as dt
>>> from deeptrack.torch.features import ToTensor

Convert a NumPy image to a torch tensor:

>>> feature = (
...     dt.Value(value=np.zeros((32, 32), dtype=np.float32))
...     >> ToTensor()
... )
>>> out = feature()
>>> out.shape
torch.Size([32, 32])

Convert a channel-last NumPy image to channel-first:

>>> feature = (
...     dt.Value(value=np.zeros((32, 32, 3), dtype=np.float32))
...     >> ToTensor(permute_mode="numpy")
... )
>>> out = feature()
>>> out.shape
torch.Size([3, 32, 32])

Return a scalar unchanged unless explicitly requested:

>>> ToTensor(add_dim_to_number=False)(1.0)
1.0

>>> ToTensor(add_dim_to_number=True)(1.0).shape
torch.Size([1])

"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from deeptrack.features import Feature


_PERMUTE_MODE_ = Literal["always", "never", "numpy", "numpy_and_not_int"]


class ToTensor(Feature):
    """Convert inputs to a PyTorch tensor.

    Parameters
    ----------
    dtype: torch.dtype | None, optional
        Dtype to cast the resulting tensor to. If `None`, no dtype cast is
        performed.
    device: torch.device | str | None, optional
        Device to move the tensor to. If `None`, no device transfer is
        performed.
    add_dim_to_number: bool, optional
        If `True`, scalar numbers are converted to a 1D tensor of shape `(1,)`.
        If `False`, scalar numbers are returned unchanged. Defaults to `False`.
    permute_mode: {"always", "never", "numpy", "numpy_and_not_int"}, optional
        Controls channel-last to channel-first permutation:

        - `"always"`: permute whenever the resulting tensor has `ndim > 2`
        - `"never"`: never permute
        - `"numpy"`: permute only if the input was a NumPy array
        - `"numpy_and_not_int"`: permute only if the input was a NumPy array
        and its dtype is not an integer dtype

        Defaults to `"never"`.

    Notes
    -----
    NumPy arrays with negative strides (e.g. `x[:, ::-1]`) cannot be converted
    to torch tensors without copying. This feature detects such arrays and
    copies them before conversion.

    """

    def __init__(
        self: ToTensor,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        add_dim_to_number: bool = False,
        permute_mode: _PERMUTE_MODE_ = "never",
        **kwargs: Any,
    ) -> None:
        """Initialize the `ToTensor` feature.

        Parameters
        ----------
        dtype: torch.dtype | None, optional
            Dtype to cast the resulting tensor to. If `None`, no dtype cast is
            performed.
        device: torch.device | str | None, optional
            Device to move the tensor to. If `None`, no device transfer is
            performed.
        add_dim_to_number: bool, optional
            If `True`, scalar numbers are converted to a 1D tensor of shape
            `(1,)`. If `False`, scalar numbers are returned unchanged.
            Defaults to `False`.
        permute_mode: {"always", "never", "numpy", "numpy_and_not_int"}, optional
            Controls channel-last to channel-first permutation:

            - `"always"`: permute whenever the resulting tensor has `ndim > 2`
            - `"never"`: never permute
            - `"numpy"`: permute only if the input was a NumPy array
            - `"numpy_and_not_int"`: permute only if the input was a NumPy
              array and its dtype is not an integer dtype

            Defaults to `"never"`.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Feature` class.

        """

        super().__init__(
            dtype=dtype,
            device=device,
            add_dim_to_number=add_dim_to_number,
            permute_mode=permute_mode,
            **kwargs,
        )

    def get(
        self: ToTensor,
        x: Any,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
        add_dim_to_number: bool,
        permute_mode: _PERMUTE_MODE_,
        **kwargs: Any,
    ) -> Any:
        """Convert a single input to a PyTorch tensor.

        This method is called internally by the `Feature` resolution
        mechanism. It converts the input `x` to a `torch.Tensor`
        according to the specified configuration.

        Parameters
        ----------
        x: Any
            The input object to convert. Supported types include:

            - `torch.Tensor`
            - `numpy.ndarray`
            - Python scalars (`int`, `float`, `bool`, `complex`)
            - Array-like sequences

        dtype: torch.dtype | None
            If provided, the resulting tensor is cast to this dtype.

        device: torch.device | str | None
            If provided, the resulting tensor is moved to this device.

        add_dim_to_number: bool
            If `True`, scalar numbers are converted to tensors of shape
            `(1,)`. If `False`, scalar numbers are returned unchanged.

        permute_mode: {"always", "never", "numpy", "numpy_and_not_int"}
            Controls whether channel-last inputs are permuted to
            channel-first layout.

        Returns
        -------
        Any
            A `torch.Tensor` if conversion occurs, otherwise the original
            input (for scalar numbers when `add_dim_to_number=False`).

        Notes
        -----
        - NumPy arrays with negative strides are copied before conversion.
        - Permutation is only applied when the resulting tensor has
          more than two dimensions.

        """

        numpy_dtype = x.dtype if isinstance(x, np.ndarray) else None

        if isinstance(x, torch.Tensor):
            tensor = x

        elif isinstance(x, np.ndarray):
            if any(stride < 0 for stride in x.strides):
                x = x.copy()
            tensor = torch.from_numpy(x)

        elif isinstance(x, (int, float, bool, complex)):
            if add_dim_to_number:
                tensor = torch.tensor([x])
            else:
                return x
        else:
            tensor = torch.as_tensor(x)

        should_permute = False
        if tensor.ndim > 2:
            if permute_mode == "always":
                should_permute = True
            elif permute_mode == "numpy" and isinstance(x, np.ndarray):
                should_permute = True
            elif (
                permute_mode == "numpy_and_not_int"
                and isinstance(x, np.ndarray)
                and numpy_dtype is not None
                and numpy_dtype.kind not in ("i", "u")
            ):
                should_permute = True
            
        if should_permute:
            tensor = tensor.permute(-1, *range(tensor.dim() - 1))

        if dtype is not None:
            tensor = tensor.to(dtype)

        if device is not None:
            tensor = tensor.to(device)

        return tensor
