"""PyTorch dataset adapter for DeepTrack2.

This module provides a lightweight wrapper that exposes a DeepTrack2 pipeline
as a `torch.utils.data.Dataset`. It is intended for integrating DeepTrack2
data generation pipelines with standard PyTorch training workflows.

Key Features
------------
- **On-Demand Evaluation with Caching**

    Samples are generated when accessed and stored in an internal cache.

- **Flexible Cache Replacement Policies**

    Cached samples can be regenerated always, never, probabilistically, or
    using a user-supplied callable.

- **Robust Conversion to PyTorch Tensors**

    NumPy arrays, scalars, and array-like sequences are converted to
    `torch.Tensor`. NumPy arrays with negative strides are copied before
    conversion.

Module Structure
----------------
Classes:

- `Dataset`

    Wraps a DeepTrack2 pipeline as a PyTorch dataset.

Examples
--------
>>> import deeptrack as dt
>>> from deeptrack.pytorch.data import Dataset

Create a simple pipeline and dataset of fixed length:

>>> import numpy as np
>>>
>>> pipeline = dt.Value(value=np.ones((1, 2), dtype=np.float32))
>>> ds = Dataset(pipeline, length=3)
>>> ds[0]
(tensor([[1., 1.]]),)

Use probabilistic replacement:

>>> ds = Dataset(pipeline, length=3, replace=0.5)

"""

from __future__ import annotations

from typing import Any, cast, Callable, Sequence

import numpy as np
import torch

from deeptrack import Feature


__all__ = ["Dataset"]


class Dataset(torch.utils.data.Dataset):
    """Expose a DeepTrack2 pipeline as a PyTorch `Dataset`.

    This class evaluates a DeepTrack2 pipeline on demand. Each item is cached
    after it is generated. Cache replacement is controlled by the `replace`
    parameter.

    Parameters
    ----------
    pipeline: Feature
        The DeepTrack2 pipeline to evaluate. The pipeline is expected to
        support `.update()` and `__call__()`.
    inputs: Sequence[Any] or None, optional
        Sequence of inputs, one per dataset index. If `None`, `length` must be
        provided and the dataset will use empty lists as inputs.
    length: int or None, optional
        Length of the dataset when `inputs` is not provided.
    replace: bool or float or Callable, optional
        Policy for regenerating cached samples:
        - `False`: never replace (cache once generated).
        - `True`: always replace (regenerate every access).
        - `float` in [0, 1]: replace with that probability.
        - `callable`: either `replace()` or `replace(index)` returning bool.
        Defaults to `False`.
    float_dtype: torch.dtype | str | None, optional
        If not `None`, floating-point tensors are cast to this dtype.
        Use `"default"` to cast to `torch.get_default_dtype()`. Defaults to
        `"default"`.

    Attributes
    ----------
    pipeline: Feature
        The wrapped DeepTrack2 pipeline.
    replace: bool | float | Callable[[], bool] | Callable[[int], bool]
        Replacement policy for cached samples.
    inputs: Sequence[Any]
        Input objects passed to the pipeline at each index.
    data: list[tuple[Any, ...] | None]
        Cache of generated samples. Each cached sample is a tuple of tensors.
    float_dtype: torch.dtype | str | None
        Floating dtype used for casting.

    Notes
    -----
    The pipeline is assumed to produce tensor-like outputs (NumPy arrays,
    tensors, scalars, or array-like sequences). If the pipeline returns
    objects that cannot be converted by `torch.as_tensor`, a `TypeError`
    will be raised during conversion.

    """

    pipeline: Feature
    replace: bool | float | Callable[[], bool] | Callable[[int], bool]
    inputs: Sequence[Any]
    data: list[tuple[Any, ...] | None]
    float_dtype: torch.dtype | str | None

    def __init__(
        self: Dataset,
        pipeline: Feature,
        inputs: Sequence[Any] | None = None,
        length: int | None = None,
        replace: (
            bool
            | float
            | Callable[[], bool]
            | Callable[[int], bool]
        ) = False,
        float_dtype: torch.dtype | str | None = "default",
    ) -> None:
        """Initialize the dataset wrapper.

        Parameters
        ----------
        pipeline: Feature
            The DeepTrack2 pipeline to evaluate.
        inputs: Sequence[Any] | None, optional
            Inputs passed to the pipeline at each index.
        length: int | None, optional
            Dataset length if `inputs` is `None`.
        replace: bool | float | Callable, optional
            Cache replacement policy.
        float_dtype: torch.dtype | str | None, optional
            Floating dtype used for casting.

        """

        self.pipeline = pipeline

        self.replace = replace

        if inputs is None:
            if length is None:
                raise ValueError("Either inputs or length must be specified.")
            inputs = [[] for _ in range(length)]
        self.inputs = inputs

        self.data = [None for _ in inputs]

        if float_dtype == "default":
            float_dtype = torch.get_default_dtype()
        self.float_dtype = float_dtype

    def __getitem__(
        self: Dataset,
        index: int,
    ) -> tuple[Any, ...]:
        """Return the sample at `index`.

        If a cached sample exists and the replacement policy does not request
        regeneration, the cached sample is returned.

        Parameters
        ----------
        index: int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[Any, ...]
            A tuple of outputs converted to `torch.Tensor`.

        """

        if self._should_replace(index):
            self.pipeline.update()
            result = self.pipeline(self.inputs[index])
            if not isinstance(result, (tuple, list)):
                result = (result,)
            result = tuple(self._as_tensor(r) for r in result)

            self.data[index] = result

        out = self.data[index]
        if out is None:  # pragma: no cover
            raise RuntimeError("Dataset cache invariant broken.")
        return out
    
    def _as_tensor(
        self: Dataset,
        x: Any,
    ) -> torch.Tensor:
        """Convert an object to a `torch.Tensor`.

        Parameters
        ----------
        x: Any
            Object to convert. Supported inputs include `torch.Tensor`,
            `numpy.ndarray`, Python scalars, and array-like sequences.

        Returns
        -------
        torch.Tensor
            Converted tensor.

        Notes
        -----
        NumPy arrays with negative strides are copied before conversion.

        """

        if isinstance(x, torch.Tensor):
            tensor = x
        elif isinstance(x, (int, float, bool, complex)):
            tensor = torch.as_tensor([x])
        elif isinstance(x, np.ndarray):
            if any(stride < 0 for stride in x.strides):
                x = x.copy()

            numpy_dtype = x.dtype
            tensor = torch.from_numpy(x)

            if tensor.ndim > 2 and numpy_dtype not in (
                np.uint8, np.uint16, np.uint32, np.uint64,
            ):
                tensor = tensor.permute(-1, *range(tensor.ndim - 1))
        else:
            tensor = torch.as_tensor(x)

        if self.float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(self.float_dtype)

        if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            tensor = tensor.to(torch.long)

        return tensor

    def _should_replace(
        self: Dataset,
        index: int,
    ) -> bool:
        """Determine whether a cached sample should be regenerated.

        This method implements the cache replacement policy defined by the
        `replace` attribute.

        The behavior is as follows:
        - If no cached value exists at `index`, return `True`.
        - If `replace` is a bool, return its value directly.
        - If `replace` is a float in [0, 1], return `True` with that
        probability.
        - If `replace` is callable, call it either as `replace()` or
        `replace(index)` and interpret the result as a boolean.

        Note: When `replace` is a callable, it may either:
        - take no arguments: `replace()`
        - take the dataset index: `replace(index)`
        In both cases, the return value must be interpretable as a boolean.

        Parameters
        ----------
        index: int
            Index of the dataset element.

        Returns
        -------
        bool
            `True` if the sample should be regenerated, `False` otherwise.

        Raises
        ------
        TypeError
            If `replace` is not a bool, float in [0, 1], or a callable
            returning a boolean.

        """

        if self.data[index] is None:
            return True

        if isinstance(self.replace, bool):
            return self.replace

        if callable(self.replace):
            replace_fn = cast(Callable[..., bool], self.replace)
            try:
                return bool(replace_fn())
            except TypeError:
                return bool(replace_fn(index))

        if isinstance(self.replace, (int, float)) and 0 <= self.replace <= 1:
            return bool(np.random.rand() < self.replace)

        raise TypeError(
            "The replace parameter must be a bool, a float in [0, 1], "
            "or a callable returning bool (optionally accepting index). "
            f"Got {self.replace!r} of type {type(self.replace).__name__}."
        )

    def __len__(
        self: Dataset,
    ) -> int:
        """Return the number of samples in the dataset.

        The length corresponds to the number of input elements provided
        during initialization, or the value of `length` if `inputs`
        was not explicitly given.

        Note: The dataset length is fixed at initialization and does not change
        even if samples are regenerated according to the replacement policy.

        Returns
        -------
        int
            The number of dataset elements.

        """

        return len(self.inputs)
