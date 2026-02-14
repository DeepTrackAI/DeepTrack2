from __future__ import annotations

from typing import Any, cast, Callable, Sequence

import numpy as np
import torch

from deeptrack import Feature


__all__ = ["Dataset"]


class Dataset(torch.utils.data.Dataset):

    pipeline: Feature
    replace: bool | float | Callable[[], bool] | Callable[[int], bool]
    inputs: Sequence[Any]
    data: list[tuple[Any, ...]]
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

        if self._should_replace(index):
            self.pipeline.update()
            result = self.pipeline(self.inputs[index])
            if not isinstance(result, (tuple, list)):
                result = (result, )
            result = tuple(self._as_tensor(r) for r in result)

            self.data[index] = result

        return self.data[index]
    
    def _as_tensor(
        self: Dataset,
        x: (
            torch.Tensor
            | np.ndarray
            | int
            | float
            | bool
            | complex
            | Sequence[int | float | bool | complex]
            | Any
        ),
    ) -> torch.Tensor:

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

        if self.float_dtype is not None and tensor.dtype in (
            torch.float16, torch.float32, torch.float64,
        ):
            tensor = tensor.to(self.float_dtype)

        if tensor.is_floating_point():
            tensor = tensor.to(torch.long)

        return tensor

    def _should_replace(
        self: Dataset,
        index: int,
    ) -> bool:

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

        if isinstance(self.replace, float) and 0 <= self.replace <= 1:
            return bool(np.random.rand() < self.replace)

        raise TypeError(
            "The replace parameter must be a bool, a float in [0, 1], "
            "or a callable returning bool (optionally accepting index). "
            f"Got {self.replace!r} of type {type(self.replace).__name__}."
        )

    def __len__(
        self: Dataset,
    ) -> int:

        return len(self.inputs)
