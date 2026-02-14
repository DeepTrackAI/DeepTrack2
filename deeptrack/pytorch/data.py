from __future__ import annotations

from typing import Any, Callable, Sequence

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
            else:
                inputs = [[]] * length
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
            res =  self.pipeline(self.inputs[index])
            if not isinstance(res, (tuple, list)):
                res = (res, )
            res = tuple(self._as_tensor(r) for r in res)

            # Convert all numpy arrays to torch tensors
            # res = tuple(self._as_tensor(r) for r in res)

            self.data[index] = res

        return self.data[index]
    
    def _as_tensor(
        self: Dataset,
        x: Any,
    ) -> torch.Tensor:
        if isinstance(x, (int, float, bool)):
            x = torch.from_numpy(np.array([x]))
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.ndim > 2 and x.dtype not in [np.uint8, np.uint16, np.uint32,
                                              np.uint64]:
                x = x.permute(-1, *range(x.ndim - 1))
        x = torch.Tensor(x)

        # if float, convert to torch default float
        if self.float_dtype and x.dtype in [torch.float16, torch.float32,
                                            torch.float64]:
            x = x.to(self.float_dtype)
        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            x = x.to(torch.long)

        return x

    def _should_replace(
        self: Dataset,
        index: int,
    ) -> bool:
        if self.data[index] is None:
            return True

        if isinstance(self.replace, bool):
            return self.replace
        elif callable(self.replace):
            try:
                return self.replace()
            except TypeError:
                return self.replace(index)
        elif isinstance(self.replace, float) and 0 <= self.replace <= 1:
            return np.random.rand() < self.replace
        else:
            raise TypeError(
                "replace must be a boolean, a float between 0 and 1, "
                "or a callable."
            )

    def __len__(
        self: Dataset,
    ) -> int:
        return len(self.inputs)
