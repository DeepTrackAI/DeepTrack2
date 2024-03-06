from deeptrack.features import Feature
from deeptrack.backend import config
import torch
import numpy as np
from typing import Literal

class ToTensor(Feature):

    def __init__(self, 
                 dtype=None, 
                 device=None, 
                 add_dim_to_number=False, 
                 permute_mode: Literal["always", "never", "numpy", "numpy_and_not_int"] = "never",
                 **kwargs):
        """Converts the input to a torch tensor.
        
        Parameters
        ----------
        dtype : torch.dtype, optional
            The dtype of the resulting tensor. If None, the dtype is inferred from the input.
        device : torch.device, optional
            The device of the resulting tensor. If None, the device is inferred from the input.
        add_dim_to_number : bool, optional
            If True, a dimension is added to single numbers. This is useful when the input is a
            single number, but the output should be a tensor with a single dimension.
            Default value is False.
        permute_mode : {"always", "never", "numpy", "numpy_and_not_int"}, optional
            Whether to permute the input to channel first. If "always", the input is always permuted.
            If "never", the input is never permuted. If "numpy", the input is permuted if it is a numpy
            array. If "numpy_and_not_int", the input is permuted if it is a numpy array and the dtype
            is not an integer.
        """
        super().__init__(dtype=dtype, device=device, add_dim_to_number=add_dim_to_number, permute_mode=permute_mode, **kwargs)

    def get(self, x, dtype, device, add_dim_to_number, permute_mode, **kwargs):

        is_numpy = isinstance(x, np.ndarray)

        dtype = dtype or x.dtype
        if isinstance(x, torch.Tensor):
            ...
        elif isinstance(x, np.ndarray):
            if any(stride < 0 for stride in x.strides):
                x = x.copy()
            x = torch.from_numpy(x)
        elif isinstance(x, (int, float, bool, complex)):
            if add_dim_to_number:
                x = torch.tensor([x])
            else:
                return x
        else:
            x = torch.Tensor(x)

        if (
            permute_mode == "always"
            or (permute_mode == "numpy" and is_numpy)
            or (permute_mode == "numpy_and_not_int" and is_numpy and dtype not in [torch.int8, torch.int16, torch.int32, torch.int64])
        ):
            x = x.permute(-1, *range(x.dim() - 1))
        if dtype:
            x = x.to(dtype)
        if device:
            x = x.to(device)

        
        return x