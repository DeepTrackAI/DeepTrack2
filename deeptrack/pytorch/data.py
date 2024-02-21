import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
from deeptrack.image import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 pipeline,
                 inputs=None,
                 length=None,
                 replace: Union[bool, float] = False,
                 float_dtype: Optional[Union[torch.dtype, str]] = "default",
                 permute_channels: bool = False):
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

        self.permute_channels = permute_channels

    def __getitem__(self, index):
        if self._should_replace(index):
            self.pipeline.update()
            res =  self.pipeline(self.inputs[index])
            if not isinstance(res, (tuple, list)):
                res = (res, )
            res = tuple(res._value if isinstance(res, Image) else res for res in res)
            res = tuple(self._as_tensor(res) for res in res)

            # Convert all numpy arrays to torch tensors
            # res = tuple(self._as_tensor(r) for r in res)

            self.data[index] = res

        return self.data[index]
    
    def _as_tensor(self, x):
        if isinstance(x, (int, float, bool)):
            x = torch.from_numpy(np.array([x]))
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, Image):
            self._as_tensor(x._value)
        else:
            x = torch.Tensor(x)
        
        # if float, convert to torch default float
        if self.float_dtype and x.dtype in [torch.float16, torch.float32, torch.float64]:
            x = x.to(self.float_dtype)

        if self.permute_channels and x.dim() > 2:
            x = x.permute(-1, *range(0, x.dim() - 1))
        return x
    
    def _should_replace(self, index):
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
            raise TypeError("replace must be a boolean, a float between 0 and 1, or a callable.")
        
    def __len__(self):
        return len(self.inputs)
    