__all__ = ["config", "cupy", "CUPY_AVAILABLE"]

import warnings
import numpy as cupy
from typing import *

CUPY_AVAILABLE = True
try:
    import cupy
except ImportError:
    CUPY_AVAILABLE = False



class Config:

    @property
    def gpu_enabled(self):
        return self.device == "gpu"
    

    def __init__(self):
        self.set_device("cpu")
        self.set_backend_numpy()
        self.disable_image_wrapper()

    def enable_gpu(self):
        warnings.warn("(enable/disable)_gpu is deprecated. Use set_device instead", DeprecationWarning, stacklevel=2)
        if CUPY_AVAILABLE:
            self.device = "gpu"
        else:
            warnings.warn("cupy not installed, CPU acceleration not enabled")

    def disable_gpu(self):
        warnings.warn("(enable/disable)_gpu is deprecated. Use set_device instead", DeprecationWarning, stacklevel=2)
        self.device = "cpu"

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device
    
    def set_backend_numpy(self):
        self.set_backend("numpy")
    
    def set_backend_cupy(self):
        self.set_backend("cupy")

    def set_backend_torch(self):
        self.set_backend("torch")

    def set_backend(self, backend: Literal["numpy", "cupy", "torch"]):
        self.backend = backend

    def get_backend(self):
        return self.backend
    
    def disable_image_wrapper(self):
        self.image_wrapper = False

    def enable_image_wrapper(self):
        self.image_wrapper = True

    def wrapper_enabled_context(self):
        class NullContext:
            def __enter__(self):
                pass
            def __exit__(self, *args):
                pass

        class ImageWrapperContext:
            def __enter__(_):
                self.enable_image_wrapper()
            def __exit__(_, *args):
                self.disable_image_wrapper()
        
        return ImageWrapperContext() if not self.image_wrapper else NullContext()
        

config = Config()