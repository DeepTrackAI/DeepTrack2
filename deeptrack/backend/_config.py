__all__ = ["config", "cupy", "CUPY_AVAILABLE"]

import warnings
import numpy as cupy

CUPY_AVAILABLE = True
try:
    import cupy
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn(
        "cupy not installed. GPU-accelerated simulations will not be possible"
    )


class Config:
    def __init__(self):
        self.disable_gpu()
        self.enable_gpu()

    def enable_gpu(self):
        if CUPY_AVAILABLE:
            self.gpu_enabled = True
        else:
            warnings.warn("cupy not installed, CPU acceleration not enabled")

    def disable_gpu(self):
        self.gpu_enabled = False


config = Config()