from __future__ import annotations

__all__ = ["config"]

import importlib
import warnings
import numpy as np
import array_api_compat as apc
from array_api_compat import numpy as apcnumpy
import array

import types, sys, numpy as _np, torch as _torch
from typing import *
import array_api_strict


class _Proxy(types.ModuleType):
    """Object to keep track of the current backend."""

    _backend: array_api_strict
    __name__: str

    def __init__(self, name: str):
        self._backend = apcnumpy
        self.__name__ = name

    def __getattr__(self, attribute):
        return getattr(self._backend, attribute)

    def __dir__(self):
        return dir(self._backend)


# TODO: once intersection types are available, use them here
xp: array_api_strict = _Proxy(
    __name__ + ".xp"
)  # Module instance  # the type is to make IDEs see this as if an array
sys.modules[xp.__name__] = (
    xp  # Register module name  # make the systems use this as a module
)


class NullContext:
    """A context manager that does nothing.

    Used when no context is needed, but the output expects
    a context manager."""

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class ImageWrapperContext:

    def __enter__(_):
        self.enable_image_wrapper()

    def __exit__(_, *args):
        self.disable_image_wrapper()


class Config:

    @property
    def gpu_enabled(self):
        return self.device == "gpu"

    def __init__(self):
        self.set_device("cpu")
        self.set_backend_numpy()
        self.disable_image_wrapper()


    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device

    def set_backend_numpy(self):
        self.set_backend("numpy")

    def set_backend_torch(self):
        self.set_backend("torch")

    def set_backend(self, backend: Literal["numpy", "torch"]):
        self.backend = backend
        xp._backend = importlib.import_module(f"array_api_compat.{backend}")

    def get_backend(self):
        return self.backend

    def disable_image_wrapper(self):
        self.image_wrapper = False

    def enable_image_wrapper(self):
        self.image_wrapper = True

    def wrapper_enabled_context(self):

        return ImageWrapperContext() if not self.image_wrapper else NullContext()

    def with_backend(self, backend: Literal["numpy", "torch"]):
        current_backend = self.backend
        if current_backend == backend:
            return NullContext()

        class BackendContext:
            def __enter__(_):
                self.set_backend(backend)

            def __exit__(_, *args):
                self.set_backend_numpy()

        return BackendContext()


config = Config()
