from __future__ import annotations

__all__ = ["config", "cupy", "CUPY_AVAILABLE"]

import importlib
import warnings
import numpy as np
import array_api_compat as apc
from array_api_compat import numpy as apcnumpy
import array

import types, sys, numpy as _np, torch as _torch
from typing import *
import array_api_strict

# TODO: remove the need for this (errors when removed)
cupy = np

CUPY_AVAILABLE = True
try:
    import cupy
except ImportError:
    CUPY_AVAILABLE = False


class _Proxy(types.ModuleType):
    """Object to keep track of the current backend, and forward calls to the correct backend.

    An instance of this object will be treated as the module `xp`. It acts like a
    shallow wrapper around the actual backend (for example `numpy` or `torch`),
    and forwards calls to the correct backend.

    This is especially useful for array creation functions, to ensure that the correct
    array type is created.

    Parameters
    ----------
    name : str
        Name of the proxy object. This is used when printing the object.

    Attributes
    ----------
    _backend : backend modukle
        The actual backend module.
    __name__ : str
        The name of the proxy object.
    """

    _backend: types.ModuleType
    __name__: str

    def __init__(self, name: str):
        self._backend = apcnumpy
        self.__name__ = name

    def __getattr__(self, attribute):
        return getattr(self._backend, attribute)

    def __dir__(self):
        return dir(self._backend)


# TODO: once intersection types are available, use them here

# This creates the xp object, which we will use a module.
# We assign the type to be `array_api_strict` to make IDEs see this as if it were
# an array api module, instead of the wrapper _Proxy object.
xp: array_api_strict = _Proxy(__name__ + ".xp")

# This registers the xp object as a module. This should make import statements
# treat xp as a module.
sys.modules[xp.__name__] = xp


class NullContext:
    """A context manager that does nothing.

    Used when no context is needed, but the output expects
    a context manager."""

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class ImageWrapperContext:
    """A context manager that enables the image wrapper.

    Example
    -------
    >>> pipeline = dt.Value(1)
    >>> normal_result = pipeline()
    >>> with ImageWrapperContext():
    ...     wrapped_result = pipeline()
    ...
    >>> print(normal_result) # 1
    >>> print(wrapped_result) # Image(1)
    """

    def __enter__(self, config: Config):
        config.enable_image_wrapper()

    def __exit__(self, *args):
        config.disable_image_wrapper()


class Config:

    @property
    def gpu_enabled(self):
        return self.device == "gpu"

    def __init__(self):
        self.set_device("cpu")
        self.set_backend_numpy()
        self.disable_image_wrapper()

    def enable_gpu(self):
        warnings.warn(
            "(enable/disable)_gpu is deprecated. Use set_device instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if CUPY_AVAILABLE:
            self.device = "gpu"
        else:
            warnings.warn("cupy not installed, CPU acceleration not enabled")

    def disable_gpu(self):
        warnings.warn(
            "(enable/disable)_gpu is deprecated. Use set_device instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.device = "cpu"

    def set_device(self, device):
        """Set the device to use.

        Can be ["cpu", "gpu", "cuda", "mps", torch.device],
        but needs to be used with a compatible backend. Can only be "cpu"
        if using numpy backend.

        Parameters
        ----------
        device : str
            The device to use.
        """
        self.device = device

    def get_device(self):
        """Get the device to use.

        Returns
        -------
        str
            The device to use.
        """
        return self.device

    def set_backend_numpy(self):
        """Set the backend to numpy."""
        self.set_backend("numpy")

    def set_backend_cupy(self):
        self.set_backend("cupy")

    def set_backend_torch(self):
        """Set the backend to torch."""
        self.set_backend("torch")

    def set_backend(self, backend: Literal["numpy", "cupy", "torch"]):
        """Set the backend to use.

        One of ["numpy", "torch"].

        Parameters
        ----------
        backend : str
            The backend to use.
        """
        self.backend = backend
        xp._backend = importlib.import_module(f"array_api_compat.{backend}")

    def get_backend(self):
        """Get the current backend."""
        return self.backend

    def disable_image_wrapper(self):
        """Disable the image wrapper.

        This will ensure that `Image` objects are not used."""
        self.image_wrapper = False

    def enable_image_wrapper(self):
        """Enable the image wrapper.

        This will ensure that `Image` objects are used."""
        self.image_wrapper = True

    def wrapper_enabled_context(self):
        """Return a context manager that enables the image wrapper.

        This will ensure that `Image` objects are used.

        Examples
        --------
        >>> pipeline = dt.Value(1)
        >>> with config.wrapper_enabled_context():
        ...     result = pipeline()
        >>> print(result) # Image(1)
        """
        return ImageWrapperContext(self) if not self.image_wrapper else NullContext()

    def with_backend(self, backend: Literal["numpy", "torch"]):
        """Return a context manager that changes the backend."""
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
