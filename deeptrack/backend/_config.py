"""Configuration and backend management for DeepTrack.

This module provides the core configuration class and context managers used to
control the computational backend (NumPy, PyTorch, or others), device selection
(CPU, GPU, etc.), and image wrapper behavior for DeepTrack pipelines.

The main entry point is the `Config` class, which allows you to:
- Select the array backend for computation (e.g., NumPy or PyTorch)
- Specify the device to run on (CPU, GPU, or torch.device)
- Control whether outputs are wrapped as Image objects

Additional context managers such as `NullContext` and `ImageWrapperContext`
are provided to facilitate temporary changes to configuration and to enable or
disable image wrapping within a code block.

Classes
-------
Config
    Main configuration class for DeepTrack backend, device, and image wrapper.
_Proxy
    Internal class used to proxy backend calls and ensure correct array types.

Attributes
----------
config : Config
    The default configuration object used by DeepTrack.
xp : module
    The currently active backend module (NumPy, PyTorch, etc.).

Examples
--------
Set the backend to PyTorch and use the GPU:

>>> config.set_backend_torch()
>>> config.set_device("cuda")

Temporarily enable the image wrapper within a context:

>>> with ImageWrapperContext(config):
...     result = some_pipeline()
>>> # Image wrapper state is automatically restored on exit

Switch backend temporarily within a context:

>>> with config.with_backend("numpy"):
...     result = some_numpy_operation()

"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Literal, TYPE_CHECKING

from array_api_compat import numpy as apcnumpy
import array_api_strict


__all__ = ["config"]


if TYPE_CHECKING:
    import torch


class _Proxy(types.ModuleType):
    """Keep track of current backend and forward calls to the correct backend.

    An instance of this object will be treated as the module `xp`. It acts like
    a shallow wrapper around the actual backend (for example `numpy` or 
    `torch`), and forwards calls to the correct backend.

    This is especially useful for array creation functions, to ensure that the
    correct array type is created.

    Parameters
    ----------
    name: str
        Name of the proxy object. This is used when printing the object.

    Attributes
    ----------
    _backend: backend module
        The actual backend module.
    __name__: str
        The name of the proxy object.

    """

    _backend: types.ModuleType  # array_api_strict
    __name__: str

    def __init__(self: _Proxy, name: str) -> None:
        """Initialize the _Proxy object.

        Parameters
        ----------
        name: str
            Name of the proxy object. This is used when printing the object.

        """

        self._backend = apcnumpy
        self.__name__ = name

    def __getattr__(self: _Proxy, attribute: str) -> Any:
        """Forward attribute access to the current backend.

        Parameters
        ----------
        attribute: str
            The attribute name to retrieve from the backend.

        Returns
        -------
        Any
            The attribute from the current backend module.

        """

        return getattr(self._backend, attribute)

    def __dir__(self: _Proxy) -> list[str]:
        """List attributes of the current backend.

        Returns
        -------
        list
            List of attribute names in the current backend module.

        """

        return dir(self._backend)


# TODO: Once intersection types are available, use them here.
# Intersection types are in the pipeline for python 3.13 or 3.14. They let you
# define types that are the combination of many subtypes. So Intersection[A, B]
# would have all the properties of A and B. Here, it would let us define
# exactly the type of xp as Intersection[_Proxy, apcnumpy, apctorch].


# This creates the xp object, which we will use a module.
# We assign the type to be `array_api_strict` to make IDEs see this as if it
# were an array API module, instead of the wrapper _Proxy object.
xp: array_api_strict = _Proxy(__name__ + ".xp")

# This registers the xp object as a module. This should make import statements
# treat xp as a module.
sys.modules[xp.__name__] = xp


class Config:
    """Configuration object for managing backend and device settings.

    This class manages the backend (such as NumPy or PyTorch), the computing
    device (such as CPU, GPU, or torch.device), and whether the image wrapper
    is enabled. It provides methods for switching between backends and devices,
    and for enabling or disabling the image wrapper.

    Attributes
    ----------
    device: str | torch.device
        The currently set device for computation.
    backend: "numpy" | "torch"
        The currently active backend.
    image_wrapper: bool
        Whether the image wrapper is enabled.

    """

    device: str | torch.device
    backend: Literal["numpy", "torch"]
    image_wrapper: bool

    @property
    def gpu_enabled(self: Config) -> bool:
        """Check if the current device is GPU.

        Returns
        -------
        bool
            True if the current device is "gpu", otherwise False.

        """

        return self.device == "gpu"

    def __init__(self: Config) -> None:
        """Initializes the configuration with default values.

        It sets the device to "cpu", the backend to "numpy", and disables the
        image wrapper.

        """

        self.set_device("cpu")
        self.set_backend_numpy()
        self.disable_image_wrapper()

    def set_device(
        self: Config,
        device: str | torch.device,
    ) -> None:
        """Set the device to use.

        Can be a string, most typically "cpu", "gpu", "cuda", "mps", or
        torch.device. In any case, it needs to be used with a compatible
        backend.
        
        It can only be "cpu" when using NumPy backend.

        Parameters
        ----------
        device: str | torch.device
            The device to use.

        """

        self.device = device

    def get_device(self: Config) -> str | torch.device:
        """Get the device to use.

        Returns
        -------
        str | torch.device
            The device to use.

        """

        return self.device

    def set_backend_numpy(self):
        """Set the backend to numpy."""

        self.set_backend("numpy")

    def set_backend_torch(self):
        """Set the backend to torch."""

        self.set_backend("torch")

    def set_backend(
        self: Config,
        backend: Literal["numpy", "torch"],
    ) -> None:
        """Set the backend to use for array operations.

        Parameters
        ----------
        backend : "numpy" | "torch"
            The backend to use for array operations.

        """

        # This import is only necessary when using the torch backend.
        if backend == "torch":
            # pylint: disable=import-outside-toplevel,unused-import
            # flake8: noqa: E402
            from deeptrack.backend import array_api_compat_ext

        self.backend = backend
        xp._backend = importlib.import_module(f"array_api_compat.{backend}")

    def get_backend(self: Config) -> Literal["numpy", "torch"]:
        """Get the current backend.

        Returns
        -------
        str
            The backend currently in use, "numpy" or "torch".

        """

        return self.backend

    def disable_image_wrapper(self: Config) -> None:
        """Disable the image wrapper.

        When disabled, `Image` objects are not used for wrapping outputs.

        """

        self.image_wrapper = False

    def enable_image_wrapper(self: Config) -> None:
        """Enable the image wrapper.

        When enabled, outputs are wrapped as `Image` objects.

        """

        self.image_wrapper = True

    def with_backend(
        self: Config,
        context_backend: Literal["numpy", "torch"],
    ) -> object:
        """Return a context manager that temporarily changes the backend.

        The backend is switched to the specified backend upon entering the
        context, and restored to the previous backend upon exiting.

        Parameters
        ----------
        context_backend : "numpy" | "torch"
            The backend to temporarily use within the context.

        Returns
        -------
        object
            A context manager that switches the backend.

        """

        self_backend = self.backend

        class BackendContext:

            def __enter__(_):
                if self_backend != context_backend:
                    self.set_backend(context_backend)

            def __exit__(_, *args):
                if self_backend != context_backend:
                    self.set_backend(self_backend)

        return BackendContext()


config = Config()
