from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Literal

from array_api_compat import numpy as apcnumpy
import array_api_strict


__all__ = ["config"]


class _Proxy(types.ModuleType):
    """Keep track of current backend and forward calls to the correct backend.

    An instance of this object will be treated as the module `xp`. It acts like
    a shallow wrapper around the actual backend (for example `numpy` or 
    `torch`), and forwards calls to the correct backend.

    This is especially useful for array creation functions, to ensure that the
    correct array type is created.

    Parameters
    ----------
    name : str
        Name of the proxy object. This is used when printing the object.

    Attributes
    ----------
    _backend : backend module
        The actual backend module.
    __name__ : str
        The name of the proxy object.

    """

    _backend: types.ModuleType  # array_api_strict
    __name__: str

    def __init__(self: _Proxy, name: str) -> None:
        """Initialize the _Proxy object.

        Parameters
        ----------
        name : str
            Name of the proxy object. This is used when printing the object.

        """

        self._backend = apcnumpy
        self.__name__ = name

    def __getattr__(self: _Proxy, attribute: str) -> Any:
        """Forward attribute access to the current backend.

        Parameters
        ----------
        attribute : str
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


class NullContext:
    """A context manager that does nothing.

    Used when no context is needed, but the output expects a context manager.

    Examples
    --------
    >>> with NullContext():
    ...     print("No special context is active.")

    """

    def __enter__(self: NullContext) -> None:
        """Enter the runtime context related to this object."""
        pass

    def __exit__(
        self: NullContext,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Exit the runtime context related to this object."""
        pass


class Config:

    @property
    def gpu_enabled(self):
        return self.device == "gpu"

    def __init__(self):
        self.set_device("cpu")
        self.set_backend_numpy()
        self.disable_image_wrapper()

    def set_device(self: Config, device) -> None:
        """Set the device to use.

        Can be "cpu", "gpu", "cuda", "mps", torch.device, but needs to be
        used with a compatible backend.
        
        It can only be "cpu" if using NumPy backend.

        Parameters
        ----------
        device : str
            The device to use.

        """

        self.device = device

    def get_device(self: Config) -> str:
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

        if backend == "torch":
            # pylint: disable=import-outside-toplevel,unused-import
            # flake8: noqa: E402
            from deeptrack.backend import array_api_compat_ext

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
