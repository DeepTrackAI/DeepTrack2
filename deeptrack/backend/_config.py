"""Configuration and backend management for DeepTrack2.

This module provides the core configuration class `Config` used to control the
computational backend (NumPy or PyTorch) and the device selection (CPU, GPU,
etc.).

Key Features
------------
- **Backend Selection and Management**

    It enables users to select and seamlessly switch between supported
    computational backends, including NumPy and PyTorch. This allows for
    backend-agnostic code and flexible pipeline design.

- **Device Control**

    It provides mechanisms to specify the computation device (e.g., CPU, GPU,
    or `torch.device`). This gives users fine-grained control over
    computational resources.

- **Context Managers for Temporary Configuration**

    Offers context managers that allow temporary changes to backend. This
    supports safe and readable experimental or context-specific configuration.

Module Structure
----------------
Classes:

- `Config`: Main configuration class for backend and device.

    It encapsulates methods to get/set backend and device, and provides a
    context manager for temporary configuration changes.

- `_Proxy`: Internal class to call proxy backend and correct array types.

    It forwards function calls to the current backend module (NumPy or PyTorch)
    and ensures arrays are created with the correct type and context.

Attributes:

- config: Config

    The default configuration object used by DeepTrack2. This singleton
    instance maintains global state for backend and device.

- xp: array_api_strict

    The currently active backend module. Provides the array API (NumPy,
    PyTorch) as selected by the user, via the `_Proxy` interface.

- TORCH_AVAILABLE: bool

    True if PyTorch (torch) is available, otherwise False. Used to control
    backend switching and PyTorch-specific features.

- DEEPLAY_AVAILABLE: bool

    True if Deeplay (deeplay) is available, otherwise False. Used to control
    backend switching and Deeplay-specific features.

- OPENCV_AVAILABLE: bool

    True if OpenCV (cv2) is available, otherwise False. Used for conditional
    logic when image processing requires OpenCV.

Examples
--------
IMPORTANT: Users should ensure backend and device compatibility.

Import the global config object and the xp proxy for backend-agnostic code:

>>> from deeptrack.backend import config, xp

Check the default backend and device:

>>> config.get_backend()
'numpy'

>>> config.get_device()
'cpu'

Use the xp proxy to create a NumPy array:

>>> array = xp.arange(5)
>>> type(array)
numpy.ndarray

Switch to the PyTorch backend and use GPU:

>>> config.set_backend_torch()
>>> config.get_backend()
'torch'

>>> config.set_device("cuda")
>>> config.get_device()
'cuda'

Create a tensor using the xp proxy:

>>> tensor = xp.arange(3)
>>> type(tensor)
torch.Tensor

Temporarily switch backends within a context manager:

>>> config.get_backend()
'torch'

>>> with config.with_backend("numpy"):
...     print(config.get_backend())
numpy

>>> config.get_backend()
'torch'

Use PyTorch-specific device objects if desired:

>>> import torch
>>>
>>> config.set_device(torch.device("cuda:0"))
>>> config.get_device()
device(type='cuda', index=0)

Check PyTorch availability:

>>> from deeptrack.backend import TORCH_AVAILABLE
>>>
>>> print(TORCH_AVAILABLE)

Check Deeplay availability:

>>> from deeptrack.backend import DEEPLAY_AVAILABLE
>>>
>>> print(DEEPLAY_AVAILABLE)

Check OpenCV availability:

>>> from deeptrack.backend import OPENCV_AVAILABLE
>>>
>>> print(OPENCV_AVAILABLE)

"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Literal, TYPE_CHECKING

from array_api_compat import numpy as apc_np
import array_api_strict


__all__ = [
    "config",
    "DEEPLAY_AVAILABLE",
    "OPENCV_AVAILABLE",
    "TORCH_AVAILABLE",
    "xp",
]


if TYPE_CHECKING:
    import torch


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import deeplay
    DEEPLAY_AVAILABLE = True
except ImportError:
    DEEPLAY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class _Proxy(types.ModuleType):
    """Keep track of current backend and forward calls to the correct backend.

    An instance of this object is treated as the module `xp`. It acts like a
    shallow wrapper around the actual backend (for example `numpy` or `torch`),
    forwarding calls to the correct backend.

    This is especially useful for array creation functions in order to ensure
    that the correct array type is created.

    This class is used internally within _config.py.

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

    Methods
    -------
    `set_backend(backend: types.ModuleType) -> None`
        Set the backend to use.

    `get_float_dtype(dtype: str) -> str`
        Get the float data type.

    `get_int_dtype(dtype: str) -> str`
        Get the int data type.

    `get_complex_dtype(dtype: str) -> str`
        Get the complex data type.

    `get_bool_dtype(dtype: str) -> str`
        Get the bool data type.

    `__getattr__(attribute: str) -> Any`
        Forward attribute access to the current backend.

    `__dir__() -> list[str]`
        List attributes of the current backend.

    Examples
    --------
    >>> from deeptrack.backend._config import _Proxy

    Create a proxy instance and set the backend to NumPy:

    >>> from array_api_compat import numpy as apc_np
    >>>
    >>> xp = _Proxy("numpy")
    >>> xp.set_backend(apc_np)

    Use the proxy to create an array (calls NumPy under the hood):

    >>> array = xp.arange(5)
    >>> array, type(array)
    array([0, 1, 2, 3, 4])
 
    >>> type(array)
    numpy.ndarray

    You can use any function or attribute provided by the backend:

    >>> ones_array = xp.ones((2, 2))

    Query dtypes in a backend-agnostic way:

    >>> xp.get_float_dtype()
    dtype('float64')
    
    >>> xp.get_int_dtype()
    dtype('int64')
    
    >>> xp.get_complex_dtype()
    dtype('complex128')

    
    >>> xp.get_bool_dtype()
    dtype('bool')

    Switch to the PyTorch backend:

    >>> from array_api_compat import torch as apc_torch
    >>>
    >>> xp = _Proxy("torch")
    >>> xp.set_backend(apc_torch)

    Now the proxy uses PyTorch:

    >>> tensor = xp.arange(5)
    >>> tensor
    tensor([0, 1, 2, 3, 4])

    >>> type(tensor)
    torch.Tensor

    The dtype helpers return PyTorch-specific types:

    >>> xp.get_float_dtype()
    torch.float32

    >>> xp.get_int_dtype()
    torch.int64

    >>> xp.get_complex_dtype()
    torch.complex64

    >>> xp.get_bool_dtype()
    torch.bool

    You can switch backends as often as needed.:

    >>> xp.set_backend(apc_np)
    >>> array = xp.arange(3)
    >>> type(array)
    numpy.ndarray

    """

    _backend: types.ModuleType  # array_api_strict
    __name__: str

    def __init__(
        self: _Proxy,
        name: str,
    ) -> None:
        """Initialize the _Proxy object.

        Parameters
        ----------
        name: str
            Name of the proxy object. This is used when printing the object.

        """

        self.set_backend(apc_np)
        self.__name__ = name

    def set_backend(
        self: _Proxy,
        backend: types.ModuleType,
    ) -> None:
        """Set the backend to use.

        Parameters
        ----------
        backend: types.ModuleType
            The backend to use.

        Examples
        --------
        >>> from deeptrack.backend._config import _Proxy
    
        Create a proxy instance and set the backend to NumPy:

        >>> from array_api_compat import numpy as apc_np
        >>>
        >>> xp = _Proxy("numpy")
        >>> xp.set_backend(apc_np)
        >>> array = xp.arange(5)
        >>> type(array)
        numpy.ndarray

        Now switch to a PyTorch backend:

        >>> from array_api_compat import torch as apc_torch
        >>>
        >>> xp = _Proxy("torch")
        >>> xp.set_backend(apc_torch)
        >>> tensor = xp.arange(5)
        >>> type(tensor)
        torch.Tensor

        """

        self._backend = backend
        self._backend_info = backend.__array_namespace_info__()

    def get_float_dtype(
        self: _Proxy,
        dtype: str = "default",
    ) -> str:
        """Get the float data type.

        Parameters
        ----------
        dtype: str, optional
            The floating-point data type to retrieve. If "default" (the
            default), returns the backend's default floating-point data type
            name. Otherwise, specify a valid floating-point data type key
            (e.g., "float32", "float64") to retrieve the corresponding type
            for the backend.

        Returns
        -------
        str
            The name of the floating data type for the current backend.
    
        Examples
        --------
        >>> from deeptrack.backend._config import _Proxy

        Create a proxy instance and set the backend to NumPy:

        >>> from array_api_compat import numpy as apc_np
        >>>
        >>> xp = _Proxy("numpy")
        >>> xp.set_backend(apc_np)

        >>> xp.get_float_dtype()
        dtype('float64')

        >>> xp.get_float_dtype("float32")
        dtype('float32')

        Now switch to a PyTorch backend:

        >>> from array_api_compat import torch as apc_torch
        >>>
        >>> xp = _Proxy("torch")
        >>> xp.set_backend(apc_torch)

        >>> xp.get_float_dtype()
        torch.float32

        >>> xp.get_float_dtype("float32")
        torch.float32

        """

        if dtype == "default":
            return self._backend_info.default_dtypes()["real floating"]

        return self._backend_info.dtypes(kind="real floating")[dtype]

    def get_int_dtype(
        self: _Proxy,
        dtype: str = "default",
    ) -> str:
        """Get the int data type.

        Parameters
        ----------
        dtype: str, optional
            The integer data type to retrieve. If "default" (the default),
            returns the backend's default integer data type name. Otherwise,
            specify a valid integer data type key (e.g., "int32", "int64") to
            retrieve the corresponding type for the backend.

        Returns
        -------
        str
            The name of the integer data type for the current backend.

        Examples
        --------
        >>> from deeptrack.backend._config import _Proxy

        Create a proxy instance and set the backend to NumPy:

        >>> from array_api_compat import numpy as apc_np
        >>>
        >>> xp = _Proxy("numpy")
        >>> xp.set_backend(apc_np)

        >>> xp.get_int_dtype()
        dtype('int64')

        >>> xp.get_int_dtype("int32")
        dtype('int32')

        Now switch to a PyTorch backend:

        >>> from array_api_compat import torch as apc_torch
        >>>
        >>> xp = _Proxy("torch")
        >>> xp.set_backend(apc_torch)

        >>> xp.get_int_dtype()
        torch.int64

        >>> xp.get_int_dtype("int32")
        torch.int32

        """

        if dtype == "default":
            return self._backend_info.default_dtypes()["integral"]

        return self._backend_info.dtypes(kind="integral")[dtype]

    def get_complex_dtype(
        self: _Proxy,
        dtype: str = "default",
    ) -> str:
        """Get the complex data type.

        Parameters
        ----------
        dtype: str, optional
            The complex data type to retrieve. If "default" (the default),
            returns the backend's default complex data type name. Otherwise,
            specify a valid complex data type key (e.g., "complex64",
            "complex128") to retrieve the corresponding type for the backend.

        Returns
        -------
        str
            The name of the complex data type for the current backend.

        Examples
        --------
        >>> from deeptrack.backend._config import _Proxy

        Create a proxy instance and set the backend to NumPy:

        >>> from array_api_compat import numpy as apc_np
        >>>
        >>> xp = _Proxy("numpy")
        >>> xp.set_backend(apc_np)

        >>> xp.get_complex_dtype()
        dtype('complex128')

        >>> xp.get_complex_dtype("complex64")
        dtype('complex64')

        Now switch to a PyTorch backend:

        >>> from array_api_compat import torch as apc_torch
        >>>
        >>> xp = _Proxy("torch")
        >>> xp.set_backend(apc_torch)

        >>> xp.get_complex_dtype()
        torch.complex64

        >>> xp.get_complex_dtype("complex64")
        torch.complex64

        """

        if dtype == "default":
            return self._backend_info.default_dtypes()["complex floating"]

        return self._backend_info.dtypes(kind="complex floating")[dtype]

    def get_bool_dtype(
        self: _Proxy,
        dtype: str = "default",
    ) -> str:
        """Get the bool data type.

        Parameters
        ----------
        dtype: str, optional
            The boolean data type to retrieve. If "default" (the default),
            returns the backend's default boolean data type name. Otherwise,
            specify a valid boolean data type key (usually "bool") to retrieve
            the corresponding type for the backend.

        Returns
        -------
        str
            The name of the boolean data type for the current backend.

        Examples
        --------
        >>> from deeptrack.backend._config import _Proxy

        Create a proxy instance and set the backend to NumPy:

        >>> from array_api_compat import numpy as apc_np
        >>>
        >>> xp = _Proxy("numpy")
        >>> xp.set_backend(apc_np)

        >>> xp.get_bool_dtype()
        dtype('bool')

        >>> xp.get_bool_dtype(dtype="bool")
        dtype('bool')

        Now switch to a PyTorch backend:

        >>> from array_api_compat import torch as apc_torch
        >>>
        >>> xp = _Proxy("torch")
        >>> xp.set_backend(apc_torch)

        >>> xp.get_bool_dtype()
        torch.bool

        >>> xp.get_bool_dtype(dtype="bool")
        torch.bool

        """

        if dtype == "default":
            dtype = "bool"

        return self._backend_info.dtypes(kind="bool")[dtype]

    def __getattr__(
        self: _Proxy,
        attribute: str,
    ) -> Any:
        """Forward attribute access to the current backend.

        Parameters
        ----------
        attribute: str
            The attribute name to retrieve from the backend.

        Returns
        -------
        Any
            The attribute from the current backend module.

        Examples
        --------
        >>> from deeptrack.backend._config import _Proxy

        Access NumPy's arange function transparently through the proxy:

        >>> from array_api_compat import numpy as apc_np
        >>>
        >>> xp = _Proxy("numpy")
        >>> xp.set_backend(apc_np)
        >>> xp.arange(4)
        array([0, 1, 2, 3])

        Now switch to a PyTorch backend:
    
        >>> from array_api_compat import torch as apc_torch
        >>>
        >>> xp = _Proxy("torch")
        >>> xp.set_backend(apc_torch)
        >>> xp.arange(4)
        tensor([0, 1, 2, 3])

        Analogously, you can access any attribute or function available in the
        current backend.

        """

        return getattr(self._backend, attribute)

    def __dir__(self: _Proxy) -> list[str]:
        """List attributes of the current backend.

        Returns
        -------
        list
            List of attribute names in the current backend module.

        Examples
        --------
        >>> from deeptrack.backend._config import _Proxy

        List the attributes (functions, constants, etc.) in the NumPy backend:
    
        >>> from array_api_compat import numpy as apc_np
        >>>
        >>> xp = _Proxy("numpy")
        >>> xp.set_backend(apc_np)
        >>> dir(xp)
        ['ALLOW_THREADS',
        ...]

        List the attributes in the PyTorch backend:
    
        >>> from array_api_compat import torch as apc_torch
        >>>
        >>> xp = _Proxy("torch")
        >>> xp.set_backend(apc_torch)
        >>> dir(xp)
        ['AVG',
        ...]

        """

        return dir(self._backend)


# TODO: Once intersection types are available, use them here.
# Intersection types are in the pipeline for python 3.13 or 3.14. They let you
# define types that are the combination of many subtypes. So Intersection[A, B]
# would have all the properties of A and B. Here, it would let us define
# exactly the type of xp as Intersection[_Proxy, apc_np, apc_torch].


# This creates the xp object, which we will use a module.
# We assign the type to be `array_api_strict` to make IDEs see this as if it
# were an array API module, instead of the wrapper _Proxy object.
xp: array_api_strict = _Proxy(__name__ + ".xp")

# This registers the xp object as a module. This should make import statements
# treat xp as a module.
sys.modules[xp.__name__] = xp


class Config:
    """Configuration object for managing backend and device settings.

    This class manages the backend (such as NumPy or PyTorch) and the computing
    device (such as CPU, GPU, or torch.device). It provides methods for
    switching between backends and devices.

    Attributes
    ----------
    device: str | torch.device
        The currently set device for computation.
    backend: "numpy" or "torch"
        The currently active backend.

    Methods
    -------
    `set_device(device: str | torch.device) -> None`
        Set the device to use.

    `get_device() -> str | torch.device`
        Get the device to use.

    `set_backend_numpy() -> None`
        Set the backend to NumPy.

    `set_backend_torch() -> None`
        Set the backend to PyTorch.

    `def set_backend(backend: Literal["numpy", "torch"]) -> None`
        Set the backend to use for array operations.

    `get_backend() -> Literal["numpy", "torch"]`
        Get the current backend.

    `with_backend(context_backend: Literal["numpy", "torch"]) -> object`
        Return a context manager that temporarily changes the backend.

    Examples
    --------
    IMPORTANT: Users should ensure backend and device compatibility.

    Create the singleton configuration object and check its defaults:

    >>> from deeptrack.backend import config

    >>> config.get_backend()
    'numpy'

    >>> config.get_device()
    'cpu'

    Set the backend to PyTorch and device to GPU:

    >>> config.set_backend_torch()
    >>> config.get_backend()
    'torch'

    >>> config.set_device("cuda")
    >>> config.get_device()
    'cuda'

    Use the xp proxy to create arrays/tensors:

    >>> from deeptrack.backend import xp

    >>> config.set_backend_numpy()
    >>> array = xp.arange(5)
    >>> type(array)
    numpy.ndarray

    >>> config.set_backend_torch()
    >>> tensor = xp.arange(5)
    >>> type(tensor)
    torch.Tensor

    Temporarily switch backend using a context manager:

    >>> config.set_backend("torch")
    >>> config.get_backend()
    'torch'

    >>> with config.with_backend("numpy"):
    ...     print(config.get_backend())
    numpy

    >>> config.get_backend()
    'torch'

    Use a torch.device object directly:

    >>> import torch
    >>>
    >>> config.set_backend_torch()
    >>> config.set_device(torch.device("cuda:0"))
    >>> config.get_device()
    device(type='cuda', index=0)

    """

    device: str | torch.device
    backend: Literal["numpy", "torch"]

    def __init__(self: Config) -> None:
        """Initialize the configuration with default values.

        By default, it sets the device to "cpu" and the backend to "numpy".

        """

        self.set_device("cpu")
        self.set_backend_numpy()

    def set_device(
        self: Config,
        device: str | torch.device,
    ) -> None:
        """Set the device to use.

        It can be a string, most typically "cpu", "gpu", "cuda", "mps", or
        torch.device. In any case, it needs to be used with a compatible
        backend.

        It can only be "cpu" when using NumPy backend.

        Parameters
        ----------
        device: str or torch.device
            The device to use.

        Examples
        --------
        IMPORTANT: Users should ensure backend and device compatibility.

        Import the singleton configuration object:

        >>> from deeptrack.backend import config

        Set device to CPU (works with both NumPy and PyTorch backends):

        >>> config.set_device("cpu")
        >>> config.get_device()
        'cpu'

        Set device to GPU (requires PyTorch backend):

        >>> config.set_backend_torch()
        >>> config.set_device("cuda")
        >>> config.get_device()
        'cuda'

        Use a specific CUDA device (PyTorch backend):

        >>> import torch
        >>>
        >>> config.set_backend_torch()
        >>> config.set_device(torch.device("cuda:0"))
        >>> config.get_device()
        device(type='cuda', index=0)

        Set device to Apple Silicon GPU (PyTorch backend on Macs):

        >>> config.set_backend_torch()
        >>> config.set_device("mps")
        >>> config.get_device()
        'mps'

        Attempting to set a GPU device with NumPy backend (should be avoided):

        >>> config.set_backend_numpy()
        >>> config.set_device("cuda")
        >>> config.get_device()
        'cuda'

        Computation will still run on CPU, since NumPy does not support GPU.

        """

        self.device = device

    def get_device(self: Config) -> str | torch.device:
        """Get the device to use.

        Returns
        -------
        str or torch.device
            The device to use. It can be a string, most typically "cpu", "gpu",
            "cuda", "mps", or torch.device. In any case, it needs to be used
            with a compatible backend.

        Examples
        --------
        Import the singleton configuration object:

        >>> from deeptrack.backend import config

        Get the current device:

        >>> device = config.get_device()

        """

        return self.device

    def set_backend_numpy(self: Config) -> None:
        """Set the backend to NumPy.

        Examples
        --------
        Import the singleton configuration object:

        >>> from deeptrack.backend import config

        Set the backend to NumPy:

        >>> config.set_backend_numpy()
        >>> config.get_backend()
        'numpy'

        NumPy backend enables use of standard NumPy arrays via the xp proxy:

        >>> from deeptrack.backend import xp
        >>>
        >>> array = xp.arange(5)
        >>> type(array)
        numpy.ndarray
    
        """

        self.set_backend("numpy")

    def set_backend_torch(self: Config) -> None:
        """Set the backend to PyTorch.

        Examples
        --------
        Import the singleton configuration object:

        >>> from deeptrack.backend import config

        Set the backend to PyTorch:

        >>> config.set_backend_torch()
        >>> config.get_backend()
        'torch'

        PyTorch backend enables use of PyTorch tensors via the xp proxy:

        >>> from deeptrack.backend import xp
        >>>
        >>> tensor = xp.arange(5)
        >>> type(tensor)
        torch.Tensor

        """

        self.set_backend("torch")

    def set_backend(
        self: Config,
        backend: Literal["numpy", "torch"],
    ) -> None:
        """Set the backend to use for array operations.

        Parameters
        ----------
        backend : "numpy" or "torch"
            The backend to use for array operations.

        Examples
        --------
        Import the singleton configuration object:

        >>> from deeptrack.backend import config

        Set the backend to NumPy:

        >>> config.set_backend("numpy")
        >>> config.get_backend()
        'numpy'

        Set the backend to PyTorch:

        >>> config.set_backend("torch")
        >>> config.get_backend()
        'torch'

        Switch between backends as needed in your workflow using the xp proxy:

        >>> from deeptrack.backend import xp

        >>> config.set_backend("numpy")
        >>> array = xp.arange(4)
        >>> type(array)
        numpy.ndarray

        >>> config.set_backend("torch")
        >>> tensor = xp.arange(4)
        >>> type(tensor)
        torch.Tensor
    
        """

        # This import is only necessary when using the torch backend.
        if backend == "torch":
            # pylint: disable=import-outside-toplevel,unused-import
            # flake8: noqa: E402
            from deeptrack.backend import array_api_compat_ext

        self.backend = backend
        xp.set_backend(importlib.import_module(f"array_api_compat.{backend}"))

    def get_backend(self: Config) -> Literal["numpy", "torch"]:
        """Get the current backend.

        Returns
        -------
        "numpy" or "torch"
            The backend currently in use, "numpy" or "torch".

        Examples
        --------
        Import the singleton configuration object:

        >>> from deeptrack.backend import config

        Get the current backend:

        >>> backend = config.get_backend()

        """

        return self.backend

    def with_backend(
        self: Config,
        context_backend: Literal["numpy", "torch"],
    ) -> object:
        """Return a context manager that temporarily changes the backend.

        The backend is switched to the specified backend upon entering the
        context, and restored to the previous backend upon exiting.

        Parameters
        ----------
        context_backend: "numpy" | "torch"
            The backend to temporarily use within the context.

        Returns
        -------
        object
            A context manager that switches the backend.

        Examples
        --------
        Import the singleton configuration object:

        >>> from deeptrack.backend import config

        Temporarily switch to the NumPy backend for a block of code:

        >>> config.set_backend("torch")
        >>> config.get_backend()
        'torch'

        >>> with config.with_backend("numpy"):
        ...     print(config.get_backend())
        numpy

        >>> config.get_backend()
        'torch'

        Temporarily switch to the PyTorch backend inside a function:

        >>> from deeptrack.backend import xp

        >>> config.set_backend("numpy")config.set_backend("numpy")

        >>> def do_torch_operation():
        ...     with config.with_backend("torch"):
        ...         return xp.arange(3)

        >>> tensor = do_torch_operation()
        >>> type(tensor)
        torch.Tensor

        >>> config.get_backend()
        'numpy'
    
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
