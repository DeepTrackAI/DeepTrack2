"""Elementwise mathematical operations for DeepTrack features.

This module defines a collection of `Feature` classes that apply mathematical
functions elementwise to arrays or tensors in a DeepTrack2 pipeline. These
operations are backend-agnostic and compatible with both NumPy and PyTorch.

Elementwise features can be created in two ways:

1. **Using the Factory Function (`create_elementwise_class`)**

   For most functions that are available in both NumPy and PyTorch and are
   supported by the `array-api-compat` backend abstraction (`xp`), the class
   can be generated dynamically using the `create_elementwise_class` factory.

   For example:

   >>> from deeptrack.backend import xp
   >>> from deeptrack.elementwise import create_elementwise_class
   >>>
   >>> Abs = create_elementwise_class("Abs", xp.abs)

   This creates a `Feature` class named `Abs` that applies `abs()` to the
   input elementwise, supporting both direct and pipeline usage.

2. **Defining a Custom Subclass of `ElementwiseFeature`**

   In cases where the `array-api-compat` implementation fails or does not
   support the required function for a backend (e.g., `torch.float32` with
   `xp.floor`), a custom subclass of `ElementwiseFeature` can be defined
   explicitly.

   These subclasses manually dispatch to the appropriate backend function 
   (e.g., `torch.floor`, `np.floor`) depending on the input type and device,
   ensuring robust and backend-safe behavior.

   For example, `Floor`, `Ceil`, `Imag`, and `Sign` are implemented this way.

This dual mechanism provides both flexibility and robustness for applying
mathematical operations in a pipeline-agnostic, extensible, and modular way.

Key Features
------------
- **Seamless Backend Compatibility**

    Most functions use `array-api-compat` (`xp`) to ensure compatibility with
    both NumPy and PyTorch backends. Manual dispatch is used for cases where
    `xp` fails (e.g., `ceil`, `floor`, `imag`, `sign`).

- **Factory-Generated and Manual Implementations**

    Elementwise operations are implemented using either:
    - `create_elementwise_class()` for standard backend-agnostic functions.
    - Dedicated subclasses of `ElementwiseFeature` for operations requiring
      manual backend dispatch.

- **Supports Direct and Pipeline Composition**

    These features can be applied directly to NumPy arrays or PyTorch tensors,
    or used in combination with other DeepTrack `Feature` objects in pipelines.

- **Extensive Documentation and Examples**

    Each class includes detailed docstrings with usage examples for both
    backends and for pipeline integration.

Module Structure
----------------
Classes:

- `ElementwiseFeature`
    
    Base class for features that apply mathematical operations elementwise to
    NumPy arrays or PyTorch tensors. Accepts a function and an optional
    input `Feature`.

Functions:

- `create_elementwise_class(name, function, docstring) -> type`

    Factory function that returns a new subclass of `ElementwiseFeature` with
    the given `name` and `function`. Automatically sets the class name,
    module, and docstring for full introspection and documentation support.

Elementwise Features
--------------------

All elementwise features inherit from `ElementwiseFeature` and apply a
mathematical operation elementwise to the output of another `Feature`, or
directly to an input array or tensor.

The following features are available:

Trigonometric Functions:
- `Sin`: Applies the sine function `sin(x)` elementwise.
- `Cos`: Applies the cosine function `cos(x)` elementwise.
- `Tan`: Applies the tangent function `tan(x)` elementwise.

Inverse Trigonometric Functions:
- `Arcsin`: Applies the arcsine function `arcsin(x)` elementwise.
- `Arctan`: Applies the arctangent function `arctan(x)` elementwise.

Hyperbolic Functions:
- `Sinh`: Applies the hyperbolic sine function `sinh(x)` elementwise.
- `Cosh`: Applies the hyperbolic cosine function `cosh(x)` elementwise.
- `Tanh`: Applies the hyperbolic tangent function `tanh(x)` elementwise.

Inverse Hyperbolic Functions:
- `Arcsinh`: Applies the inverse hyperbolic sine `arcsinh(x)` elementwise.
- `Arccosh`: Applies the inverse hyperbolic cosine `arccosh(x)` elementwise.
- `Arctanh`: Applies the inverse hyperbolic tangent `arctanh(x)` elementwise.

Rounding Functions:
- `Round`: Applies nearest-integer rounding elementwise.
- `Floor`: Applies floor function `floor(x)` elementwise.
- `Ceil`: Applies ceil function `ceil(x)` elementwise.

Exponential and Logarithmic Functions:
- `Exp`: Applies the exponential function `exp(x)` elementwise.
- `Log`: Applies the natural logarithm `log(x)` elementwise.
- `Log10`: Applies the base-10 logarithm `log10(x)` elementwise.
- `Log2`: Applies the base-2 logarithm `log2(x)` elementwise.

Complex Number Functions:
- `Angle`: Returns the phase angle `angle(x)` of complex inputs.
- `Real`: Extracts the real part `real(x)` of complex inputs.
- `Imag`: Extracts the imaginary part `imag(x)`; returns zero for real tensors.
- `Abs`: Returns the magnitude or absolute value `abs(x)`.
- `Conj`, `Conjugate`: Returns the complex conjugate `conj(x)`.

Miscellaneous Mathematical Functions:
- `Sqrt`: Applies the square root `sqrt(x)` elementwise.
- `Square`: Applies squaring operation `x**2` elementwise.
- `Sign`: Applies the sign function `sign(x)`; returns -1, 0, or 1.


Examples
--------
>>> import deeptrack as dt

Import the backend-agnostic functionality from DeepTrack2:

>>> from deeptrack.backend  import xp

Create an elementwise feature to execute a backend-agnostic function:

>>> from deeptrack.elementwise import create_elementwise_class
>>>
>>> Abs = create_elementwise_class(
...     name="Abs",
...     function=xp.abs,
...     docstring="Elementwise abs function."
... )
>>>
>>> abs_feature = Abs()

**NumPy backend with direct resolved input**

>>> import numpy as np
>>>
>>> array = np.array([-1.0, 0.0, 2.5])
>>> result = Abs()(array)
>>> result
array([1. , 0. , 2.5])

**PyTorch backend with direct resolved input**

>>> import torch
>>>
>>> tensor = torch.tensor([-1.0, 0.0, 2.5])
>>> result = Abs()(tensor)
>>> result
tensor([1.0000, 0.0000, 2.5000])

**NumPy pipeline**

>>> value = dt.Value(value=np.array([-3.0, 0.0, 3.0]))
>>> pipeline = value >> Abs()
>>> result = pipeline()
>>> result
array([3., 0., 3.])

This is equivalent to:

>>> pipeline = Abs(value)

**PyTorch pipeline**

>>> value = dt.Value(value=torch.tensor([-3.0, 0.0, 3.0]))
>>> pipeline = value >> Abs()
>>> result = pipeline()
>>> result
tensor([3., 0., 3.])

This is equivalent to:

>>> pipeline = Abs(value)

"""


from __future__ import annotations

from typing import Any, Callable, overload, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from deeptrack import Feature, TORCH_AVAILABLE, xp

if TORCH_AVAILABLE:
    import torch


__all__ = [
    "ElementwiseFeature",
    "create_elementwise_class",
    "Sin",
    "Cos",
    "Tan",
    "Arcsin",
    "Arctan",
    "Sinh",
    "Cosh",
    "Tanh",
    "Arcsinh",
    "Arccosh",
    "Arctanh",
    "Round",
    "Floor",
    "Ceil",
    "Exp",
    "Log",
    "Log10",
    "Log2",
    "Angle",
    "Real",
    "Imag",
    "Abs",
    "Conj",
    "Conjugate",
    "Sqrt",
    "Square",
    "Sign",
]


if TYPE_CHECKING:
    import torch


class ElementwiseFeature(Feature):
    """Base class for applying NumPy or PyTorch functions elementwise.

    This class wraps a backend function (e.g., `np.sin`, `torch.exp`) and
    applies it elementwise to the output of another `Feature`.

    If no input feature is provided, the function is applied directly to the
    input passed during resolution.

    Parameters
    ----------
    function: Callable[[array], array]
        A backend-specific function (e.g., `np.sin`, `torch.abs`) or a
        backend-agnostic function (e.g., `xp.sin`, `xp.abs`) that will be
        applied elementwise to the input NumPy array or PyTorch tensor.
    feature: Feature or None, optional
        The input feature to be transformed. If provided, the function is
        applied to the output of this feature. If `None`, the function is
        applied directly to the input passed during evaluation.

    Attributes
    ----------
    __distributed__: bool
        It overrides the default behavior to disable distributed resolution if
        a feature is chained. This ensures the transformation is computed
        locally.

    Methods
    -------
    `get(data, **kwargs) -> array`
        It applies the stored function to the input, optionally resolving the
        wrapped feature first.

    """

    __distributed__: bool
    function: Callable[
        [NDArray[Any] | torch.Tensor],
        NDArray[Any] | torch.Tensor,
    ]
    feature: Feature | None

    def __init__(
        self: ElementwiseFeature,
        function: Callable[
            [NDArray[Any] | torch.Tensor],
            NDArray[Any] | torch.Tensor,
        ],
        feature: Feature | None = None,
        **kwargs: Any,
    ):
        """Initialize ElementwiseFeature.

        Initializes ElementwiseFeature with function and optional input
        feature.

        Parameters
        ----------
        function: Callable[[array], array]
            The function to apply elementwise to the input NumPy array or
            PyTorch tensor.
        feature: Feature or None, optional
            The feature whose output will be transformed. If `None`, the
            function is applied to the direct input.
        **kwargs: Any
            Additional keyword arguments passed to the `Feature` base class.

        """

        super().__init__(**kwargs)

        # Store the function to be applied elementwise
        self.function = function

        # Add the feature dependency if provided
        self.feature = (
            self.add_feature(feature) if feature is not None else None
        )

        # If the feature is set, prevent distributed resolution
        if feature is not None:
            self.__distributed__ = False

    @overload
    def get(
        self: ElementwiseFeature,
        data: NDArray[Any],
        **kwargs: Any,
    ) -> NDArray[Any]:
        ...

    @overload
    def get(
        self: ElementwiseFeature,
        data: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...

    def get(
        self: ElementwiseFeature,
        data: NDArray[Any] | torch.Tensor,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor:
        """Apply the stored function.

        It applies the stored function to the input or the result of the
        wrapped feature.

        Parameters
        ----------
        data: array
            The input data to process. If `feature` was provided at
            initialization, this argument is ignored and the output of
            the wrapped feature is used instead.
        **kwargs: Any
            Additional keyword arguments for compatibility.

        Returns
        -------
        array
            The result of applying the elementwise function.

        """

        # Resolve the input from the chained feature if present
        if self.feature is not None:
            data = self.feature(**kwargs)

        # Apply the function elementwise
        return self.function(data)


def create_elementwise_class(
    name: str,
    function: Callable[
        [NDArray[Any] | torch.Tensor],
        NDArray[Any] | torch.Tensor,
    ],
    docstring: str = "",
) -> type[ElementwiseFeature]:
    """Factory function to create subclasses of ElementwiseFeature.

    This function generates a new subclass of `ElementwiseFeature` that
    applies the given function elementwise to a NumPy array or PyTorch tensor.
    It dynamically sets the class name, qualified name, module, and docstring
    to make the generated class fully compatible with IDEs and documentation
    tools such as Sphinx.

    Parameters
    ----------
    name: str
        Name of the new class to be created (e.g., "Sin", "Exp").
    function: Callable[[array], array]
        The elementwise function to apply, such as `np.sin`, `torch.exp`, or
        `xp.abs`. The arrays can be NumPy arrays or PyTorch tensors.
    docstring: str, optional
        The docstring for the generated class. This string will be visible
        in IDE tooltips and Sphinx documentation.

    Returns
    -------
    type[ElementwiseFeature]
        A dynamically generated subclass of `ElementwiseFeature` that wraps
        the given function.

    Examples
    --------
    >>> import deeptrack as dt

    Import the backend-agnostic functionality from DeepTrack2:

    >>> from deeptrack.backend import xp

    Create an elementwise feature to execute a backend-agnostic function:

    >>> from deeptrack.elementwise import create_elementwise_class
    >>>
    >>> Abs = create_elementwise_class(
    ...     name="Abs",
    ...     function=xp.abs,
    ...     docstring="Elementwise abs function."
    ... )

    **NumPy backend with direct resolved input**

    >>> import numpy as np
    >>>
    >>> array = np.array([-1.0, 0.0, 2.5])
    >>> result = Abs()(array)
    >>> result
    array([1. , 0. , 2.5])

    **PyTorch backend with direct resolved input**

    >>> import torch
    >>>
    >>> tensor = torch.tensor([-1.0, 0.0, 2.5])
    >>> result = Abs()(tensor)
    >>> result
    tensor([1.0000, 0.0000, 2.5000])

    **NumPy pipeline**

    >>> value = dt.Value(value=np.array([-3.0, 0.0, 3.0]))
    >>> pipeline = value >> Abs()
    >>> result = pipeline()
    >>> result
    array([3., 0., 3.])

    This is equivalent to:

    >>> pipeline = Abs(value)

    **PyTorch pipeline**

    >>> value = dt.Value(value=torch.tensor([-3.0, 0.0, 3.0]))
    >>> pipeline = value >> Abs()
    >>> result = pipeline()
    >>> result
    tensor([3., 0., 3.])

    This is equivalent to:

    >>> pipeline = Abs(value)

    """

    class _GeneratedElementwise(ElementwiseFeature):
        """Dynamically generated subclass of ElementwiseFeature."""

        def __init__(
            self: _GeneratedElementwise,
            feature: Feature | None = None,
            **kwargs: Any,
        ) -> None:
            # Initialize the ElementwiseFeature with the fixed function
            super().__init__(function=function, feature=feature, **kwargs)

    # Set the class name to match the intended name
    _GeneratedElementwise.__name__ = name

    # Set the qualified name to improve introspection and traceback readability
    _GeneratedElementwise.__qualname__ = name

    # Attach the user-specified docstring to enable documentation
    _GeneratedElementwise.__doc__ = docstring

    # Set correct module to ensure proper Sphinx indexing and import tracing
    _GeneratedElementwise.__module__ = __name__

    return _GeneratedElementwise


Sin = create_elementwise_class(
    name="Sin",
    function=xp.sin,
    docstring="""
    Apply the sine function elementwise.

    This feature applies `xp.sin` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the sine function will be applied.
        If None, the function is applied directly to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Sin

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Sin()(np.array([0, np.pi / 2, np.pi]))
    >>> result
    array([0.0000000e+00, 1.0000000e+00, 1.2246468e-16])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Sin()(torch.tensor([0, torch.pi / 2, torch.pi]))
    >>> result
    tensor([ 0.0000e+00,  1.0000e+00, -8.7423e-08])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([0, np.pi / 2, np.pi]))
    >>> pipeline = value >> Sin()
    >>> result = pipeline()
    >>> result
    array([0.0000000e+00, 1.0000000e+00, 1.2246468e-16])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([0, torch.pi / 2, torch.pi]))
    >>> pipeline = value >> Sin()
    >>> result = pipeline()
    >>> result
    tensor([ 0.0000e+00,  1.0000e+00, -8.7423e-08])

    These are equivalent to:

    >>> pipeline = Sin(value)

    """
)


Cos = create_elementwise_class(
    name="Cos",
    function=xp.cos,
    docstring="""
    Apply the cosine function elementwise.

    This feature applies `xp.cos` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the cosine function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Cos

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Cos()(np.array([0, np.pi / 2, np.pi]))
    >>> result
    array([ 1.000000e+00,  6.123234e-17, -1.000000e+00])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Cos()(torch.tensor([0, torch.pi / 2, torch.pi]))
    >>> result
    tensor([ 1.0000e+00, -4.3711e-08, -1.0000e+00])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([0, np.pi / 2, np.pi]))
    >>> pipeline = value >> Cos()
    >>> result = pipeline()
    >>> result
    array([ 1.000000e+00,  6.123234e-17, -1.000000e+00])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([0, torch.pi / 2, torch.pi]))
    >>> pipeline = value >> Cos()
    >>> result = pipeline()
    >>> result
    tensor([ 1.0000e+00, -4.3711e-08, -1.0000e+00])

    These are equivalent to:

    >>> pipeline = Cos(value)

    """
)


Tan = create_elementwise_class(
    name="Tan",
    function=xp.tan,
    docstring="""
    Apply the tangent function elementwise.

    This feature applies `xp.tan` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the tangent function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Tan

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Tan()(np.array([0, np.pi / 4, np.pi / 2]))
    >>> result
    array([0.00000000e+00, 1.00000000e+00, 1.63312394e+16])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Tan()(torch.tensor([0, torch.pi / 4, torch.pi / 2]))
    >>> result
    tensor([ 0.0000e+00,  1.0000e+00, -2.2877e+07])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([0, np.pi / 4, np.pi / 2]))
    >>> pipeline = value >> Tan()
    >>> result = pipeline()
    >>> result
    array([0.00000000e+00, 1.00000000e+00, 1.63312394e+16])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([0, torch.pi / 4, torch.pi / 2]))
    >>> pipeline = value >> Tan()
    >>> result = pipeline()
    >>> result
    tensor([ 0.0000e+00,  1.0000e+00, -2.2877e+07])

    These are equivalent to:

    >>> pipeline = Tan(value)

    """
)


Arcsin = create_elementwise_class(
    name="Arcsin",
    function=xp.arcsin,
    docstring="""
    Apply the arcsine function elementwise.

    This feature applies `xp.arcsin` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The input must be in the domain [-1, 1]. Values outside this range will
    produce NaNs or raise runtime warnings or errors, depending on the backend.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the arccosine function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Arcsin

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Arcsin()(np.array([0.0, 0.5, 1.0]))
    >>> result
    array([0.        , 0.52359878, 1.57079633])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Arcsin()(torch.tensor([0.0, 0.5, 1.0]))
    >>> result
    tensor([0.0000, 0.5236, 1.5708])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([0.0, 0.5, 1.0]))
    >>> pipeline = value >> Arcsin()
    >>> result = pipeline()
    >>> result
    array([0.        , 0.52359878, 1.57079633])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([0.0, 0.5, 1.0]))
    >>> pipeline = value >> Arcsin()
    >>> result = pipeline()
    >>> result
    tensor([0.0000, 0.5236, 1.5708])

    These are equivalent to:

    >>> pipeline = Arcsin(value)

    """
)


Arctan = create_elementwise_class(
    name="Arctan",
    function=xp.arctan,
    docstring="""
    Apply the arctangent function elementwise.

    This feature applies `xp.arctan` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the arctangent function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Arctan

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Arctan()(np.array([-1.0, 0.0, 1.0]))
    >>> result
    array([-0.78539816,  0.        ,  0.78539816])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Arctan()(torch.tensor([-1.0, 0.0, 1.0]))
    >>> result
    tensor([-0.7854,  0.0000,  0.7854])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Arctan()
    >>> result = pipeline()
    >>> result
    array([-0.78539816,  0.        ,  0.78539816])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Arctan()
    >>> result = pipeline()
    >>> result
    tensor([-0.7854,  0.0000,  0.7854])

    These are equivalent to:

    >>> pipeline = Arctan(value)

    """
)


Sinh = create_elementwise_class(
    name="Sinh",
    function=xp.sinh,
    docstring="""
    Apply the hyperbolic sine function elementwise.

    This feature applies `xp.sinh` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the hyperbolic sine function will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Sinh

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Sinh()(np.array([-1.0, 0.0, 1.0]))
    >>> result
    array([-1.17520119,  0.        ,  1.17520119])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Sinh()(torch.tensor([-1.0, 0.0, 1.0]))
    >>> result
    tensor([-1.1752,  0.0000,  1.1752])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Sinh()
    >>> result = pipeline()
    >>> result
    array([-1.17520119,  0.        ,  1.17520119])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Sinh()
    >>> result = pipeline()
    >>> result
    tensor([-1.1752,  0.0000,  1.1752])

    These are equivalent to:

    >>> pipeline = Sinh(value)

    """
)


Cosh = create_elementwise_class(
    name="Cosh",
    function=xp.cosh,
    docstring="""
    Apply the hyperbolic cosine function elementwise.

    This feature applies `xp.cosh` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the hyperbolic cosine function will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Cosh

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Cosh()(np.array([-1.0, 0.0, 1.0]))
    >>> result
    array([1.54308063, 1.        , 1.54308063])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Cosh()(torch.tensor([-1.0, 0.0, 1.0]))
    >>> result
    tensor([1.5431, 1.0000, 1.5431])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Cosh()
    >>> result = pipeline()
    >>> result
    array([1.54308063, 1.        , 1.54308063])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Cosh()
    >>> result = pipeline()
    >>> result
    tensor([1.5431, 1.0000, 1.5431])

    These are equivalent to:

    >>> pipeline = Cosh(value)

    """
)


Tanh = create_elementwise_class(
    name="Tanh",
    function=xp.tanh,
    docstring="""
    Apply the hyperbolic tangent function elementwise.

    This feature applies `xp.tanh` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the hyperbolic tangent function will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Tanh

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Tanh()(np.array([-1.0, 0.0, 1.0]))
    >>> result
    array([-0.76159416,  0.        ,  0.76159416])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Tanh()(torch.tensor([-1.0, 0.0, 1.0]))
    >>> result
    tensor([-0.7616,  0.0000,  0.7616])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Tanh()
    >>> result = pipeline()
    >>> result
    array([-0.76159416,  0.        ,  0.76159416])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Tanh()
    >>> result = pipeline()
    >>> result
    tensor([-0.7616,  0.0000,  0.7616])

    These are equivalent to:

    >>> pipeline = Tanh(value)

    """
)


Arcsinh = create_elementwise_class(
    name="Arcsinh",
    function=xp.arcsinh,
    docstring="""
    Apply the inverse hyperbolic sine function elementwise.

    This feature applies `xp.arcsinh` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the hyperbolic arcsine function will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Arcsinh

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Arcsinh()(np.array([-1.0, 0.0, 1.0]))
    >>> result
    array([-0.88137359,  0.        ,  0.88137359])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Arcsinh()(torch.tensor([-1.0, 0.0, 1.0]))
    >>> result
    tensor([-0.8814,  0.0000,  0.8814])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Arcsinh()
    >>> result = pipeline()
    >>> result
    array([-0.88137359,  0.        ,  0.88137359])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Arcsinh()
    >>> result = pipeline()
    >>> result
    tensor([-0.8814,  0.0000,  0.8814])

    These are equivalent to:

    >>> pipeline = Arcsinh(value)

    """
)


Arccosh = create_elementwise_class(
    name="Arccosh",
    function=xp.arccosh,
    docstring="""
    Apply the inverse hyperbolic cosine function elementwise.

    This feature applies `xp.arccosh` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The input must be greater than or equal to 1. Values below this will 
    return NaN or raise errors depending on the backend.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the hyperbolic arccosine function will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Arccosh

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Arccosh()(np.array([1.0, 2.0, 3.0]))
    >>> result
    array([0.        , 1.3169579 , 1.76274717])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Arccosh()(torch.tensor([1.0, 2.0, 3.0]))
    >>> result
    tensor([0.0000, 1.3170, 1.7627])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1.0, 2.0, 3.0]))
    >>> pipeline = value >> Arccosh()
    >>> result = pipeline()
    >>> result
    array([0.        , 1.3169579 , 1.76274717])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([1.0, 2.0, 3.0]))
    >>> pipeline = value >> Arccosh()
    >>> result = pipeline()
    >>> result
    tensor([0.0000, 1.3170, 1.7627])

    These are equivalent to:

    >>> pipeline = Arccosh(value)

    """
)


Arctanh = create_elementwise_class(
    name="Arctanh",
    function=xp.arctanh,
    docstring="""
    Apply the inverse hyperbolic tangent function elementwise.

    This feature applies `xp.arctanh` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The input must be within the open interval (-1, 1). Values outside this 
    range will produce NaNs or raise domain errors depending on the backend.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the hyperbolic arctangent function will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Arctanh

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Arctanh()(np.array([-0.5, 0.0, 0.5]))
    >>> result
    array([-0.54930614,  0.        ,  0.54930614])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Arctanh()(torch.tensor([-0.5, 0.0, 0.5]))
    >>> result
    tensor([-0.5493,  0.0000,  0.5493])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-0.5, 0.0, 0.5]))
    >>> pipeline = value >> Arctanh()
    >>> result = pipeline()
    >>> result
    array([-0.54930614,  0.        ,  0.54930614])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-0.5, 0.0, 0.5]))
    >>> pipeline = value >> Arctanh()
    >>> result = pipeline()
    >>> result
    tensor([-0.5493,  0.0000,  0.5493])

    These are equivalent to:

    >>> pipeline = Arctanh(value)

    """
)


Round = create_elementwise_class(
    name="Round",
    function=xp.round,
    docstring="""
    Apply the rounding function elementwise.

    This feature applies `xp.round` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    This function rounds to the nearest integer. For NumPy, ties round to
    the even number (bankers' rounding). For PyTorch, ties round away from
    zero.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the round function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Round

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Round()(np.array([-1.5, -0.5, 0.5, 1.5]))
    >>> result
    array([-2., -0.,  0.,  2.])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Round()(torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    >>> result
    tensor([-2., -1.,  1.,  2.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.5, -0.5, 0.5, 1.5]))
    >>> pipeline = value >> Round()
    >>> result = pipeline()
    >>> result
    array([-2., -0.,  0.,  2.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    >>> pipeline = value >> Round()
    >>> result = pipeline()
    >>> result
    tensor([-2., -1.,  1.,  2.])

    These are equivalent to:

    >>> pipeline = Round(value)

    """
)


class Floor(ElementwiseFeature):
    """Apply the floor function elementwise.

    This feature applies `xp.floor` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The floor function returns the greatest integer less than or equal to
    each element of the input.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the floor function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Floor

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Floor()(np.array([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> result
    array([-2., -1.,  0.,  0.,  1.])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Floor()(torch.tensor([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> result
    tensor([-2., -1.,  0.,  0.,  1.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> pipeline = value >> Floor()
    >>> result = pipeline()
    >>> result
    array([-2., -1.,  0.,  0.,  1.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> pipeline = value >> Floor()
    >>> result = pipeline()
    >>> result
    tensor([-2., -1.,  0.,  0.,  1.])

    These are equivalent to:

    >>> pipeline = Floor(value)

    """

    def __init__(
        self: Floor,
        feature: Feature | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Floor feature.

        Parameters
        ----------
        feature: Feature or None, optional
            The input feature whose output will be transformed.
        **kwargs: Any
            Additional keyword arguments passed to the base Feature class.

        """

        super().__init__(
            function=self._floor_dispatch,
            feature=feature,
            **kwargs,
        )

    @staticmethod
    def _floor_dispatch(x):
        """Dispatch floor function based on backend.

        This method applies `torch.floor` if the input is a PyTorch tensor,
        and `np.floor` if it is a NumPy array. It ensures compatibility
        across both backends and avoids errors caused by `array-api-compat`.

        This explicit dispatch is necessary because `array-api-compat`'s
        `xp.floor` internally calls `xp.issubdtype(x.dtype, xp.integer)`,
        which fails when `x` is a `torch.Tensor`.

        As a result, this class cannot safely use the
        `create_elementwise_class()` factory, and must be defined manually
        using this backend-aware method.

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The input to transform.

        Returns
        -------
        np.ndarray or torch.Tensor
            The result after applying floor elementwise.

        """

        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return torch.floor(x)

        return np.floor(x)


class Ceil(ElementwiseFeature):
    """Apply the ceiling function elementwise.

    This feature applies `xp.ceil` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The ceiling function returns the smallest integer greater than or equal to
    each element of the input.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the ceil function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Ceil

    Use with NumPy directly:

    >>> import numpy as np
    >>>
    >>> result = Ceil()(np.array([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> result
    array([-1., -0.,  0.,  1.,  2.])

    Use with PyTorch directly:

    >>> import torch
    >>>
    >>> result = Ceil()(torch.tensor([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> result
    tensor([-1., -0.,  0.,  1.,  2.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> pipeline = value >> Ceil()
    >>> result = pipeline()
    >>> result
    array([-1., -0.,  0.,  1.,  2.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.7, -0.5, 0.0, 0.5, 1.7]))
    >>> pipeline = value >> Ceil()
    >>> result = pipeline()
    >>> result
    tensor([-1., -0.,  0.,  1.,  2.])

    These are equivalent to:

    >>> pipeline = Ceil(value)

    """

    def __init__(
        self: Ceil,
        feature: Feature | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ceil feature.

        Parameters
        ----------
        feature: Feature or None, optional
            The input feature whose output will be transformed.
        **kwargs: Any
            Additional keyword arguments passed to the base Feature class.

        """

        super().__init__(
            function=self._ceil_dispatch,
            feature=feature,
            **kwargs
        )

    @staticmethod
    def _ceil_dispatch(x):
        """Dispatch ceiling function based on backend.

        This method applies `torch.ceil` if the input is a PyTorch tensor,
        and `np.ceil` if it is a NumPy array. It ensures compatibility
        across both backends and avoids errors caused by `array-api-compat`.

        This explicit dispatch is necessary because `array-api-compat`'s
        `xp.ceil` internally calls `xp.issubdtype(x.dtype, xp.integer)`,
        which fails when `x` is a `torch.Tensor`.

        As a result, this class cannot safely use the
        `create_elementwise_class()` factory, and must be defined manually
        using this backend-aware method.

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            The input to transform.

        Returns
        -------
        np.ndarray or torch.Tensor
            The result after applying ceil elementwise.

        """

        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return torch.ceil(x)

        return np.ceil(x)


Exp = create_elementwise_class(
    name="Exp",
    function=xp.exp,
    docstring="""
    Apply the exponential function elementwise.

    This feature applies `xp.exp` (NumPy or PyTorch) to each element in the
    input. It supports both direct input and pipeline composition.

    The exponential function computes `e**x` elementwise.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the exponential function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Exp

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Exp()(np.array([-1.0, 0.0, 1.0]))
    >>> result
    array([0.36787944, 1.        , 2.71828183])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Exp()(torch.tensor([-1.0, 0.0, 1.0]))
    >>> result
    tensor([0.3679, 1.0000, 2.7183])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Exp()
    >>> result = pipeline()
    >>> result
    array([0.36787944, 1.        , 2.71828183])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-1.0, 0.0, 1.0]))
    >>> pipeline = value >> Exp()
    >>> result = pipeline()
    >>> result
    tensor([0.3679, 1.0000, 2.7183])

    These are equivalent to:

    >>> pipeline = Exp(value)

    """
)


Log = create_elementwise_class(
    name="Log",
    function=xp.log,
    docstring="""
    Apply the natural logarithm function elementwise.

    This feature applies `xp.log` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The input must be strictly positive. Passing zero or negative values will
    return `-inf` or `NaN`, and may raise warnings or errors depending on 
    the backend.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the natural logarithm function will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Log

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Log()(np.array([1.0, np.e, 10.0]))
    >>> result
    array([0.        , 1.        , 2.30258509])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Log()(torch.tensor([1.0, torch.exp(torch.tensor(1.0)), 10.0]))
    >>> result
    tensor([0.0000, 1.0000, 2.3026])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1.0, np.e, 10.0]))
    >>> pipeline = value >> Log()
    >>> result = pipeline()
    >>> result
    array([0.        , 1.        , 2.30258509])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor(
    ...     [1.0, torch.exp(torch.tensor(1.0)), 10.0])
    ... )
    >>> pipeline = value >> Log()
    >>> result = pipeline()
    >>> result
    tensor([0.0000, 1.0000, 2.3026])

    These are equivalent to:

    >>> pipeline = Log(value)

    """
)


Log10 = create_elementwise_class(
    name="Log10",
    function=xp.log10,
    docstring="""
    Apply the base-10 logarithm function elementwise.

    This feature applies `xp.log10` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The input must be strictly positive. Passing zero or negative values will
    return `-inf` or `NaN`, and may raise warnings or errors depending on 
    the backend.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the logarithm function with base 10 will be
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Log10

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Log10()(np.array([1.0, 10.0, 100.0]))
    >>> result
    array([0., 1., 2.])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Log10()(torch.tensor([1.0, 10.0, 100.0]))
    >>> result
    tensor([0., 1., 2.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1.0, 10.0, 100.0]))
    >>> pipeline = value >> Log10()
    >>> result = pipeline()
    >>> result
    array([0., 1., 2.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([1.0, 10.0, 100.0]))
    >>> pipeline = value >> Log10()
    >>> result = pipeline()
    >>> result
    tensor([0., 1., 2.])

    These are equivalent to:

    >>> pipeline = Log10(value)

    """
)


Log2 = create_elementwise_class(
    name="Log2",
    function=xp.log2,
    docstring="""
    Apply the base-2 logarithm function elementwise.

    This feature applies `xp.log2` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The input must be strictly positive. Passing zero or negative values will
    return `-inf` or `NaN`, and may raise warnings or errors depending on 
    the backend.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the logarithm function with base 2 will be 
        applied. If None, the function is directly applied to the input array
        or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Log2

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Log2()(np.array([1.0, 2.0, 4.0, 8.0]))
    >>> result
    array([0., 1., 2., 3.])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Log2()(torch.tensor([1.0, 2.0, 4.0, 8.0]))
    >>> result
    tensor([0., 1., 2., 3.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1.0, 2.0, 4.0, 8.0]))
    >>> pipeline = value >> Log2()
    >>> result = pipeline()
    >>> result
    array([0., 1., 2., 3.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([1.0, 2.0, 4.0, 8.0]))
    >>> pipeline = value >> Log2()
    >>> result = pipeline()
    >>> result
    tensor([0., 1., 2., 3.])

    These are equivalent to:

    >>> pipeline = Log2(value)

    """
)


Angle = create_elementwise_class(
    name="Angle",
    function=xp.angle,
    docstring="""
    Apply the angle (phase) function elementwise.

    This feature applies `xp.angle` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    The angle function returns the phase angle (in radians) of a complex
    number. For real-valued inputs, it returns 0 for positive and π for
    negative values.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the angle function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Angle

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Angle()(np.array([1+0j, 0+1j, -1+0j, 1+1j]))
    >>> result
    array([0.        , 1.57079633, 3.14159265, 0.78539816])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Angle()(torch.tensor([1+0j, 0+1j, -1+0j, 1+1j]))
    >>> result
    tensor([0.0000, 1.5708, 3.1416, 0.7854])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1+0j, 0+1j, -1+0j, 1+1j]))
    >>> pipeline = value >> Angle()
    >>> result = pipeline()
    >>> result
    array([0.        , 1.57079633, 3.14159265, 0.78539816])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([1+0j, 0+1j, -1+0j, 1+1j]))
    >>> pipeline = value >> Angle()
    >>> result = pipeline()
    >>> result
    tensor([0.0000, 1.5708, 3.1416, 0.7854])

    These are equivalent to:

    >>> pipeline = Angle(value)

    """
)


Real = create_elementwise_class(
    name="Real",
    function=xp.real,
    docstring="""
    Apply the real-part function elementwise.

    This feature applies `xp.real` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    For real-valued inputs, it returns the input unchanged.
    For complex-valued inputs, it returns the real part.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the real function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Real

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Real()(np.array([1+2j, 3+0j, -4.5]))
    >>> result
    array([ 1. ,  3. , -4.5])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Real()(torch.tensor([1+2j, 3+0j, -4.5+0j]))
    >>> result
    tensor([ 1.0000,  3.0000, -4.5000])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1+2j, 3+0j, -4.5]))
    >>> pipeline = value >> Real()
    >>> result = pipeline()
    >>> result
    array([ 1. ,  3. , -4.5])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([1+2j, 3+0j, -4.5+0j]))
    >>> pipeline = value >> Real()
    >>> result = pipeline()
    >>> result
    tensor([ 1.0000,  3.0000, -4.5000])

    These are equivalent to:

    >>> pipeline = Real(value)

    """
)


class Imag(ElementwiseFeature):
    """Apply the imaginary-part function elementwise.

    This class handles real and complex inputs for both NumPy and PyTorch
    backends. For real inputs, it returns 0; for complex inputs, it extracts
    the imaginary part.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the imaginary-part function will be applied.
        If None, the function is applied directly to the input.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Imag

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Imag()(np.array([1+2j, 3+0j, -4.5]))
    >>> result
    array([ 2.,  0.,  0.])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Imag()(torch.tensor([1+2j, 3+0j, -4.5+0j]))
    >>> result
    tensor([2., 0., 0.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1+2j, 3+0j, -4.5]))
    >>> pipeline = value >> Imag()
    >>> result = pipeline()
    >>> result
    array([ 2.,  0.,  0.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([1+2j, 3+0j, -4.5+0j]))
    >>> pipeline = value >> Imag()
    >>> result = pipeline()
    >>> result
    tensor([2., 0., 0.])

    These are equivalent to:

    >>> pipeline = Imag(value)

    """

    def __init__(
        self: Imag,
        feature: Feature | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Imag feature.

        Parameters
        ----------
        feature: Feature or None, optional
            The input feature whose output will be transformed.
        **kwargs: Any
            Additional keyword arguments passed to the base Feature class.

        """
        super().__init__(
            function=self._imag_dispatch,
            feature=feature,
            **kwargs
        )

    @staticmethod
    def _imag_dispatch(x):
        """Dispatch imag function based on backend and dtype.

        This method extracts the imaginary part of the input. For NumPy arrays,
        `np.imag` always returns an array, returning zeros for real-valued
        inputs. However, PyTorch's `torch.imag()` raises a `RuntimeError` when
        called on real tensors.

        To ensure compatibility with both backends, this function checks
        whether the input is a complex tensor before calling `torch.imag`. If
        it is not complex, it returns a zero tensor of the same shape and
        dtype.

        This logic is necessary because `xp.imag` (from array-api-compat) does
        not handle real PyTorch tensors safely, and thus this function cannot
        be created using the factory-based method.

        Parameters
        ----------
        x: np.ndarray or torch.Tensor

        Returns
        -------
        np.ndarray or torch.Tensor
            Imaginary part of the input, or zero if real.

        """

        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            if x.is_complex():
                return torch.imag(x)
            return torch.zeros_like(x)

        return np.imag(x)


Abs = create_elementwise_class(
    name="Abs",
    function=xp.abs,
    docstring="""
    Apply the absolute value function elementwise.

    This feature applies `xp.abs` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    For real-valued inputs, this is the absolute value.
    For complex-valued inputs, this is the magnitude.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Abs

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Abs()(np.array([-1.0, 0.0, 2.5]))
    >>> result
    array([1. , 0. , 2.5])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Abs()(torch.tensor([-1.0, 0.0, 2.5]))
    >>> result
    tensor([1.0000, 0.0000, 2.5000])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-3.0, 0.0, 3.0]))
    >>> pipeline = value >> Abs()
    >>> result = pipeline()
    >>> result
    array([3., 0., 3.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-3.0, 0.0, 3.0]))
    >>> pipeline = value >> Abs()
    >>> result = pipeline()
    >>> result
    tensor([3., 0., 3.])

    These are equivalent to:

    >>> pipeline = Abs(value)

    """
)


Conj = create_elementwise_class(
    name="Conj",
    function=xp.conj,
    docstring="""
    Apply the complex conjugate function elementwise.

    This feature applies `xp.conj` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    For real-valued inputs, the result is unchanged. For complex-valued inputs,
    it returns the complex conjugate (i.e., `a + bj → a - bj`).

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the conjugate function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Conj

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Conj()(np.array([1+2j, 3+0j, -4.5]))
    >>> result
    array([ 1.-2.j,  3.-0.j, -4.5+0.j])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Conj()(torch.tensor([1+2j, 3+0j, -4.5+0j]))
    >>> result
    tensor([ 1.-2.j,  3.-0.j, -4.5+0.j])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([1+2j, 3+0j, -4.5]))
    >>> pipeline = value >> Conj()
    >>> result = pipeline()
    >>> result
    array([ 1.-2.j,  3.-0.j, -4.5+0.j])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([1+2j, 3+0j, -4.5+0j]))
    >>> pipeline = value >> Conj()
    >>> result = pipeline()
    >>> result
    tensor([ 1.-2.j,  3.-0.j, -4.5+0.j])

    These are equivalent to:

    >>> pipeline = Conj(value)

    """
)


Conjugate = Conj


Sqrt = create_elementwise_class(
    name="Sqrt",
    function=xp.sqrt,
    docstring="""
    Apply the square root function elementwise.

    This feature applies `xp.sqrt` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    For non-negative real values, it returns the usual square root.
    For negative inputs, the behavior depends on the backend:
    - NumPy may return `nan` or a complex result depending on dtype.
    - PyTorch raises an error unless the input is explicitly complex.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the square root function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Sqrt

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Sqrt()(np.array([0.0, 1.0, 4.0]))
    >>> result
    array([0., 1., 2.])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Sqrt()(torch.tensor([0.0, 1.0, 4.0]))
    >>> result
    tensor([0., 1., 2.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([0.0, 1.0, 4.0]))
    >>> pipeline = value >> Sqrt()
    >>> result = pipeline()
    >>> result
    array([0., 1., 2.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([0.0, 1.0, 4.0]))
    >>> pipeline = value >> Sqrt()
    >>> result = pipeline()
    >>> result
    tensor([0., 1., 2.])

    These are equivalent to:

    >>> pipeline = Sqrt(value)

    """
)


Square = create_elementwise_class(
    name="Square",
    function=xp.square,
    docstring="""
    Apply the square function elementwise.

    This feature applies `xp.square` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

    This operation computes `x ** 2` for each element.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the square function will be applied.
        If None, the function is directly applied to the input array or tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Square

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Square()(np.array([-2.0, 0.0, 3.0]))
    >>> result
    array([4., 0., 9.])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Square()(torch.tensor([-2.0, 0.0, 3.0]))
    >>> result
    tensor([4., 0., 9.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-2.0, 0.0, 3.0]))
    >>> pipeline = value >> Square()
    >>> result = pipeline()
    >>> result
    array([4., 0., 9.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-2.0, 0.0, 3.0]))
    >>> pipeline = value >> Square()
    >>> result = pipeline()
    >>> result
    tensor([4., 0., 9.])

    These are equivalent to:

    >>> pipeline = Square(value)

    """
)


class Sign(ElementwiseFeature):
    """Apply the sign function elementwise.

    This class uses a backend-aware dispatch to handle NumPy and PyTorch
    tensors safely. It returns:
    - -1 for negative values,
    -  0 for zero,
    - +1 for positive values.

    For complex numbers, it returns `x / abs(x)` when `x != 0`.

    Parameters
    ----------
    feature: Feature or None, optional
        The input feature to which the sign function will be applied.
        If None, the function is applied directly to the input array.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.elementwise import Sign

    Use with NumPy directly:

    >>> import numpy as np
    >>> result = Sign()(np.array([-5.0, 0.0, 2.0]))
    >>> result
    array([-1.,  0.,  1.])

    Use with PyTorch directly:

    >>> import torch
    >>> result = Sign()(torch.tensor([-5.0, 0.0, 2.0]))
    >>> result
    tensor([-1.,  0.,  1.])

    Use in a pipeline with a NumPy value:

    >>> value = dt.Value(value=np.array([-5.0, 0.0, 2.0]))
    >>> pipeline = value >> Sign()
    >>> result = pipeline()
    >>> result
    array([-1.,  0.,  1.])

    Use in a pipeline with a PyTorch value:

    >>> value = dt.Value(value=torch.tensor([-5.0, 0.0, 2.0]))
    >>> pipeline = value >> Sign()
    >>> result = pipeline()
    >>> result
    tensor([-1.,  0.,  1.])

    These are equivalent to:

    >>> pipeline = Sign(value)

    """

    def __init__(
        self: Sign,
        feature: Feature | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sign feature.

        This constructor sets up the elementwise sign operation using a
        backend-aware dispatch. It optionally accepts another Feature whose
        output will be processed by the sign function.

        Parameters
        ----------
        feature : Feature or None, optional
            An optional input feature to be wrapped. If provided, the sign
            operation will be applied to the result of this feature.
            If None, the sign function is applied to the input directly.
        **kwargs : Any
            Additional keyword arguments passed to the base Feature class.

        """

        super().__init__(
            function=self._sign_dispatch,
            feature=feature,
            **kwargs,
        )

    @staticmethod
    def _sign_dispatch(x):
        """Dispatch the sign operation depending on backend and input type.

        This method returns the sign of each element in the input:
        - -1 for negative values,
        -  0 for zero,
        - +1 for positive values.

        For complex inputs, it returns `x / abs(x)` if `x ≠ 0`.

        This function uses `torch.sign()` when the input is a `torch.Tensor`,
        and `np.sign()` otherwise. It avoids using `xp.sign()` from
        `array-api-compat`, which internally calls `xp.issubdtype(...)` and
        fails when provided with a `torch.float32` tensor.

        This method enables full compatibility with both NumPy and PyTorch
        backends, and supports both real and complex-valued inputs.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            The input array or tensor whose elementwise signs will be computed.

        Returns
        -------
        np.ndarray or torch.Tensor
            The elementwise sign values of the input.

        """

        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return torch.sign(x)

        return np.sign(x)
