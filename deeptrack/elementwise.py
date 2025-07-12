"""Classes that apply functions to features elementwise.

This module provides the `elementwise` DeepTrack2 classes
which work as a handle to apply various NumPy functions 
to `Feature` objects elementwise.

Key Features
------------
- **Extends NumPy Functions**

    The convenience of NumPy functions are extended with this module such that
    they can be applied elementwise to a DeepTrack `Feature` object. 

- **Trigonometric Functions**
    The elementary trigonometric functions: Sin, Cos, Tan.

- **Hyperbolic Functions**
    The trigonometric hyperbolic functions: Sinh, Cosh, Tanh.

- **Rounding Functions**
    Common rounding functions: nearest integer rounding `Round`,
    nearest lowest integer `Floor`, nearest highest integer `Ceil`.

- **Exponents And Logarithm Functions**
    Includes Exponential (exp) function, Natural Logarithm function,
    Logarithm function with base 10, and Logarithm function with base 2.

- **Complex Number Functions**
    Functions to get various values from a complex number:
    Angle, Absolute value, Real value, Imaginary value, Conjugate

- **Miscellaneous Functions**
    Contains Square root, Square, Sign function.

Module Structure
----------------
Classes:

- `ElementwiseFeature`
   Forms the base from which other classes inherit from.

- `Sin`

- `Cos`

- `Tan`

- `ArcSin`

- `Arccos`

- `ArcTan`

- `Sinh`

- `Cosh`

- `Tanh`

- `ArcSinh`

- `Arccosh`

- `ArcTanh`

- `Round`

- `Floor`

- `Ceil`

- `Exp`

- `Log`

- `Log10`

- `Log2`

- `Angle`

- `Real`

- `Imag`

- `Abs`

- `Conjugate`

- `Sqrt`

- `Square`

- `Sign`


Examples
--------
import deeptrack as dt

Import the backend-agnostic functionality from DeepTrack2:
>>> from deeptrack.backend  import xp

Create a elementwise feature to execute a backend-agnostic function:
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

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT389

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

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
        applied to the output of this feature. If None, the function is applied
        directly to the input passed to `resolve`.

    Attributes
    ----------
    __distributed__: bool
        It overrides the default behavior to disable distributed resolution if
        a feature is chained. This ensures the transformation is computed
        locally.

    Methods
    -------
    get(image: array, **kwargs: Any) -> array
        It applies the stored function to the input, optionally resolving the
        wrapped feature first.

    """

    __distributed__: bool
    function: Callable[[NDArray[Any] | torch.Tensor],
                       NDArray[Any] | torch.Tensor] | None
    feature: Feature

    def __init__(
        self: ElementwiseFeature,
        function: Callable[
            [NDArray[Any] | torch.Tensor],
            NDArray[Any] | torch.Tensor
        ],
        feature: Feature | None = None,
        **kwargs: Any,
    ):
        """Initialize ElementwiseFeature.

        It initializes ElementwiseFeature with function and optional input
        feature.

        Parameters
        ----------
        function: Callable[[array], array]
            The function to apply elementwise to the input NumPy array or
            PyTorch tensor.
        feature: Feature or None, optional
            The feature whose output will be transformed. If None, the
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
        if feature:
            self.__distributed__ = False

    def get(
        self: ElementwiseFeature,
        image: NDArray[Any] | torch.Tensor,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor:
        """Apply the stored function.

        It applies the stored function to the input or the result of the
        wrapped feature.

        Parameters
        ----------
        image: array
            The input data to process, or a placeholder if a feature is
            chained.
        **kwargs: Any
            Additional keyword arguments for compatibility.

        Returns
        -------
        array
            The result of applying the elementwise function.

        """

        # Resolve the input from the chained feature if present
        if self.feature:
            image = self.feature()

        # Apply the function elementwise
        return self.function(image)


def create_elementwise_class(
    name: str,
    function: Callable[
        [NDArray[Any] | torch.Tensor],
        NDArray[Any] | torch.Tensor
    ],
    docstring: str = "",
) -> type:
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
        `xp.abs`.
    docstring: str, optional
        The docstring for the generated class. This string will be visible
        in IDE tooltips and Sphinx documentation.

    Returns
    -------
    type
        A dynamically generated subclass of `ElementwiseFeature` that wraps
        the given function.

    Examples
    --------
    import deeptrack as dt

    Import the backend-agnostic functionality from DeepTrack2:
    >>> from deeptrack.backend  import xp

    Create a elementwise feature to execute a backend-agnostic function:
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
        ):
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
    Apply the sine function elementwise to the output of a feature.

    This feature applies `xp.sin` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

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

    Use in a pipeline with a Torch value:
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
    Apply the cosine function elementwise to the output of a feature.

    This feature applies `xp.cos` to each element in a NumPy array or a
    PyTorch tensor. It supports both direct input and pipeline composition.

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

    Use in a pipeline with a Torch value:
    >>> value = dt.Value(value=torch.tensor([0, torch.pi / 2, torch.pi]))
    >>> pipeline = value >> Cos()
    >>> result = pipeline()
    >>> result
    tensor([ 1.0000e+00, -4.3711e-08, -1.0000e+00])

    These are equivalent to:
    >>> pipeline = Cos(value)

    """
)

if False:
    #TODO ***??*** revise Tan - torch, typing, docstring, unit test
    class Tan(ElementwiseFeature):
        """
        Applies the tangent function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the tangent function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Tan,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.tan, feature=feature, **kwargs)


    #TODO ***??*** revise Arcsin - torch, typing, docstring, unit test
    class Arcsin(ElementwiseFeature):
        """
        Applies the arcsine function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the arcsine function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Arcsin,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.arcsin, feature=feature, **kwargs)


    #TODO ***??*** revise Arccos - torch, typing, docstring, unit test
    class Arccos(ElementwiseFeature):
        """
        Applies the arccosine function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the arccosine function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Arccos,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.arccos, feature=feature, **kwargs)


    #TODO ***??*** revise Arctan - torch, typing, docstring, unit test
    class Arctan(ElementwiseFeature):
        """
        Applies the arctangent function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the arctangent function will be applied. 
            If None, the function is applied to the input array directly.
        
        """
        def __init__(
            self: Arctan,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.arctan, feature=feature, **kwargs)


    #TODO ***??*** revise Sinh - torch, typing, docstring, unit test
    class Sinh(ElementwiseFeature):
        """
        Applies the hyperbolic sine function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the hyperbolic sine function will be 
            applied. If None, the function is applied to the input array directly.
        
        """
        def __init__(
            self: Sinh,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.sinh, feature=feature, **kwargs)


    #TODO ***??*** revise Cosh - torch, typing, docstring, unit test
    class Cosh(ElementwiseFeature):
        """
        Applies the hyperbolic cosine function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the hyperbolic cosine function will be 
            applied. If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Cosh,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.cosh, feature=feature, **kwargs)


    #TODO ***??*** revise Tanh - torch, typing, docstring, unit test
    class Tanh(ElementwiseFeature):
        """
        Applies the hyperbolic tangent function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the hyperbolic tangent function will be 
            applied. If None, the function is applied to the input array directly.
        
        """
            
        def __init__(
            self: Tanh,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.tanh, feature=feature, **kwargs)


    #TODO ***??*** revise Arcsinh - torch, typing, docstring, unit test
    class Arcsinh(ElementwiseFeature):
        """
        Applies the hyperbolic arcsine function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the hyperbolic arcsine function will be 
            applied. If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Arcsinh,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.arcsinh, feature=feature, **kwargs)


    #TODO ***??*** revise Arccosh - torch, typing, docstring, unit test
    class Arccosh(ElementwiseFeature):
        """
        Applies the hyperbolic arccosine function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the hyperbolic arccosine function will be 
            applied. If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Arccosh,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.arccosh, feature=feature, **kwargs)


    #TODO ***??*** revise Arctanh - torch, typing, docstring, unit test
    class Arctanh(ElementwiseFeature):
        """
        Applies the hyperbolic arctangent function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the hyperbolic arctangent function will be 
            applied. If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Arctanh,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.arctanh, feature=feature, **kwargs)


    #TODO ***??*** revise Round - torch, typing, docstring, unit test
    class Round(ElementwiseFeature):
        """
        Applies the round function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the round function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Round,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.around, feature=feature, **kwargs)


    #TODO ***??*** revise Floor - torch, typing, docstring, unit test
    class Floor(ElementwiseFeature):
        """
        Applies the floor function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the floor function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Floor,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.floor, feature=feature, **kwargs)


    #TODO ***??*** revise Ceil - torch, typing, docstring, unit test
    class Ceil(ElementwiseFeature):
        """
        Applies the ceil function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the ceil function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Ceil,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.ceil, feature=feature, **kwargs)


    #TODO ***??*** revise Exp - torch, typing, docstring, unit test
    class Exp(ElementwiseFeature):
        """
        Applies the exponential function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the exponential function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Exp,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.exp, feature=feature, **kwargs)


    #TODO ***??*** revise Log - torch, typing, docstring, unit test
    class Log(ElementwiseFeature):
        """
        Applies the natural logarithm function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the natural logarithm function will be 
            applied. If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Log,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.log, feature=feature, **kwargs)


    #TODO ***??*** revise Log10 - torch, typing, docstring, unit test
    class Log10(ElementwiseFeature):
        """
        Applies the logarithm function with base 10 elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the logarithm function with base 10 will be
            applied. If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Log10,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.log10, feature=feature, **kwargs)


    #TODO ***??*** revise Log2 - torch, typing, docstring, unit test
    class Log2(ElementwiseFeature):
        """
        Applies the logarithm function with base 2 elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the logarithm function with base 2 will be 
            applied. If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Log2,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.log2, feature=feature, **kwargs)


    #TODO ***??*** revise Angle - torch, typing, docstring, unit test
    class Angle(ElementwiseFeature):
        """
        Applies the angle function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the angle function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Angle,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.angle, feature=feature, **kwargs)


    #TODO ***??*** revise Real - torch, typing, docstring, unit test
    class Real(ElementwiseFeature):
        """
        Applies the real function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the real function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Real,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.real, feature=feature, **kwargs)


    #TODO ***??*** revise Imag - torch, typing, docstring, unit test
    class Imag(ElementwiseFeature):
        """
        Applies the imaginary function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the imaginary function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Imag,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.imag, feature=feature, **kwargs)


    #TODO ***??*** revise Abs - torch, typing, docstring, unit test
    class Abs(ElementwiseFeature):
        """
        Applies the absolute value function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the absolute value function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Abs,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.abs, feature=feature, **kwargs)


    #TODO ***??*** revise Conjugate - torch, typing, docstring, unit test
    class Conjugate(ElementwiseFeature):
        """
        Applies the conjugate function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the conjugate function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Conjugate,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.conjugate, feature=feature, **kwargs)


    #TODO ***??*** revise Sqrt - torch, typing, docstring, unit test
    class Sqrt(ElementwiseFeature):
        """
        Applies the square root function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the square root function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Sqrt,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.sqrt, feature=feature, **kwargs)


    #TODO ***??*** revise Square - torch, typing, docstring, unit test
    class Square(ElementwiseFeature):
        """
        Applies the square function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the square function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Square,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.square, feature=feature, **kwargs)


    #TODO ***??*** revise Sign - torch, typing, docstring, unit test
    class Sign(ElementwiseFeature):
        """
        Applies the sign function elementwise.

        Parameters
        ----------
        feature : Feature or None, optional
            The input feature to which the sign function will be applied. 
            If None, the function is applied to the input array directly.
        
        """

        def __init__(
            self: Sign,
            feature: Feature | None = None,
            **kwargs: Any
        ) -> None:
            super().__init__(np.sign, feature=feature, **kwargs)