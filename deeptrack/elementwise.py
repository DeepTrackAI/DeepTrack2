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
Apply cosine elementwise to a Feature:

>>> import numpy as np

>>> from deeptrack import Feature, elementwise

>>> class TestFeature(Feature):
>>>     __distributed__ = False
>>>        def get(self, image, **kwargs):
>>>            output = np.array([[np.pi, 0],
...                               [np.pi / 4, 0]])
>>>            return output

>>> test_feature = TestFeature()
>>> elementwise_cosine = test_feature >> elementwise.Cos()
[[-1.          1.        ]
 [ 0.70710678  1.        ]]

"""

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT389

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from deeptrack.features import Feature


#TODO ***??*** revise ElementwiseFeature - torch, typing, docstring, unit test
class ElementwiseFeature(Feature):
    """
    Base class for applying NumPy functions elementwise.

    This class provides the foundation for subclasses that apply specific 
    NumPy functions (e.g., sin, cos, exp) to the elements of an input array.

    Parameters
    ----------
    function : Callable[[np.ndarray], np.ndarray]
        The NumPy function to be applied elementwise.
    feature : Feature or None, optional
        The input feature to which the function will be applied. If None, 
        the function will be applied to the input array directly.

    Methods
    -------
    `get(image: np.ndarray, **kwargs: Any) -> np.ndarray`
        Returns the result of applying the function to the input array.

    """

    def __init__(
        self: ElementwiseFeature,
        function: Callable[[np.ndarray], np.ndarray],
        feature: Feature | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.function = function
        self.feature = (
           self.add_feature(feature) if feature is not None else None
        )
        if feature:
            self.__distributed__ = False

    def get(
        self: ElementwiseFeature,
        image: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        if self.feature:
            image = self.feature()
        return self.function(image)


#TODO ***??*** revise Sin - torch, typing, docstring, unit test
class Sin(ElementwiseFeature):
    """
    Applies the sine function elementwise.

    Parameters
    ----------
    feature : Feature or None, optional
        The input feature to which the sine function will be applied. 
        If None, the function is applied to the input array directly.
    
    """
    
    def __init__(
        self: Sin,
        feature: Feature | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.sin, feature=feature, **kwargs)


#TODO ***??*** revise Cos - torch, typing, docstring, unit test
class Cos(ElementwiseFeature):
    """
    Applies the cosine function elementwise.

    Parameters
    ----------
    feature : Feature or None, optional
        The input feature to which the cosine function will be applied. 
        If None, the function is applied to the input array directly.
    
    """

    def __init__(
        self: Cos,
        feature: Feature | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.cos, feature=feature, **kwargs)


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


#TODO ***GV*** Consider creating classes dynamically
#TODO ***BM*** Does this eliminate the need for the classes above?

# def create_elementwise_class(name: str, np_function: Callable) -> type:
#    """Factory function to create an ElementwiseFeature subclass."""
#    return type(
#        name,
#        (ElementwiseFeature,),
#        {
#            "__init__": lambda self, feature=None, **kwargs: ElementwiseFeature.__init__(self, np_function, feature, **kwargs),
#        },
#    )


# Sin = create_elementwise_class("Sin", np.sin)
# Cos = create_elementwise_class("Cos", np.cos)
# Tan = create_elementwise_class("Tan", np.tan)