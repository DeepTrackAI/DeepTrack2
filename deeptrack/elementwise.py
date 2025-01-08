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
Perform cosine elementwise to a Feature:

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

from typing import Callable, Optional, Any

import numpy as np

from .features import Feature


class ElementwiseFeature(Feature):

    __gpu_compatible__: bool = True

    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        feature = None,
        **kwargs: Any
    ) -> None:

        self.function = function
        super().__init__(**kwargs)
        self.feature = self.add_feature(feature) if feature else feature

        if feature:
            self.__distributed__ = False

    def get(
        self,
        image: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        if self.feature:
            image = self.feature()
        return self.function(image)


class Sin(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.sin, feature=feature, **kwargs)


class Cos(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.cos, feature=feature, **kwargs)


class Tan(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.tan, feature=feature, **kwargs)


class Arcsin(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.arcsin, feature=feature, **kwargs)


class Arccos(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.arccos, feature=feature, **kwargs)


class Arctan(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.arctan, feature=feature, **kwargs)


class Sinh(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.sinh, feature=feature, **kwargs)


class Cosh(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.cosh, feature=feature, **kwargs)


class Tanh(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.tanh, feature=feature, **kwargs)


class Arcsinh(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.arcsinh, feature=feature, **kwargs)


class Arccosh(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.arccosh, feature=feature, **kwargs)


class Arctanh(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.arctanh, feature=feature, **kwargs)


class Round(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.around, feature=feature, **kwargs)


class Floor(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.floor, feature=feature, **kwargs)


class Ceil(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.ceil, feature=feature, **kwargs)


class Exp(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.exp, feature=feature, **kwargs)


class Log(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.log, feature=feature, **kwargs)


class Log10(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.log10, feature=feature, **kwargs)


class Log2(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.log2, feature=feature, **kwargs)


class Angle(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.angle, feature=feature, **kwargs)


class Real(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.real, feature=feature, **kwargs)


class Imag(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.imag, feature=feature, **kwargs)


class Abs(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.abs, feature=feature, **kwargs)


class Conjugate(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.conjugate, feature=feature, **kwargs)


class Sqrt(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.sqrt, feature=feature, **kwargs)


class Square(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.square, feature=feature, **kwargs)


class Sign(ElementwiseFeature):
    def __init__(
        self,
        feature = None,
        **kwargs: Any
    ) -> None:
        super().__init__(np.sign, feature=feature, **kwargs)
