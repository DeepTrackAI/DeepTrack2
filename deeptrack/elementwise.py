
"""Classes that apply functions to features elementwise.

This module provides the elementwise DeepTrack2 classes
which work as a handle to perform various numpy functions 
to features elementwise.

Module Structure
----------------

Elementwise Base Class:
- `ElementwiseFeature` 

Trigonometric Classes:
- `Sin`
- `Cos`
- `Tan`
- `ArcSin`
- `Arccos`
- `ArcTan`

Hyperbolic Classes:
- `Sinh`
- `Cosh`
- `Tanh`
- `ArcSinh`
- `Arccosh`
- `ArcTanh`

Rounding Classes:
- `Round`
- `Floor`
- `Ceil`

Exponents & Logarithm Classes:
- `Exp`
- `Log`
- `Log10`
- `Log2`

Complex Number Classes:
- `Angle`
- `Real`
- `Imag`
- `Abs`
- `Conjugate`

Miscellaneous Classes:
- `Sqrt`
- `Square`
- `Sign`

"""

from .features import Feature
import numpy as np


class ElementwiseFeature(Feature):
    
    __gpu_compatible__ = True

    def __init__(self, function, feature=None, **kwargs):
        self.function = function
        super().__init__(**kwargs)
        self.feature = self.add_feature(feature) if feature else feature

        if feature:
            self.__distributed__ = False

    def get(self, image, **kwargs):
        if self.feature:
            image = self.feature()
        return self.function(image)


"""
Trigonometric Functions
"""

class Sin(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.sin, feature=feature, **kwargs)


class Cos(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.cos, feature=feature, **kwargs)


class Tan(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.tan, feature=feature, **kwargs)


class Arcsin(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.arcsin, feature=feature, **kwargs)


class Arccos(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.arccos, feature=feature, **kwargs)


class Arctan(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.arctan, feature=feature, **kwargs)


"""
Hyperbolic Functions
"""


class Sinh(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.sinh, feature=feature, **kwargs)


class Cosh(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.cosh, feature=feature, **kwargs)


class Tanh(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.tanh, feature=feature, **kwargs)


class Arcsinh(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.arcsinh, feature=feature, **kwargs)


class Arccosh(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.arccosh, feature=feature, **kwargs)


class Arctanh(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.arctanh, feature=feature, **kwargs)


"""
Rounding
"""


class Round(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.around, feature=feature, **kwargs)


class Floor(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.floor, feature=feature, **kwargs)


class Ceil(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.ceil, feature=feature, **kwargs)


"""
Exponents and Logarithms
"""


class Exp(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.exp, feature=feature, **kwargs)


class Log(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.log, feature=feature, **kwargs)


class Log10(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.log10, feature=feature, **kwargs)


class Log2(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.log2, feature=feature, **kwargs)


"""
Complex Numbers
"""


class Angle(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.angle, feature=feature, **kwargs)


class Real(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.real, feature=feature, **kwargs)


class Imag(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.imag, feature=feature, **kwargs)


class Abs(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.abs, feature=feature, **kwargs)


class Conjugate(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.conjugate, feature=feature, **kwargs)


"""
Miscellaneous

"""


class Sqrt(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.sqrt, feature=feature, **kwargs)


class Square(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.square, feature=feature, **kwargs)


class Sign(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.sign, feature=feature, **kwargs)
