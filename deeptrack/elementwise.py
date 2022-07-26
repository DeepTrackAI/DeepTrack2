"""Features that apply functions to arrays elementwise.

All features defined here can be inserted into a pipeline as::
   A >> ElementwiseFeature()
or::
   ElementwiseFeature(A)
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
Trigonometric functions
=======================
"""


class Cos(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.cos, feature=feature, **kwargs)


class Sin(ElementwiseFeature):
    def __init__(self, feature=None, **kwargs):
        super().__init__(np.sin, feature=feature, **kwargs)


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
Hyperbolic functions
=======================
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
========
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
Exponents and logaritms
=======================
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
Complex numbers
===============
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
Misc.
=====
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
