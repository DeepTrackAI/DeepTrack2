from ...features import Feature
import numpy as np


class Equivariance(Feature):
    def __init__(self, mul, add, indexes=slice(None, None, 1), **kwargs):
        super().__init__(mul=mul, add=add, indexes=indexes, **kwargs)

    def get(self, matvec, mul, add, indexes, **kwargs):

        A, b = matvec._value

        mulf = np.eye(len(b))
        addf = np.zeros((len(b), 1))
        mulf[indexes, indexes] = mul
        addf[indexes] = add

        A = mulf @ A
        b = mulf @ b
        b += addf
        return (A, b)


class TranslationalEquivariance(Equivariance):
    def __init__(self, translation, indexes=None):
        if indexes is None:
            indexes = self.get_indexes
        super().__init__(
            translation=translation, add=self.get_add, mul=self.get_mul, indexes=indexes
        )

    def get_add(self, translation):
        return np.array(translation[::-1]).reshape((-1, 1))

    def get_mul(self, translation):
        return np.eye(len(translation))

    def get_indexes(self, translation):
        return slice(len(translation))


class Rotational2DEquivariance(Equivariance):
    def __init__(self, rotation, indexes=None):
        if indexes is None:
            indexes = self.get_indexes
        super().__init__(
            rotation=rotation, add=self.get_add, mul=self.get_mul, indexes=indexes
        )

    def get_add(self):
        return np.zeros((2, 1))

    def get_mul(self, rotation):
        s, c = np.sin(rotation), np.cos(rotation)
        return np.array([[c, s], [-s, c]])

    def get_indexes(self):
        return slice(2)


class ScaleEquivariance(Equivariance):
    def __init__(self, scale, indexes=None):
        if indexes is None:
            indexes = self.get_indexes
        super().__init__(
            scale=scale, add=self.get_add, mul=self.get_mul, indexes=indexes
        )

    def get_add(self, scale):
        return np.zeros((len(scale), 1))

    def get_mul(self, scale):
        return np.diag(scale)

    def get_indexes(self, scale):
        return slice(len(scale))


class LogScaleEquivariance(Equivariance):
    def __init__(self, scale, indexes=None):
        if indexes is None:
            indexes = self.get_indexes
        super().__init__(
            scale=scale, add=self.get_add, mul=self.get_mul, indexes=indexes
        )

    def get_add(self, scale):
        return np.log(scale).reshape((-1, 1))

    def get_mul(self, scale):
        return np.eye(len(scale))

    def get_indexes(self, scale):
        return slice(len(scale))