""" User defined equivariances for training LodeSTAR.

Example:
    action = Multiply(value=np.random.uniform(0.5, 1.5))
    equivariance = Equivariance(mul=action.value) 

"""

from ...features import Feature
import numpy as np


class Equivariance(Feature):
    """Defines equivariance between action and prediction.

    Should define both a multiplicative equivariance (1, if invariant) and a additive equivariance (0, if invariant)

    Parameters
    ----------
    mul : float, array-like
        Multiplicative equivariance
    add : float, array-like
        Additive equivariance
    indices : optional, int or slice
        Index of related predicted value(s)

    """

    def __init__(self, mul, add, indices=slice(None, None, 1), indexes=None, **kwargs):
        if indexes is not None:
            indices = indexes

        super().__init__(mul=mul, add=add, indices=indices, **kwargs)

    def get(self, matvec, mul, add, indices, **kwargs):

        A, b = matvec._value

        mulf = np.eye(len(b))
        addf = np.zeros((len(b), 1))

        addf[indices] = add

        if isinstance(indices, (slice, int)):
            mulf[indices, indices] = mul
        else:
            for i in indices:
                for j in indices:
                    mulf[i, j] = mul[i, j]

        A = mulf @ A
        b = mulf @ b
        b += addf
        return (A, b)


class TranslationalEquivariance(Equivariance):
    """Defines translation-like equivariance between action and prediction, for use with dt.Affine

    Parameters
    ----------
    translate : float, array-like
        Should be exactly `affine.translate`

    """

    def __init__(self, translation, indices=None):
        if indices is None:
            indices = self.get_indices
        super().__init__(
            translation=translation, add=self.get_add, mul=self.get_mul, indices=indices
        )

    def get_add(self, translation):
        return np.array(translation[::-1]).reshape((-1, 1))

    def get_mul(self, translation):
        return np.eye(len(translation))

    def get_indices(self, translation):
        return slice(len(translation))


class Rotational2DEquivariance(Equivariance):
    """Defines rotation-like equivariance between action and prediction, for use with dt.Affine

    Parameters
    ----------
    rotate : float, array-like
        Should be exactly `affine.rotate`

    """

    def __init__(self, rotate, indices=None):
        if indices is None:
            indices = self.get_indices
        super().__init__(
            rotate=rotate, add=self.get_add, mul=self.get_mul, indices=indices
        )

    def get_add(self):
        return np.zeros((2, 1))

    def get_mul(self, rotate):
        s, c = np.sin(rotate), np.cos(rotate)
        return np.array([[c, s], [-s, c]])

    def get_indices(self):
        return slice(2)


class ScaleEquivariance(Equivariance):
    """Defines scale-like equivariance between action and prediction, for use with dt.Affine

    Parameters
    ----------
    scale : float, array-like
        Should be exactly `affine.scale`

    """

    def __init__(self, scale, indices=None):
        if indices is None:
            indices = self.get_indices
        super().__init__(
            scale=scale, add=self.get_add, mul=self.get_mul, indices=indices
        )

    def get_add(self, scale):
        return np.zeros((len(scale), 1))

    def get_mul(self, scale):
        return np.diag(scale)

    def get_indices(self, scale):
        return slice(len(scale))


class LogScaleEquivariance(Equivariance):
    """Defines scale-like equivariance between action and prediction, for use with dt.Affine

    Converts the scaling to log scale, for an additive equivariance.

    Parameters
    ----------
    scale : float, array-like
        Should be exactly `affine.scale`

    """

    def __init__(self, scale, indices=None):
        if indices is None:
            indices = self.get_indices
        super().__init__(
            scale=scale, add=self.get_add, mul=self.get_mul, indices=indices
        )

    def get_add(self, scale):
        return np.log(scale).reshape((-1, 1))

    def get_mul(self, scale):
        return np.eye(len(scale))

    def get_indices(self, scale):
        return slice(len(scale))