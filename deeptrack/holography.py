from .features import Feature
import numpy as np


def get_propagation_matrix(shape, to_z, pixel_size, wavelength):

    k = 2 * np.pi / wavelength
    yr, xr, *_ = shape

    x = 2 * np.pi / pixel_size * np.arange(-(xr / 2 - 1 / 2), (xr / 2 + 1 / 2), 1) / xr
    y = 2 * np.pi / pixel_size * np.arange(-(yr / 2 - 1 / 2), (yr / 2 + 1 / 2), 1) / yr
    KXk, KYk = np.meshgrid(x, y)

    K = np.real(
        np.sqrt(np.array(1 - (KXk / k) ** 2 - (KYk / k) ** 2, dtype=np.complex64))
    )
    C = np.fft.fftshift(((KXk / k) ** 2 + (KYk / k) ** 2 < 1) * 1.0)

    return C * np.fft.fftshift(np.exp(k * 1j * to_z * (K - 1)))


class Rescale(Feature):
    """Rescales an optical field by subtracting the real part of the field beofre multiplication.

    Parameters
    ----------
    rescale : float
        index of z-propagator matrix
    """

    def __init__(self, rescale=1, **kwargs):
        super().__init__(rescale=rescale, **kwargs)

    def get(self, image, rescale, **kwargs):
        image = np.array(image)
        image[..., 0] = (image[..., 0] - 1) * rescale + 1
        image[..., 1] *= rescale

        return image


class FourierTransform(Feature):
    """Creates matrices for propagating an optical field.

    Parameters
    ----------
    i : int
        index of z-propagator matrix
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, image, padding=32, **kwargs):

        im = np.copy(image[..., 0] + 1j * image[..., 1])
        im = np.pad(im, ((padding, padding), (padding, padding)), mode="symmetric")
        f1 = np.fft.fft2(im)
        return f1


class InverseFourierTransform(Feature):
    """Creates matrices for propagating an optical field.

    Parameters
    ----------
    i : int
        index of z-propagator matrix
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, image, padding=32, **kwargs):
        im = np.fft.ifft2(image)
        imnew = np.zeros(
            (image.shape[0] - padding * 2, image.shape[1] - padding * 2, 2)
        )
        imnew[..., 0] = np.real(im[padding:-padding, padding:-padding])
        imnew[..., 1] = np.imag(im[padding:-padding, padding:-padding])
        return imnew


class FourierTransformTransformation(Feature):
    def __init__(self, Tz, Tzinv, i, **kwargs):
        super().__init__(Tz=Tz, Tzinv=Tzinv, i=i, **kwargs)

    def get(self, image, Tz, Tzinv, i, **kwargs):
        if i < 0:
            image *= Tzinv ** np.abs(i)
        else:
            image *= Tz ** i
        return image
