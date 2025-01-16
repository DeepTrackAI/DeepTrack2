"""
Provides features for manipulating optical fields using Fourier transforms and 
propagation matrices.

This module includes operations to simulate optical field propagation and
perform transformations in the frequency domain. These features can be combined
in processing pipelines for optical simulations and holographic
reconstructions.

Module Structure
----------------
Functions:
- `get_propagation_matrix`: Computes the propagation matrix.

Classes:
- `Rescale`: Rescales an optical field by subtracting the real part of the
    field before multiplication.
- `FourierTransform`: Creates matrices for propagating an optical field.
- `InverseFourierTransform`: Creates matrices for propagating an optical field.
- `FourierTransformTransformation`: Applies a power of the forward or inverse
    propagation matrix to an optical field.

Example
-------
Simulate optical field propagation with Fourier transforms:

>>> from deeptrack import holography
>>> import numpy as np
>>> field = np.random.rand(128, 128, 2)  # Random optical field
>>> rescale_op = holography.Rescale(0.5)
>>> scaled_field = rescale_op(field)
>>> ft_op = holography.FourierTransform()
>>> transformed_field = ft_op(scaled_field)
>>> ift_op = holography.InverseFourierTransform()
>>> reconstructed_field = ift_op(transformed_field)

"""

from deeptrack.image import maybe_cupy
from deeptrack import Feature
import numpy as np


def get_propagation_matrix(shape, to_z, pixel_size, wavelength, dx=0, dy=0):
    """
    Computes the propagation matrix for simulating the propagation of an
    optical field.

    Parameters
    ----------
    shape : tuple of int
        Shape of the optical field (height, width).
    to_z : float
        Propagation distance along the z-axis.
    pixel_size : float
        Size of each pixel in the field.
    wavelength : float
        Wavelength of the optical field.
    dx : float, optional
        Lateral shift in the x-direction (default is 0).
    dy : float, optional
        Lateral shift in the y-direction (default is 0).

    Returns
    -------
    ndarray
        The computed propagation matrix as a complex-valued array.
    """

    k = 2 * np.pi / wavelength
    yr, xr, *_ = shape

    x = np.arange(0, xr, 1) - xr / 2 + (xr % 2) / 2
    y = np.arange(0, yr, 1) - yr / 2 + (yr % 2) / 2

    x = 2 * np.pi / pixel_size * x / xr
    y = 2 * np.pi / pixel_size * y / yr

    KXk, KYk = np.meshgrid(x, y)
    KXk = maybe_cupy(KXk.astype(complex))
    KYk = maybe_cupy(KYk.astype(complex))

    K = np.real(np.sqrt(1 - (KXk / k) ** 2 - (KYk / k) ** 2))
    C = np.fft.fftshift(((KXk / k) ** 2 + (KYk / k) ** 2 < 1) * 1.0)

    return C * np.fft.fftshift(
        np.exp(k * 1j * (to_z * (K - 1) - dx * KXk / k - dy * KYk / k))
    )


class Rescale(Feature):
    """
    Rescales an optical field by subtracting the real part of the field
    before multiplication.

    Parameters
    ----------
    rescale : float
        rescaling factor
    """

    def __init__(self, rescale=1, **kwargs):
        super().__init__(rescale=rescale, **kwargs)

    def get(self, image, rescale, **kwargs):
        image = np.array(image)
        image[..., 0] = (image[..., 0] - 1) * rescale + 1
        image[..., 1] *= rescale

        return image


class FourierTransform(Feature):
    """
    Computes the Fourier transform of an optical field with optional 
    symmetric padding.
    
    Parameters
    ----------
    padding : int, optional
        Number of pixels to pad symmetrically around the image (default is 32).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, image, padding=32, **kwargs):

        im = np.copy(image[..., 0] + 1j * image[..., 1])
        im = np.pad(
            im,
            ((padding, padding), (padding, padding)),
            mode="symmetric"
            )
        f1 = np.fft.fft2(im)
        return f1


class InverseFourierTransform(Feature):
    """
    Computes the inverse Fourier transform and removes padding.

    Parameters
    ----------
    padding : int, optional
        Number of pixels removed symmetrically after inverse transformation
        (default is 32).
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
    """
    Applies a power of the forward or inverse propagation matrix to an optical
    field.

    Parameters
    ----------
    Tz : ndarray
        Forward propagation matrix.
    Tzinv : ndarray
        Inverse propagation matrix.
    i : int
        Power of the propagation matrix to apply. Negative values apply the
        inverse.
    """
    def __init__(self, Tz, Tzinv, i, **kwargs):
        super().__init__(Tz=Tz, Tzinv=Tzinv, i=i, **kwargs)

    def get(self, image, Tz, Tzinv, i, **kwargs):
        if i < 0:
            image *= Tzinv ** np.abs(i)
        else:
            image *= Tz ** i
        return image
