"""Core features for manipulating optical fields using Fourier transforms and 
propagation matrices.

This module includes operations to simulate optical field propagation and
perform transformations in the frequency domain. These features can be combined
in processing pipelines for optical simulations and holographic
reconstructions.

Key Features
------------
- **Optical Field Processing**

    Provides Fourier transforms, rescaling, and wavefront propagation for 
    complex-valued optical fields, handling both real and imaginary components.

- **Fourier Optics and Wave Propagation**

    Implements Fourier transforms with optional padding for accurate 
    frequency-domain analysis and propagation matrices to simulate free-space 
    wavefront propagation with spatial and frequency domain shifts.

- **Phase & Amplitude Manipulation**

    Enables scaling, normalization, and modulation of phase and amplitude to 
    preserve intensity distribution and enhance wavefront reconstruction.

Module Structure
----------------
Classes:

- `Rescale`:
    
    Rescales an optical field by subtracting the real part of the
    field before multiplication.

- `FourierTransform`:
    
    Creates matrices for propagating an optical field.

- `InverseFourierTransform`:
    
    Creates matrices for propagating an optical field.

- `FourierTransformTransformation`:
    
    Applies a power of the forward or inverse
    propagation matrix to an optical field.

Functions:

- `get_propagation_matrix`

    def get_propagation_matrix(
        shape: tuple[int, int],
        to_z: float,
        pixel_size: float,
        wavelength: float,
        dx: float = 0,
        dy: float = 0
    ) -> np.ndarray

    Computes the propagation matrix.


Examples
--------
Simulate optical field propagation with Fourier transforms:

>>> import deeptrack as dt
>>> import numpy as np

Define a random optical field:
>>> field = np.random.rand(128, 128, 2)  

Rescale the field and compute the Fourier transform:
>>> rescale_op = dt.holography.Rescale(0.5)
>>> scaled_field = rescale_op(field)
>>> ft_op = dt.holography.FourierTransform()
>>> transformed_field = ft_op(scaled_field)

Reconstruct the field using the inverse Fourier transform:
>>> ift_op = dt.holography.InverseFourierTransform()
>>> reconstructed_field = ift_op(transformed_field)

"""

from __future__ import annotations
from typing import Any
from deeptrack.image import maybe_cupy, Image
from deeptrack import Feature
import numpy as np


def get_propagation_matrix(
    shape: tuple[int, int],
    to_z: float,
    pixel_size: float,
    wavelength: float,
    dx: float = 0,
    dy: float = 0
) -> np.ndarray:
    """Computes the propagation matrix for simulating the propagation of an
    optical field.

    The propagation matrix is used to model wavefront propagation in free space 
    based on the angular spectrum method.

    Parameters
    ----------
    shape: tuple[int, int]
        The dimensions of the optical field (height, width).
    to_z: float
        Propagation distance along the z-axis.
    pixel_size: float
        The physical size of each pixel in the optical field.
    wavelength: float
        The wavelength of the optical field.
    dx: float, optional
        Lateral shift in the x-direction (default: 0).
    dy: float, optional
        Lateral shift in the y-direction (default: 0).

    Returns
    -------
    np.ndarray
        A complex-valued 2D NumPy array representing the propagation matrix.

    Notes
    -----
    - Uses `np.fft.fftshift` to shift the zero-frequency component to the 
      center.
    - Computed based on the wave equation in Fourier space.

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
    """Rescales an optical field by modifying its real and imaginary 
    components.

    The transformation is applied as:
        - The real part is shifted and scaled: `(real - 1) * rescale + 1`
        - The imaginary part is scaled by `rescale`

    Parameters
    ----------
    rescale: float
       The scaling factor applied to both real and imaginary components.

    Methods
    -------
    `get(image: Image | np.ndarray, rescale: float, **kwargs: dict[str, Any]) -> Image | np.ndarray`
        Rescales the image while preserving phase information.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> field = np.random.rand(128, 128, 2)
    >>> rescaled_field = dt.holography.Rescale(0.5)(field)

    """

    def __init__(self, rescale=1, **kwargs):
        super().__init__(rescale=rescale, **kwargs)

    def get(
        self: Rescale, 
        image: Image | np.ndarray, 
        rescale: float, 
        **kwargs: dict[str, Any],
    ) -> Image | np.ndarray:
        """Rescales the image by subtracting the real part of the field before
        multiplication.

        Parameters
        ----------
        image: Image or ndarray
            The image to rescale.
        rescale: float
            The rescaling factor.
        **kwargs: dict of str to Any
            Additional keyword arguments.

        Returns
        -------
        Image or ndarray
            The rescaled image.

        """

        image = np.array(image)
        image[..., 0] = (image[..., 0] - 1) * rescale + 1
        image[..., 1] *= rescale

        return image


class FourierTransform(Feature):
    """Computes the Fourier transform of an optical field with optional 
    symmetric padding.

    The Fourier transform converts a spatial-domain optical field into 
    the frequency domain.

    Parameters
    ----------
    padding: int, optional
        Number of pixels to pad symmetrically around the image (default: 32).

    Methods
    -------
    `get(image: Image | np.ndarray, padding: int, **kwargs: dict[str, Any]) -> np.ndarray`
        Computes the 2D Fourier transform of the input image.

    Returns
    -------
    np.ndarray
        The complex Fourier-transformed image.

    Notes
    -----
    - Uses `np.fft.fft2` for fast computation.
    - Pads the image symmetrically to avoid edge artifacts.
    - Returns a complex-valued result.
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(
        self: FourierTransform,
        image: Image | np.ndarray,
        padding: int = 32,
        **kwargs: dict[str, Any],
    ) -> np.ndarray: 
        """Computes the Fourier transform of the image.

        Parameters
        ----------
        image: Image or ndarray
            The image to transform.
        padding: int, optional
            Number of pixels to pad symmetrically around the image (default is 32).
        **kwargs: dict of str to Any

        Returns
        -------
        np.ndarray
            The Fourier transform of the image.
        
        """

        im = np.copy(image[..., 0] + 1j * image[..., 1])
        im = np.pad(
            im,
            ((padding, padding), (padding, padding)),
            mode="symmetric"
            )
        f1 = np.fft.fft2(im)
        return f1


class InverseFourierTransform(Feature):
    """Applies a power of the forward or inverse propagation matrix to an 
    optical field.

    This operation simulates multiple propagation steps in Fourier optics.
    Negative values of `i` apply the inverse transformation.

    Parameters
    ----------
    Tz: np.ndarray
        Forward propagation matrix.
    Tzinv: np.ndarray
        Inverse propagation matrix.
    i: int
        Power of the propagation matrix to apply. Negative values apply the
        inverse.

    Methods
    -------
    `get(image: Image | np.ndarray, padding: int, **kwargs: dict[str, Any]) -> np.ndarray`
        Applies the power of the propagation matrix to the image.

    Returns
    -------
    Image | np.ndarray
        The transformed image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> Tz = np.random.rand(128, 128) + 1j * np.random.rand(128, 128)
    >>> Tzinv = 1 / Tz
    >>> field = np.random.rand(128, 128, 2)
    >>> transformed_field = dt.holography.FourierTransformTransformation(
    >>>     Tz, Tzinv, i=2,
    >>> )(field)

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(
        self: InverseFourierTransform,
        image: Image | np.ndarray,
        padding: int = 32,
        **kwargs: dict[str, Any],
    ) -> Image | np.ndarray:
        """Computes the inverse Fourier transform and removes padding.

        Parameters
        ----------
        image: Image or ndarray
            The image to transform.
        padding: int, optional
            Number of pixels removed symmetrically after inverse transformation
            (default is 32).
        **kwargs: dict of str to Any

        Returns
        -------
        np.ndarray
            The inverse Fourier transform of the image.

        """

        im = np.fft.ifft2(image)
        imnew = np.zeros(
            (image.shape[0] - padding * 2, image.shape[1] - padding * 2, 2)
        )
        imnew[..., 0] = np.real(im[padding:-padding, padding:-padding])
        imnew[..., 1] = np.imag(im[padding:-padding, padding:-padding])
        return imnew


class FourierTransformTransformation(Feature):
    """Applies a power of the forward or inverse propagation matrix to an 
    optical field.

    Parameters
    ----------
    Tz: ndarray
        Forward propagation matrix.
    Tzinv: ndarray
        Inverse propagation matrix.
    i: int
        Power of the propagation matrix to apply. Negative values apply the
        inverse.

    Methods
    -------
    `get(image: Image | np.ndarray, Tz: np.ndarray, Tzinv: np.ndarray, i: int, **kwargs: dict[str, Any]) -> Image | np.ndarray`
        Applies the power of the propagation matrix to the image.

    Returns
    -------
    Image | np.ndarray
        The transformed image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> Tz = np.random.rand(128, 128) + 1j * np.random.rand(128, 128)
    >>> Tzinv = 1 / Tz
    >>> field = np.random.rand(128, 128, 2)
    >>> transformed_field = dt.holography.FourierTransformTransformation(
    >>>     Tz, Tzinv, i=2,
    >>> )(field)

    """

    def __init__(self, Tz, Tzinv, i, **kwargs):
        super().__init__(Tz=Tz, Tzinv=Tzinv, i=i, **kwargs)

    def get(
        self: FourierTransformTransformation,
        image: Image | np.ndarray,
        Tz: np.ndarray,
        Tzinv: np.ndarray,
        i: int,
        **kwargs: dict[str, Any],
    ) -> Image | np.ndarray:
        """Applies the power of the propagation matrix to the image.

        Parameters
        ----------
        image: Image or ndarray
            The image to transform.
        Tz: np.ndarray
            Forward propagation matrix.
        Tzinv: np.ndarray
            Inverse propagation matrix.
        i: int
            Power of the propagation matrix to apply. Negative values apply the
            inverse.
        **kwargs: dict of str to Any
            Additional keyword arguments.
        
        Returns
        -------
        Image or ndarray
            The transformed image.

        """

        if i < 0:
            image *= Tzinv ** np.abs(i)
        else:
            image *= Tz ** i
        return image
