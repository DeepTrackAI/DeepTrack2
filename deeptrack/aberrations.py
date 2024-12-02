""" Features that aberrate and modify pupil functions."""

from typing import List, Tuple

import numpy as np

from .features import Feature
from .types import PropertyLike
from .utils import as_list


class Aberration(Feature):
    """Base abstract class.

    Ensures that the method `.get()` receives rho and theta as optional
    arguments, describing the polar coordinates of each pixel in the image
    scaled so that rho is 1 at the edge of the pupil.
    """

    __distributed__ = True

    # Adds rho and theta of the pupil to the input.
    def _process_and_get(self, image_list, **kwargs):
        new_list = []
        for image in image_list:
            x = np.arange(image.shape[0]) - image.shape[0] / 2
            y = np.arange(image.shape[1]) - image.shape[1] / 2
            X, Y = np.meshgrid(y, x)
            rho = np.sqrt(X ** 2 + Y ** 2)
            rho /= np.max(rho[image != 0])
            theta = np.arctan2(Y, X)

            new_list += super()._process_and_get(
                [image], rho=rho, theta=theta, **kwargs
            )
        return new_list


# AMPLITUDE ABERRATIONS


class GaussianApodization(Aberration):
    """Introduces pupil apodization.

    Decreases the amplitude of the pupil at high frequencies according
    to a Gaussian distribution.

    Parameters
    ----------
    sigma : float
        The standard deviation of the apodization. The edge of the pupil
        is at one deviation from the center.
    offset : (float, float)
        Offsets the center of the gaussian.

    Examples
    --------
    >>> particle = dt.PointParticle(z = 1 * dt.units.micrometer)
    >>> aberrated_optics = dt.Fluorescence(aberration=dt.GaussianApodization(sigma=0.1))
    >>> pipeline = aberrated_optics(particle)
    >>> pipeline.plot()
    >>> plt.show()
    """

    def __init__(
        self,
        sigma: PropertyLike[float] = 1,
        offset: PropertyLike[Tuple[int, int]] = (0, 0),
        **kwargs
    ):
        super().__init__(sigma=sigma, offset=offset, **kwargs)

    def get(self, pupil, offset, sigma, rho, **kwargs):
        if offset != (0, 0):
            x = np.arange(pupil.shape[0]) - pupil.shape[0] / 2 - offset[0]
            y = np.arange(pupil.shape[1]) - pupil.shape[1] / 2 - offset[1]
            X, Y = np.meshgrid(x, y)
            rho = np.sqrt(X ** 2 + Y ** 2)
            rho /= np.max(rho[pupil != 0])
            rho[rho > 1] = np.inf

        pupil = pupil * np.exp(-((rho / sigma) ** 2))
        return pupil


# PHASE ABERRATIONS


class Zernike(Aberration):
    """Introduces a Zernike phase aberration.

    Calculates the Zernike polynomial defined by the numbers `n` and `m` at
    each pixel in the pupil, multiplies it by `coefficient`, and adds the
    result to the phase of the pupil.

    If `n`, `m` and `coefficient` are lists of equal lengths, sum the
    Zernike polynomials corresponding to each set of values in these lists
    before adding them to the phase.

    Parameters
    ----------
    n, m : int or list of ints
        The zernike polynomial numbers.
    coefficient : float or list of floats
        The coefficient of the polynomial
    """

    def __init__(
        self,
        n: PropertyLike[int or List[int]],
        m: PropertyLike[int or List[int]],
        coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(n=n, m=m, coefficient=coefficient, **kwargs)

    def get(self, pupil, rho, theta, n, m, coefficient, **kwargs):
        m_list = as_list(m)
        n_list = as_list(n)
        coefficients = as_list(coefficient)

        assert len(m_list) == len(n_list), "The number of indices need to match"
        assert len(m_list) == len(
            coefficients
        ), "The number of indices need to match the number of coefficients"

        pupil_bool = pupil != 0

        rho = rho[pupil_bool]
        theta = theta[pupil_bool]

        Z = 0

        for n, m, coefficient in zip(n_list, m_list, coefficients):
            if (n - m) % 2 or coefficient == 0:
                continue

            R = 0
            for k in range((n - np.abs(m)) // 2 + 1):
                R += (
                    (-1) ** k
                    * np.math.factorial(n - k)
                    / (
                        np.math.factorial(k)
                        * np.math.factorial((n - m) // 2 - k)
                        * np.math.factorial((n + m) // 2 - k)
                    )
                    * rho ** (n - 2 * k)
                )

            if m > 0:
                R = R * np.cos(m * theta) * (np.sqrt(2 * n + 2) * coefficient)
            elif m < 0:
                R = R * np.sin(-m * theta) * (np.sqrt(2 * n + 2) * coefficient)
            else:
                R = R * (np.sqrt(n + 1) * coefficient)

            Z += R

        phase = np.exp(1j * Z)

        pupil[pupil_bool] *= phase

        return pupil


# COMMON ABERRATIONS


class Piston(Zernike):
    """Zernike polynomial with n=0, m=0.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=0, m=0, coefficient=coefficient, **kwargs)


class VerticalTilt(Zernike):
    """Zernike polynomial with n=1, m=-1.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=1, m=-1, coefficient=coefficient, **kwargs)


class HorizontalTilt(Zernike):
    """Zernike polynomial with n=1, m=1.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=1, m=1, coefficient=coefficient, **kwargs)


class ObliqueAstigmatism(Zernike):
    """Zernike polynomial with n=2, m=-2.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=2, m=-2, coefficient=coefficient, **kwargs)


class Defocus(Zernike):
    """Zernike polynomial with n=2, m=0.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=2, m=0, coefficient=coefficient, **kwargs)


class Astigmatism(Zernike):
    """Zernike polynomial with n=2, m=2.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=2, m=2, coefficient=coefficient, **kwargs)


class ObliqueTrefoil(Zernike):
    """Zernike polynomial with n=3, m=-3.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=3, m=-3, coefficient=coefficient, **kwargs)


class VerticalComa(Zernike):
    """Zernike polynomial with n=3, m=-1.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=3, m=-1, coefficient=coefficient, **kwargs)


class HorizontalComa(Zernike):
    """Zernike polynomial with n=3, m=1.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=3, m=1, coefficient=coefficient, **kwargs)


class Trefoil(Zernike):
    """Zernike polynomial with n=3, m=3.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=3, m=3, coefficient=coefficient, **kwargs)


class SphericalAberration(Zernike):
    """Zernike polynomial with n=4, m=0.

    Parameters
    ----------
    coefficient : float
        The coefficient of the polynomial
    """

    def __init__(
        self, *args, coefficient: PropertyLike[float or List[float]] = 1,
        **kwargs
    ):
        super().__init__(*args, n=4, m=0, coefficient=coefficient, **kwargs)
