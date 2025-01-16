"""Features that aberrate and modify pupil functions.

This module provides tools to simulate optical aberrations in microscopy 
by modifying the pupil function of an image. These aberrations can be used 
to study and model the effects of real-world optical imperfections.

Key Features
------------
The module allows simulation of both amplitude and phase aberrations, 
including specific common aberrations, through a set of modular classes:

- **Amplitude Aberrations**
    
    Modulate the intensity profile of the pupil function:
        - `GaussianApodization`: Introduces Gaussian pupil apodization to 
        reduce the amplitude at higher spatial frequencies.

- **Phase Aberrations**
    
    Introduce phase shifts in the pupil function using Zernike polynomials:
        - `Zernike`: Adds phase aberrations based on user-defined Zernike 
        coefficients.

- **Common Aberrations**
    
    Implements commonly encountered Zernike phase aberrations for convenience:
        - `Piston`: Uniform phase shift (n=0, m=0).
        - `VerticalTilt`: Linear tilt along the y-axis (n=1, m=-1).
        - `HorizontalTilt`: Linear tilt along the x-axis (n=1, m=1).
        - `ObliqueAstigmatism`: Oblique astigmatism (n=2, m=-2).
        - `Defocus`: Defocus (n=2, m=0).
        - `Astigmatism`: Regular astigmatism (n=2, m=2).
        - `ObliqueTrefoil`: Oblique trefoil (n=3, m=-3).
        - `VerticalComa`: Vertical coma (n=3, m=-1).
        - `HorizontalComa`: Horizontal coma (n=3, m=1).
        - `Trefoil`: Trefoil aberration (n=3, m=3).
        - `SphericalAberration`: Spherical aberration (n=4, m=0).

Module Structure
----------------
Classes:

- `Aberration`: Base class for all aberrations.

- `GaussianApodization`: Implements pupil apodization.

- `Zernike`: Adds phase aberrations using Zernike polynomials.

- Specific Zernike-based aberration subclasses, e.g., `Defocus`, 
    `Astigmatism`, etc.

Examples
--------
Applying Gaussian Apodization

>>> import deeptrack as dt

>>> particle = dt.PointParticle(position=(32, 32))
>>> aberrated_optics = dt.Fluorescence(
>>>     NA=0.6,
>>>     resolution=1e-7,
>>>     magnification=1,
>>>     wavelength=530e-9,
>>>     output_region=(0, 0, 64, 48),
>>>     padding=(64, 64, 64, 64),
>>>     aberration=aberrations.GaussianApodization(sigma=0.9),
>>>     z = -1.0 * dt.units.micrometer,
>>> )
>>> aberrated_particle = aberrated_optics(particle)
>>> aberrated_particle.plot(cmap="gray")

"""

from typing import List, Tuple, Dict, Any, Union

import numpy as np
import math

from .features import Feature
from .types import PropertyLike
from .utils import as_list


class Aberration(Feature):
    """Base class for optical aberrations.

    This class represents a generic optical aberration. It computes the 
    radial (rho) and angular (theta) pupil coordinates for each input image, 
    normalizes rho by the maximum value within the non-zero region of the 
    image, and passes these coordinates for further processing.

    Parameters
    ----------
    Some common parameters inherited from Feature, such as `sigma`, `offset`, 
    etc., depending on the specific subclass.

    Attributes
    ----------
    __distributed__: bool
        Indicates that the feature can be distributed across multiple 
        processing units.

    Methods
    -------
    `_process_and_get(image_list: List[np.ndarray], **kwargs: dict) -> List[np.ndarray]`
        Processes a list of input images to compute pupil coordinates (rho and
        theta) and passes them, along with the original images, to the 
        superclass method for further processing.

    """

    __distributed__ = True

    def _process_and_get(
        self: Feature,
        image_list: List[np.ndarray],
        **kwargs: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """Computes pupil coordinates.
        
        Computes pupil coordinates (rho and theta) for each input image and 
        processes the images along with these coordinates.

        Parameters
        ----------
        image_list: List[np.ndarray]
            A list of 2D input images to be processed.
        **kwargs: Dict[str, np.ndarray]
            Additional parameters to be passed to the superclass's 
            `_process_and_get` method.

        Returns
        -------
        list: List[np.ndarray]
            A list of processed images with added pupil coordinates.

        """

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


class GaussianApodization(Aberration):
    """Introduces pupil apodization.

    This class modifies the amplitude of the pupil function to decrease 
    progressively at higher spatial frequencies, following a Gaussian 
    distribution. The apodization helps simulate the effects of optical 
    attenuation at the edges of the pupil.

    Parameters
    ----------
    sigma: float, optional
        The standard deviation of the Gaussian apodization. Defines how 
        quickly the amplitude decreases towards the pupil edges. A smaller 
        value leads to a more rapid decay. The default is 1.
    offset: tuple of float, optional
        Specifies the (x, y) offset of the Gaussian center relative to the 
        pupil's geometric center. The default is (0, 0).

    Methods
    -------
    `get(pupil: np.ndarray, offset: Tuple[float, float], sigma: float, rho: np.ndarray, **kwargs: Dict[str, Any]) -> np.ndarray`
        Applies Gaussian apodization to the input pupil function.

    Examples
    --------
    Apply Gaussian apodization to a simulated fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z = 2 * dt.units.micrometer)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil = dt.GaussianApodization(sigma=0.5),
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")

    """

    def __init__(
        self: 'GaussianApodization',
        sigma: PropertyLike[float] = 1,
        offset: PropertyLike[Tuple[int, int]] = (0, 0),
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the GaussianApodization class.
 
        Initializes the GaussianApodization class with parameters that control
        the Gaussian distribution applied to the pupil function.

        Parameters
        ----------
        sigma: float, optional
            The standard deviation of the Gaussian apodization. A smaller
            value results in more rapid attenuation at the edges. Default is 1.
        offset: tuple of float, optional
            The (x, y) coordinates of the Gaussian center's offset relative
            to the geometric center of the pupil. Default is (0, 0).
        **kwargs: dict, optional
            Additional parameters passed to the parent class `Aberration`.

        """

        super().__init__(sigma=sigma, offset=offset, **kwargs)

    def get(
        self: 'GaussianApodization', 
        pupil: np.ndarray, 
        offset: Tuple[float, float], 
        sigma: float, 
        rho: np.ndarray, 
        **kwargs: Dict[str, Any]
    ) -> np.ndarray:
        """Applies Gaussian apodization to the input pupil function.

        This method attenuates the amplitude of the pupil function based 
        on a Gaussian distribution, where the amplitude decreases as the 
        distance from the Gaussian center increases.
        
        Parameters
        ----------
        pupil: np.ndarray
            A 2D array representing the input pupil function.
        offset: tuple of float
            Specifies the (x, y) offset of the Gaussian center relative 
            to the pupil's center.
        sigma: float
            The standard deviation of the Gaussian apodization.
        rho: np.ndarray
            A 2D array of radial coordinates normalized to the pupil 
            aperture.
        **kwargs: dict, optional
            Additional parameters for compatibility with other features 
            or inherited methods. These are typically passed by the 
            parent class and may include:
            - `z` (float): The depth or axial position of the image, 
              used in certain contexts.

        Returns
        -------
        np.ndarray
            The modified pupil function after applying Gaussian apodization.

        Examples
        --------
        >>> import deeptrack as dt
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> pupil = np.ones((128, 128))
        >>> rho = np.linspace(0, 1, 128).reshape(-1, 1) @ np.ones((1, 128))
        >>> x = np.linspace(-1, 1, 128)
        >>> y = np.linspace(-1, 1, 128)
        >>> X, Y = np.meshgrid(x, y)
        >>> rho = np.sqrt(X**2 + Y**2) 
        >>> pupil[rho > 1] = 0
        >>> apodizer = dt.GaussianApodization(sigma=0.5, offset=(25, -3))
        >>> modified_pupil = apodizer.get(
        >>>     pupil, 
        >>>     offset=(5, -3), 
        >>>     sigma=0.5, 
        >>>     rho=rho
        >>> )
        >>> fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        >>> axes[0].imshow(np.abs(modified_pupil), cmap="gray")
        >>> axes[0].set_title("Modified Pupil Magnitude")
        >>> axes[1].imshow(np.angle(modified_pupil), cmap="hsv")
        >>> axes[1].set_title("Modified Pupil Phase")
        >>> plt.show()
        >>> modified_pupil.shape
        (128, 128)

        """

        if offset != (0, 0):
            x = np.arange(pupil.shape[0]) - pupil.shape[0] / 2 - offset[0]
            y = np.arange(pupil.shape[1]) - pupil.shape[1] / 2 - offset[1]
            X, Y = np.meshgrid(x, y)
            rho = np.sqrt(X ** 2 + Y ** 2)
            rho /= np.max(rho[pupil != 0])
            rho[rho > 1] = np.inf

        pupil = pupil * np.exp(-((rho / sigma) ** 2))
        return pupil

class Zernike(Aberration):
    """Introduces a Zernike phase aberration.

    This class applies Zernike polynomial-based phase aberrations to an input 
    pupil function. The Zernike polynomials are used to model various optical 
    aberrations such as defocus, astigmatism, and coma.

    The Zernike polynomial is defined by the radial index `n` and the azimuthal 
    index `m`. The phase contribution is weighted by a specified `coefficient`. 
    When multiple values are provided for `n`, `m`, and `coefficient`, the 
    corresponding Zernike polynomials are summed and applied to the pupil 
    phase.

    Parameters
    ----------
    n: PropertyLike[Union[int, List[int]]]
        The radial index or indices of the Zernike polynomials.
    m: PropertyLike[Union[int, List[int]]]
        The azimuthal index or indices of the Zernike polynomials.
    coefficient: PropertyLike[Union[float, List[float]]]
        The scaling coefficient(s) for the Zernike polynomials.

    Attributes
    ----------
    n: PropertyLike[Union[int, List[int]]]
        The radial index or indices of the Zernike polynomials.
    m: PropertyLike[Union[int, List[int]]]
        The azimuthal index or indices of the Zernike polynomials.
    coefficient: PropertyLike[Union[float, List[float]]]
        The scaling coefficient(s) for the Zernike polynomials.

    Methods
    -------
    `get(pupil: np.ndarray, rho: np.ndarray, theta: np.ndarray, n: Union[int, List[int]], m: Union[int, List[int]], coefficient: Union[float, List[float]], **kwargs: Dict[str, Any]) -> np.ndarray`
        Applies the Zernike phase aberration to the input pupil function.
    
    Notes
    -----
    The Zernike polynomials are normalized to ensure orthogonality. The phase 
    aberration is added in the form of a complex exponential.

    Examples
    --------
    Apply Zernike polynomial-based phase aberrations to a simulated 
    fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z = 1 * dt.units.micrometer)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=dt.Zernike(
    >>>         n=[0, 1], 
    >>>         m = [1, 2], 
    >>>        coefficient=[1, 1]
    >>>     )
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")

    """

    def __init__(
        self: "Zernike",
        n: PropertyLike[Union[int, List[int]]],
        m: PropertyLike[Union[int, List[int]]],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any]
    ) -> None:
        """ Initializes the Zernike class. 
        
        Initializes the Zernike class with the specified indices and coefficients 
        for the Zernike polynomials.

        Parameters
        ----------
        n: int or list of ints
            The radial indices of the Zernike polynomials.
        m: int or list of ints
            The azimuthal indices of the Zernike polynomials.
        coefficient: float or list of floats, optional
            The coefficients for the Zernike polynomials. These determine the 
            relative contribution of each polynomial. Default is 1.
        **kwargs: dict, optional
            Additional parameters passed to the parent class `Aberration`.

        Notes
        -----
        The `n`, `m`, and `coefficient` parameters must have the same length if 
        provided as lists.
        
        """

        super().__init__(n=n, m=m, coefficient=coefficient, **kwargs)

    def get(
        self: "Zernike",
        pupil: np.ndarray,
        rho: np.ndarray,
        theta: np.ndarray,
        n: Union[int, List[int]],
        m: Union[int, List[int]],
        coefficient: Union[float, List[float]],
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Applies the Zernike phase aberration to the input pupil function.

        The method calculates Zernike polynomials for the specified indices `n`
        and `m`, scales them by `coefficient`, and adds the resulting phase to
        the input pupil function. Multiple polynomials are summed if `n`, `m`,
        and `coefficient` are provided as lists.

        Parameters
        ----------
        pupil: np.ndarray
            A 2D array representing the input pupil function. The values should 
            represent the amplitude and phase across the aperture.
        rho: np.ndarray
            A 2D array of radial coordinates normalized to the pupil aperture. 
            The values should range from 0 to 1 within the aperture.
        theta: np.ndarray
            A 2D array of angular coordinates in radians. These define the 
            azimuthal positions for the pupil.
        n: int or list of ints
            The radial indices of the Zernike polynomials.
        m: int or list of ints
            The azimuthal indices of the Zernike polynomials.
        coefficient: float or list of floats
            The coefficients for the Zernike polynomials, controlling their 
            relative contributions to the phase.
        **kwargs: dict, optional
            Additional parameters for compatibility with other features or 
            inherited methods.

        Returns
        -------
        np.ndarray
            The modified pupil function with the applied Zernike phase 
            aberration.

        Raises
        ------
        AssertionError
            If the lengths of `n`, `m`, and `coefficient` lists do not match.

        Notes
        -----
        - The method first calculates the Zernike polynomials for each 
        combination of `n` and `m` and scales them by the corresponding 
        `coefficient`.
        - The resulting polynomials are summed and converted into a phase 
        factor, which is applied to the pupil.

        Examples
        --------
        >>> import deeptrack as dt
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> pupil = np.ones((128, 128), dtype=complex)
        >>> x = np.linspace(-1, 1, 128)
        >>> y = np.linspace(-1, 1, 128)
        >>> X, Y = np.meshgrid(x, y)
        >>> rho = np.sqrt(X**2 + Y**2)  
        >>> theta = np.arctan2(Y, X) 
        >>> pupil[rho > 1] = 0 
        >>> n = [2, 3]
        >>> m = [0, 1]
        >>> coefficient = [0.5, 0.3]
        >>> zernike = dt.Zernike(n=n, m=m, coefficient=coefficient)
        >>> modified_pupil = zernike.get(pupil, rho, theta, n, m, coefficient)
        >>> fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        >>> axes[0].imshow(np.abs(modified_pupil), cmap="gray")
        >>> axes[0].set_title("Modified Pupil Magnitude")
        >>> axes[1].imshow(np.angle(modified_pupil), cmap="hsv")
        >>> axes[1].set_title("Modified Pupil Phase")
        >>> plt.show()

        """

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
                    * math.factorial(n - k)
                    / (
                        math.factorial(k)
                        * math.factorial((n - m) // 2 - k)
                        * math.factorial((n + m) // 2 - k)
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

class Piston(Zernike):
    """Zernike polynomial with n=0, m=0.

    This class represents the simplest Zernike polynomial, often referred to as the piston term, 
    which has no radial or azimuthal variations (n=0, m=0). It adds a uniform phase contribution 
    to the pupil function.

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.

    Attributes
    ----------
    n: int
        The radial index of the Zernike polynomial (always 0 for Piston).
    m: int
        The azimuthal index of the Zernike polynomial (always 0 for Piston).
    coefficient: PropertyLike[float or List[float]]
        The coefficient of the polynomial.

    Examples
    --------
    Apply a Piston Zernike phase aberration (n=0, m=0) to a simulated 
    fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z=1 * dt.units.micrometer)
    >>> piston_aberration = dt.Piston(coefficient=0.9)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=piston_aberration,
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")
    
    """

    def __init__(
        self: "Piston", 
        *args: Tuple[Any, ...], 
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the Piston class.

        Parameters
        ----------
        coefficient: float or list of floats, optional
            The coefficient for the piston term. Default is 1.
        *args: tuple, optional
            Additional arguments passed to the parent Zernike class.
        **kwargs: dict, optional
            Additional parameters passed to the parent Zernike class.
        
        """
        
        super().__init__(*args, n=0, m=0, coefficient=coefficient, **kwargs)


class VerticalTilt(Zernike):
    """Zernike polynomial with n=1, m=-1.

    This class represents a Zernike polynomial corresponding to a vertical tilt 
    aberration. It introduces a linear phase variation across the aperture 
    aligned with the vertical axis.

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.

    Attributes
    ----------
    n: int
        The radial index of the Zernike polynomial (always 1 for VerticalTilt).
    m: int
        The azimuthal index of the Zernike polynomial (always -1 for VerticalTilt).
    coefficient: PropertyLike[float or List[float]]
        The coefficient of the polynomial.

    Examples
    --------
    Apply a VerticalTilt Zernike phase aberration (n=1, m=-1) to a simulated 
    fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z=1 * dt.units.micrometer)
    >>> vertical_tilt_aberration = dt.VerticalTilt(coefficient=-10)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=vertical_tilt_aberration,
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")
    """

    def __init__(
        self: "VerticalTilt", 
        *args: Tuple[Any, ...], 
        coefficient: PropertyLike[Union[float, List[float]]] = 1, 
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the VerticalTilt class.

        Parameters
        ----------
        coefficient: float or list of floats, optional
            The coefficient for the vertical tilt term. Default is 1.
        *args: tuple, optional
            Additional arguments passed to the parent Zernike class.
        **kwargs: dict, optional
            Additional parameters passed to the parent Zernike class.
        """
        super().__init__(*args, n=1, m=-1, coefficient=coefficient, **kwargs)


class HorizontalTilt(Zernike):
    """Zernike polynomial with n=1, m=1.

    This class represents a Zernike polynomial corresponding to a horizontal 
    tilt aberration. It introduces a linear phase variation across the aperture 
    aligned with the horizontal axis.

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.

    Attributes
    ----------
    n: int
        The radial index of the Zernike polynomial (always 1 for 
        HorizontalTilt).
    m: int
        The azimuthal index of the Zernike polynomial (always 1 for 
        HorizontalTilt).
    coefficient: PropertyLike[float or List[float]]
        The coefficient of the polynomial.

    Examples
    --------
    Apply a HorizontalTilt Zernike phase aberration (n=1, m=1) to a simulated 
    fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z=1 * dt.units.micrometer)
    >>> horizontal_tilt_aberration = dt.HorizontalTilt(coefficient=6)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=horizontal_tilt_aberration,
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")
    """

    def __init__(
        self: "HorizontalTilt",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the HorizontalTilt class.

        Parameters
        ----------
        coefficient: float or list of floats, optional
            The coefficient for the horizontal tilt term. Default is 1.
        *args: tuple, optional
            Additional arguments passed to the parent Zernike class.
        **kwargs: dict, optional
            Additional parameters passed to the parent Zernike class.
        """
        super().__init__(*args, n=1, m=1, coefficient=coefficient, **kwargs)



class ObliqueAstigmatism(Zernike):
    """Zernike polynomial with n=2, m=-2.

    This class represents a Zernike polynomial corresponding to oblique 
    astigmatism, characterized by a phase aberration with a radial index of n=2 
    and an azimuthal index of m=-2. It describes astigmatism with axes oriented 
    obliquely to the horizontal and vertical.

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.

    Attributes
    ----------
    n: int
        The radial index of the Zernike polynomial (always 2 for 
        ObliqueAstigmatism).
    m: int
        The azimuthal index of the Zernike polynomial (always -2 for 
        ObliqueAstigmatism).
    coefficient: PropertyLike[float or List[float]]
        The coefficient of the polynomial.

    Examples
    --------
    Apply an ObliqueAstigmatism Zernike phase aberration (n=2, m=-2) to a 
    simulated fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z=4 * dt.units.micrometer)
    >>> oblique_astigmatism_aberration = dt.ObliqueAstigmatism(coefficient=0.2)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=oblique_astigmatism_aberration,
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")
    """

    def __init__(
        self: "ObliqueAstigmatism",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the ObliqueAstigmatism class.

        Parameters
        ----------
        coefficient: float or list of floats, optional
            The coefficient for the oblique astigmatism term. Default is 1.
        *args: tuple, optional
            Additional arguments passed to the parent Zernike class.
        **kwargs: dict, optional
            Additional parameters passed to the parent Zernike class.
        """
        super().__init__(*args, n=2, m=-2, coefficient=coefficient, **kwargs)



class Defocus(Zernike):
    """Zernike polynomial with n=2, m=0.

    This class represents the Zernike polynomial for defocus aberration, 
    characterized by a radial index of n=2 and an azimuthal index of m=0. 
    It describes phase aberrations that result in a uniform spherical defocus 
    across the optical system.

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.

    Attributes
    ----------
    n: int
        The radial index of the Zernike polynomial (always 2 for Defocus).
    m: int
        The azimuthal index of the Zernike polynomial (always 0 for Defocus).
    coefficient: PropertyLike[float or List[float]]
        The coefficient of the polynomial.

    Examples
    --------
    Apply a Defocus Zernike phase aberration (n=2, m=0) to a simulated 
    fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z=0 * dt.units.micrometer)
    >>> defocus_aberration = dt.Defocus(coefficient= 1.5)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=defocus_aberration,
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")
    """

    def __init__(
        self: "Defocus",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the Defocus class.

        Parameters
        ----------
        coefficient: float or list of floats, optional
            The coefficient for the defocus term. Default is 1.
        *args: tuple, optional
            Additional arguments passed to the parent Zernike class.
        **kwargs: dict, optional
            Additional parameters passed to the parent Zernike class.
        """
        super().__init__(*args, n=2, m=0, coefficient=coefficient, **kwargs)



class Astigmatism(Zernike):
    """Zernike polynomial with n=2, m=2.

    This class represents the Zernike polynomial for astigmatism aberration, 
    characterized by a radial index of n=2 and an azimuthal index of m=2. 
    It describes phase aberrations that result in elliptical distortions 
    in the optical system.

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.

    Attributes
    ----------
    n: int
        The radial index of the Zernike polynomial (always 2 for Astigmatism).
    m: int
        The azimuthal index of the Zernike polynomial (always 2 for Astigmatism).
    coefficient: PropertyLike[float or List[float]]
        The coefficient of the polynomial.

    Examples
    --------
    Apply an Astigmatism Zernike phase aberration (n=2, m=2) to a simulated 
    fluorescence image:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(z=1 * dt.units.micrometer)
    >>> astigmatism_aberration = dt.Astigmatism(coefficient=0.75)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=astigmatism_aberration,
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")
    """

    def __init__(
        self: "Astigmatism",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the Astigmatism class.

        Parameters
        ----------
        coefficient: float or list of floats, optional
            The coefficient for the astigmatism term. Default is 1.
        *args: tuple, optional
            Additional arguments passed to the parent Zernike class.
        **kwargs: dict, optional
            Additional parameters passed to the parent Zernike class.
        """
        super().__init__(*args, n=2, m=2, coefficient=coefficient, **kwargs)


class ObliqueTrefoil(Zernike):
    """Zernike polynomial with n=3, m=-3.

    This class represents the Zernike polynomial for oblique trefoil 
    aberration, characterized by a radial index of n=3 and an azimuthal index 
    of m=-3. It describes phase aberrations with triangular symmetry.

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.

    Examples
    --------
    Apply an Oblique Trefoil Zernike phase aberration (n=3, m=-3) to a 
    simulated fluorescence image:

    >>> import deeptrack as dt
    
    >>> particle = dt.PointParticle(z=0 * dt.units.micrometer)
    >>> oblique_trefoil = dt.ObliqueTrefoil(coefficient=1.1)
    >>> aberrated_optics = dt.Fluorescence(
    >>>     pupil=oblique_trefoil,
    >>> )
    >>> aberrated_particle = aberrated_optics(particle)
    >>> aberrated_particle.plot(cmap="gray")
    """

    def __init__(
        self: "ObliqueTrefoil",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, n=3, m=-3, coefficient=coefficient, **kwargs)


class VerticalComa(Zernike):
    """Zernike polynomial with n=3, m=-1.

    This class represents the Zernike polynomial for vertical coma aberration, 
    characterized by a radial index of n=3 and an azimuthal index of m=-1. 

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.
    """

    def __init__(
        self: "VerticalComa",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, n=3, m=-1, coefficient=coefficient, **kwargs)


class HorizontalComa(Zernike):
    """Zernike polynomial with n=3, m=1.

    This class represents the Zernike polynomial for horizontal coma aberration, 
    characterized by a radial index of n=3 and an azimuthal index of m=1. 

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.
    """

    def __init__(
        self: "HorizontalComa",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, n=3, m=1, coefficient=coefficient, **kwargs)


class Trefoil(Zernike):
    """Zernike polynomial with n=3, m=3.

    This class represents the Zernike polynomial for trefoil aberration, 
    characterized by a radial index of n=3 and an azimuthal index of m=3. 

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.
    """

    def __init__(
        self: "Trefoil",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, n=3, m=3, coefficient=coefficient, **kwargs)


class SphericalAberration(Zernike):
    """Zernike polynomial with n=4, m=0.

    This class represents the Zernike polynomial for spherical aberration, 
    characterized by a radial index of n=4 and an azimuthal index of m=0. 

    Parameters
    ----------
    coefficient: PropertyLike[float or List[float]], optional
        The coefficient of the polynomial. Default is 1.
    """

    def __init__(
        self: "SphericalAberration",
        *args: Tuple[Any, ...],
        coefficient: PropertyLike[Union[float, List[float]]] = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(*args, n=4, m=0, coefficient=coefficient, **kwargs)
