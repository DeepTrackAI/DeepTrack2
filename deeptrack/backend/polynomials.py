"""Bessel and Riccati-Bessel polynomials.

This module defines a set of functions for computing Bessel and Riccati-Bessel 
polynomials and their derivatives. It expands the corresponding capabilities of 
`scipy`.

Module Structure
-----------------
Functions:

- `besselj`: Bessel polynomial of the 1st kind.
- `dbesselj`: First derivative of the Bessel polynomial of the 1st kind.
- `bessely`: Bessel polynomial of the 2nd kind.
- `dbessely`: First derivative of the Bessel polynomial of the 2nd kind.
- `ricbesj`: Riccati-Bessel polynomial of the 1st kind.
- `dricbesj`: First derivative of the Riccati-Bessel polynomial of the 1st kind.
- `ricbesy`: Riccati-Bessel polynomial of the 2nd kind.
- `dricbesy`: First derivative of the Riccati-Bessel polynomial of the 2nd kind.
- `ricbesh`: Riccati-Bessel polynomial of the 3rd kind.
- `dricbesh`: First derivative of the Riccati-Bessel polynomial of the 3rd kind.
    
"""

from typing import Union

import numpy as np
from scipy.special import jv, h1vp, yv


def besselj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Bessel polynomial of the 1st kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return jv(l, x)


def dbesselj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Bessel polynomial of the 1st kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return 0.5 * (besselj(l - 1, x) - besselj(l + 1, x))


def bessely(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Bessel polynomial of the 2nd kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return yv(l, x)


def dbessely(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Bessel polynomial of the 2nd kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return 0.5 * (bessely(l - 1, x) - bessely(l + 1, x))


def ricbesj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Riccati-Bessel polynomial of the 1st kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return np.sqrt(np.pi * x / 2) * besselj(l + 0.5, x)


def dricbesj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Riccati-Bessel polynomial of the 1st kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return 0.5 * np.sqrt(np.pi / x / 2) * besselj(l + 0.5, x) + np.sqrt(
        np.pi * x / 2
    ) * dbesselj(l + 0.5, x)


def ricbesy(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Riccati-Bessel polynomial of the 2nd kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return -np.sqrt(np.pi * x / 2) * bessely(l + 0.5, x)


def dricbesy(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Riccati-Bessel polynomial of the 2nd kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return -0.5 * np.sqrt(np.pi / 2 / x) * yv(l + 0.5, x) - np.sqrt(
        np.pi * x / 2
    ) * dbessely(l + 0.5, x)


def ricbesh(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Riccati-Bessel polynomial of the 3rd kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    return np.sqrt(np.pi * x / 2) * h1vp(l + 0.5, x, False)


def dricbesh(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Riccati-Bessel polynomial of the 3rd kind.

    Parameters
    ----------
    l : int or float
        Polynomial order.
    x : int or float or np.ndarray
        The point(s) where the polynomial is evaluated.

    Returns
    -------
    float or np.ndarray
        The polynomial evaluated at x.

    """

    xi = 0.5 * np.sqrt(np.pi / 2 / x) * h1vp(l + 0.5, x, False) + np.sqrt(
        np.pi * x / 2
    ) * h1vp(l + 0.5, x, True)
    return xi
