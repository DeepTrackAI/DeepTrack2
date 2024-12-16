"""Bessel and Riccati-Bessel polynomials.

This file defines a set of functions for computing Bessel and Riccati-Bessel 
polynomials and their derivatives. It expands the capabilities of `scipy`.

Functions
---------
besselj(
    l: Union[int, float],
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the Bessel polynomial of the first kind for a given order `l`
    and input `x`.

dbesselj(
    l: Union[int, float],
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the first derivative of the Bessel polynomial of the first kind.

bessely(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the Bessel polynomial of the second kind for a given order `l`
    and input `x`.

dbessely(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the first derivative of the Bessel polynomial of the second kind.

ricbesj(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the Riccati-Bessel polynomial of the first kind.

dricbesj(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the first derivative of the Riccati-Bessel polynomial of the
    first kind.

ricbesy(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the Riccati-Bessel polynomial of the second kind.

dricbesy(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the first derivative of the Riccati-Bessel polynomial of the
    second kind.

ricbesh(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the Riccati-Bessel polynomial of the third kind.

dricbesh(
    l: Union[int, float], 
    x: Union[int, float, np.ndarray]
) -> Union[float, np.ndarray]
    Computes the first derivative of the Riccati-Bessel polynomial of the
    third kind.
    
"""

from typing import Union

import numpy as np
from scipy.special import jv, h1vp, yv


def besselj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Bessel polynomial of the first kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return jv(l, x)


def dbesselj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Bessel polynomial of the first kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return 0.5 * (besselj(l - 1, x) - besselj(l + 1, x))


def bessely(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Bessel polynomial of the second kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return yv(l, x)


def dbessely(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Bessel polynomial of the second kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return 0.5 * (bessely(l - 1, x) - bessely(l + 1, x))


def ricbesj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Riccati-Bessel polynomial of the first kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return np.sqrt(np.pi * x / 2) * besselj(l + 0.5, x)


def dricbesj(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Riccati-Bessel polynomial of the first
    kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return 0.5 * np.sqrt(np.pi / x / 2) * besselj(l + 0.5, x) + np.sqrt(
        np.pi * x / 2
    ) * dbesselj(l + 0.5, x)


def ricbesy(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Riccati-Bessel polynomial of the second kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return -np.sqrt(np.pi * x / 2) * bessely(l + 0.5, x)


def dricbesy(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Riccati-Bessel polynomial of the second
    kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return -0.5 * np.sqrt(np.pi / 2 / x) * yv(l + 0.5, x) - np.sqrt(
        np.pi * x / 2
    ) * dbessely(l + 0.5, x)


def ricbesh(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The Riccati-Bessel polynomial of the third kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    return np.sqrt(np.pi * x / 2) * h1vp(l + 0.5, x, False)


def dricbesh(
        l: Union[int, float],
        x: Union[int, float, np.ndarray],
    ) -> Union[float, np.ndarray]:
    """The first derivative of the Riccati-Bessel polynomial of the third kind.

    Parameters
    ----------
    l : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x

    """

    xi = 0.5 * np.sqrt(np.pi / 2 / x) * h1vp(l + 0.5, x, False) + np.sqrt(
        np.pi * x / 2
    ) * h1vp(l + 0.5, x, True)
    return xi
