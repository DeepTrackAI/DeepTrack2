""" Expands the set of polynomials available through scipy
"""

import numpy as np
from scipy.special import jv, spherical_jn, h1vp, jvp, yv


def ricbesj(l, x):
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


def dricbesj(l, x):
    """The first derivative of the Riccati-Bessel polynomial of the first kind.

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


def ricbesy(l, x):
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


def dricbesy(l, x):
    """The first derivative of the Riccati-Bessel polynomial of the second kind.

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


def ricbesh(l, x):
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


def dricbesh(l, x):
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


def besselj(l, x):
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


def dbesselj(l, x):
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


def bessely(l, x):
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


def dbessely(l, x):
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