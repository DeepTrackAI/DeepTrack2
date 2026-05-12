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

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT399D

from __future__ import annotations

import array_api_compat as apc
import numpy as np
from numpy.typing import NDArray
from scipy.special import jv, h1vp, yv


def _integer_order(l: int | float) -> int:
    """Return l as an integer order supported by recurrence formulas."""

    order = int(l)

    if order != l or order < 0:
        raise ValueError(
            "Array API Riccati-Bessel functions require non-negative integer "
            "orders."
        )

    return order


def _array_namespace(x):
    """Return the array namespace for x, or None for Python scalars."""

    try:
        return apc.array_namespace(x)
    except TypeError:
        return None


def _ricbesj_array_api(l: int | float, x, namespace=None):
    """Array-API Riccati-Bessel polynomial of the first kind."""

    l = _integer_order(l)
    xp = namespace or apc.array_namespace(x)

    if l == 0:
        return xp.sin(x)

    previous = xp.sin(x)
    current = xp.sin(x) / x - xp.cos(x)

    for order in range(1, l):
        previous, current = current, (2 * order + 1) / x * current - previous

    return current


def _dricbesj_array_api(l: int | float, x, namespace=None):
    """Array-API derivative of ricbesj."""

    l = _integer_order(l)
    xp = namespace or apc.array_namespace(x)

    if l == 0:
        return xp.cos(x)

    return (
        _ricbesj_array_api(l - 1, x, xp)
        - l / x * _ricbesj_array_api(l, x, xp)
    )


def _ricbesy_array_api(l: int | float, x, namespace=None):
    """Array-API Riccati-Bessel polynomial of the second kind."""

    l = _integer_order(l)
    xp = namespace or apc.array_namespace(x)

    if l == 0:
        return xp.cos(x)

    previous = xp.cos(x)
    current = xp.cos(x) / x + xp.sin(x)

    for order in range(1, l):
        previous, current = current, (2 * order + 1) / x * current - previous

    return current


def _dricbesy_array_api(l: int | float, x, namespace=None):
    """Array-API derivative of ricbesy."""

    l = _integer_order(l)
    xp = namespace or apc.array_namespace(x)

    if l == 0:
        return -xp.sin(x)

    return (
        _ricbesy_array_api(l - 1, x, xp)
        - l / x * _ricbesy_array_api(l, x, xp)
    )


def _ricbesh_array_api(l: int | float, x, namespace=None):
    """Array-API Riccati-Bessel polynomial of the third kind."""

    xp = namespace or apc.array_namespace(x)
    return _ricbesj_array_api(l, x, xp) - 1j * _ricbesy_array_api(l, x, xp)


def _dricbesh_array_api(l: int | float, x, namespace=None):
    """Array-API derivative of ricbesh."""

    xp = namespace or apc.array_namespace(x)
    return _dricbesj_array_api(l, x, xp) - 1j * _dricbesy_array_api(l, x, xp)


#TODO ***??*** revise besselj - torch, docstring, unit test
def besselj(
    l: int | float,
    x: int | float | NDArray,
) -> float | NDArray:
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


#TODO ***??*** revise dbesselj - torch, docstring, unit test
def dbesselj(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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


#TODO ***??*** revise bessely - torch, docstring, unit test
def bessely(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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


#TODO ***??*** revise dbessely - torch, docstring, unit test
def dbessely(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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


#TODO ***??*** revise ricbesj - torch, docstring, unit test
def ricbesj(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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

    namespace = _array_namespace(x)
    if namespace is not None and not apc.is_numpy_namespace(namespace):
        return _ricbesj_array_api(l, x, namespace)

    return np.sqrt(np.pi * x / 2) * besselj(l + 0.5, x)


#TODO ***??*** revise dricbesj - torch, docstring, unit test
def dricbesj(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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

    namespace = _array_namespace(x)
    if namespace is not None and not apc.is_numpy_namespace(namespace):
        return _dricbesj_array_api(l, x, namespace)

    return 0.5 * np.sqrt(np.pi / x / 2) * besselj(l + 0.5, x) + np.sqrt(
        np.pi * x / 2
    ) * dbesselj(l + 0.5, x)


#TODO ***??*** revise ricbesy - torch, docstring, unit test
def ricbesy(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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

    namespace = _array_namespace(x)
    if namespace is not None and not apc.is_numpy_namespace(namespace):
        return _ricbesy_array_api(l, x, namespace)

    return -np.sqrt(np.pi * x / 2) * bessely(l + 0.5, x)


#TODO ***??*** revise dricbesy - torch, docstring, unit test
def dricbesy(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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

    namespace = _array_namespace(x)
    if namespace is not None and not apc.is_numpy_namespace(namespace):
        return _dricbesy_array_api(l, x, namespace)

    return -0.5 * np.sqrt(np.pi / 2 / x) * yv(l + 0.5, x) - np.sqrt(
        np.pi * x / 2
    ) * dbessely(l + 0.5, x)


#TODO ***??*** revise ricbesh - torch, docstring, unit test
def ricbesh(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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

    namespace = _array_namespace(x)
    if namespace is not None and not apc.is_numpy_namespace(namespace):
        return _ricbesh_array_api(l, x, namespace)

    return np.sqrt(np.pi * x / 2) * h1vp(l + 0.5, x, False)


#TODO ***??*** revise dricbesh - torch, docstring, unit test
def dricbesh(
        l: int | float,
        x: int | float | NDArray,
    ) -> float | NDArray:
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

    namespace = _array_namespace(x)
    if namespace is not None and not apc.is_numpy_namespace(namespace):
        return _dricbesh_array_api(l, x, namespace)

    xi = 0.5 * np.sqrt(np.pi / 2 / x) * h1vp(l + 0.5, x, False) + np.sqrt(
        np.pi * x / 2
    ) * h1vp(l + 0.5, x, True)
    return xi
