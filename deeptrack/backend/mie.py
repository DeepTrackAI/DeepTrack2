"""Mie scattering calculations.

This module provides functions to perform Mie scattering calculations,
including computation of spherical harmonics coefficients and related
operations.

Module Structure
-----------------
Functions:

- `coefficients`: Coefficients for spherical harmonics.
- `stratified_coefficients`: Coefficients for stratified spherical harmonics.
- `harmonics`: Evaluates spherical harmonics of the Mie field.

Examples
--------
Define the parameters of the particle and the Mie scattering:

>>> relative_refract_index = 1.5 + 0.01j
>>> particle_radius = 0.5
>>> max_order = 5

Calculate Mie coefficients for a solid particle:

>>> from deeptrack.backend import mie
>>> A, B = mie.coefficients(relative_refract_index, particle_radius, max_order)

Print them:

>>> print("A coefficients:", A)
>>> print("B coefficients:", B)

"""

from __future__ import annotations

import array_api_compat as apc
import numpy as np
from numpy.typing import NDArray

from ._config import config, xp
from .polynomials import (
    ricbesh,
    ricbesy,
    ricbesj,
    dricbesh,
    dricbesj,
    dricbesy,
)


def _iter_arrays(*values):
    """Yield array API objects from values, including nested sequences."""

    for value in values:
        if apc.is_array_api_obj(value):
            yield value
        elif isinstance(value, (list, tuple)):
            yield from _iter_arrays(*value)


def _first_array(*values):
    """Return the first array API object in values, if any."""

    return next(_iter_arrays(*values), None)


def _complex_dtype(*values):
    """Return the complex dtype to use for the current xp backend."""

    for value in _iter_arrays(*values):
        if value.dtype in (xp.float64, xp.complex128):
            return xp.complex128

    return xp.get_complex_dtype()


def _asarray(value, dtype=None, reference=None):
    """Convert value with xp without detaching existing arrays."""

    is_current_backend_array = (
        config.get_backend() == "numpy"
        and apc.is_numpy_array(value)
        or config.get_backend() == "torch"
        and apc.is_torch_array(value)
    )

    if is_current_backend_array:
        return xp.astype(value, dtype) if dtype is not None else value

    kwargs = {}

    if dtype is not None:
        kwargs["dtype"] = dtype

    if reference is not None:
        try:
            kwargs["device"] = apc.device(reference)
        except TypeError:
            pass

    try:
        return xp.asarray(value, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return xp.asarray(value, **kwargs)


def _asarray_vector(value, dtype=None, reference=None):
    """Convert a tensor or sequence of scalars to a one-dimensional array."""

    if apc.is_array_api_obj(value):
        return xp.reshape(_asarray(value, dtype, reference), (-1,))

    return xp.stack(
        [
            xp.reshape(_asarray(element, dtype, reference), ())
            for element in value
        ]
    )


def _zeros(shape, dtype, reference=None):
    """Create a zero array on the same backend as reference."""

    kwargs = {"dtype": dtype}

    if reference is not None:
        try:
            kwargs["device"] = apc.device(reference)
        except TypeError:
            pass

    try:
        return xp.zeros(shape, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return xp.zeros(shape, **kwargs)


def coefficients(
    m: float | complex,
    a: float,
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the Mie scattering coefficients for a spherical particle.

    These coefficients are used in the computation of the scattering
    and absorption of light by the particle. The terms up to (and including)
    order L are calculated using Riccati-Bessel polynomials.

    Parameters
    ----------
    m: float | complex
        The relative refractive index of the particle n_particle / n_medium.
    a: float
        The radius of the particle (> 0).
    L: int
        The maximum order of the spherical harmonics to be calculated. If 0,
        two empty arrays are returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays of complex numbers, A and B, which
        are the Mie scattering coefficients up to (and including) order L.

    """

    dtype = _complex_dtype(m, a)
    reference = _first_array(m, a)
    m = _asarray(m, dtype=dtype, reference=reference)
    a = _asarray(a, dtype=dtype, reference=reference)

    if L == 0:
        return (
            _zeros((0,), dtype=dtype, reference=reference),
            _zeros((0,), dtype=dtype, reference=reference),
        )

    A = []
    B = []

    for l in range(1, L + 1):
        Sx = ricbesj(l, a)
        dSx = dricbesj(l, a)
        Smx = ricbesj(l, m * a)
        dSmx = dricbesj(l, m * a)
        xix = ricbesh(l, a)
        dxix = dricbesh(l, a)

        A.append((m * Smx * dSx - Sx * dSmx) / (m * Smx * dxix - xix * dSmx))
        B.append((Smx * dSx - m * Sx * dSmx) / (Smx * dxix - m * xix * dSmx))

    return xp.stack(A), xp.stack(B)


def stratified_coefficients(
    m: list[complex],
    a: list[float],
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the Mie scattering coefficients for stratified spherical
    particles.

    This function calculates the terms up to (and including) order L using
    Riccati-Bessel polynomials.

    Parameters
    ----------
    m: list[complex]
        The relative refractive indices of the particle layers
        (n_particle / n_medium).
    a: list[float]
        The radii of the particle layers (> 0).
    L: int
        The maximum order of the spherical harmonics to be calculated. If 0,
        two empty arrays are returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing arrays of coefficients an and bn, up to (and
        including) order L.

    """

    dtype = _complex_dtype(m, a)
    reference = _first_array(m, a)
    m = _asarray_vector(m, dtype=dtype, reference=reference)
    a = _asarray_vector(a, dtype=dtype, reference=reference)
    n_layers = a.shape[0]

    if n_layers == 1:
        return coefficients(m[0], a[0], L)

    if L == 0:
        return (
            _zeros((0,), dtype=dtype, reference=reference),
            _zeros((0,), dtype=dtype, reference=reference),
        )

    an = []
    bn = []

    for n in range(L):
        A_rows = []
        C_rows = []
        zero = _zeros((), dtype=dtype, reference=reference)

        for i in range(2 * n_layers):
            for j in range(2 * n_layers):
                p = (j + 1) // 2
                q = i // 2
                A_ij = zero
                C_ij = zero

                if (p - q == 0) or (p - q == 1):
                    if i % 2 == 0:
                        if j < 2 * n_layers - 1 and (j == 0 or j % 2 == 1):
                            A_ij = dricbesj(n + 1, m[p] * a[q])
                        elif j % 2 == 0:
                            A_ij = dricbesy(n + 1, m[p] * a[q])
                        else:
                            A_ij = dricbesj(n + 1, a[q])

                        if j != 2 * n_layers - 1:
                            C_ij = m[p] * A_ij
                        else:
                            C_ij = A_ij
                    else:
                        if j < 2 * n_layers - 1 and (j == 0 or j % 2 == 1):
                            C_ij = ricbesj(n + 1, m[p] * a[q])
                        elif j % 2 == 0:
                            C_ij = ricbesy(n + 1, m[p] * a[q])
                        else:
                            C_ij = ricbesj(n + 1, a[q])

                        if j != 2 * n_layers - 1:
                            A_ij = m[p] * C_ij
                        else:
                            A_ij = C_ij

                A_rows.append(A_ij)
                C_rows.append(C_ij)

        shape = (2 * n_layers, 2 * n_layers)
        A = xp.reshape(xp.stack(A_rows), shape)
        C = xp.reshape(xp.stack(C_rows), shape)

        B = A * 1
        B[-2, -1] = dricbesh(n + 1, a[-1])
        B[-1, -1] = ricbesh(n + 1, a[-1])
        an.append(xp.linalg.det(A) / xp.linalg.det(B))

        D = C * 1
        D[-2, -1] = dricbesh(n + 1, a[-1])
        D[-1, -1] = ricbesh(n + 1, a[-1])
        bn.append(xp.linalg.det(C) / xp.linalg.det(D))

    return xp.stack(an), xp.stack(bn)


def harmonics(
    x: np.ndarray,
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the spherical harmonics of the Mie field.

    The harmonics are calculated up to order L using an iterative method.

    Parameters
    ----------
    x: np.ndarray
        An array representing the cosine of the polar angle (theta) for each
        evaluation point relative to the scattering particle's center
        (the origin).
        The polar angle is the angle between the z-axis (aligned with the
        direction of wave propagation) and the vector from the particle's
        center to the evaluation point.

        Values in `x` should lie in the range [-1, 1], where `x = 1`
        corresponds to theta = 0° (point directly forward along the z-axis),
        `x = -1` corresponds to theta = 180° (point directly backward along the
        z-axis), and `x = 0` corresponds to theta = 90° (point perpendicular to
        the z-axis).

    L: int
        The order up to which to evaluate the harmonics. If 0, two empty
        arrays of shape (0, *x.shape) are returned.


    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing arrays of harmonics PI and TAU of
        shape (L, *x.shape).

    """

    x = _asarray(x)
    reference = _first_array(x)

    if L == 0:
        return (
            _zeros((0, *x.shape), dtype=x.dtype, reference=reference),
            _zeros((0, *x.shape), dtype=x.dtype, reference=reference),
        )

    PI = []
    TAU = []

    if L >= 1:
        PI.append(xp.ones_like(x))
        TAU.append(x)

    if L >= 2:
        PI.append(3 * x)
        TAU.append(6 * x * x - 3)

    for i in range(3, L + 1):
        PI.append(
            (2 * i - 1) / (i - 1) * x * PI[i - 2] - i / (i - 1) * PI[i - 3]
        )
        TAU.append(i * x * PI[i - 1] - (i + 1) * PI[i - 2])

    return xp.stack(PI), xp.stack(TAU)
