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

Example
-------
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

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT399E

from __future__ import annotations

import array_api_compat as apc
import numpy as np
from numpy.typing import NDArray

from .polynomials import (
    _dricbesh_array_api,
    _dricbesj_array_api,
    _dricbesy_array_api,
    _ricbesh_array_api,
    _ricbesj_array_api,
    _ricbesy_array_api,
    ricbesh,
    ricbesy,
    ricbesj,
    dricbesh,
    dricbesj,
    dricbesy,
)


def _first_array(*values):
    """Return the first array API object in values."""

    for value in values:
        if apc.is_array_api_obj(value):
            return value

        if isinstance(value, (list, tuple)):
            found = _first_array(*value)
            if found is not None:
                return found

    return None


def _array_api_namespace(*values):
    """Return a non-NumPy array API namespace and reference array if present."""

    reference = _first_array(*values)

    if reference is None:
        return None, None

    namespace = apc.array_namespace(reference)

    if apc.is_numpy_namespace(namespace):
        return None, None

    return namespace, reference


def _complex_dtype(namespace, *values):
    """Return a complex dtype compatible with the array inputs."""

    for value in values:
        if apc.is_array_api_obj(value):
            if value.dtype in (namespace.float64, namespace.complex128):
                return namespace.complex128

        if isinstance(value, (list, tuple)):
            dtype = _complex_dtype(namespace, *value)
            if dtype == namespace.complex128:
                return dtype

    return namespace.complex64


def _asarray(value, namespace, dtype, reference):
    """Convert value to an array on the same backend as reference."""

    if apc.is_array_api_obj(value):
        return namespace.astype(value, dtype)

    try:
        return namespace.asarray(
            value, dtype=dtype, device=apc.device(reference)
        )
    except TypeError:
        return namespace.asarray(value, dtype=dtype)


def _asarray_vector(value, namespace, dtype, reference):
    """Convert a tensor or sequence of scalars to a one-dimensional array."""

    if apc.is_array_api_obj(value):
        return namespace.reshape(
            _asarray(value, namespace, dtype, reference), (-1,)
        )

    return namespace.stack(
        [
            namespace.reshape(
                _asarray(element, namespace, dtype, reference), ()
            )
            for element in value
        ]
    )


def _empty(namespace, shape, dtype, reference):
    """Create an empty array on the same backend as reference."""

    try:
        return namespace.empty(
            shape, dtype=dtype, device=apc.device(reference)
        )
    except TypeError:
        return namespace.empty(shape, dtype=dtype)


def _zeros(namespace, shape, dtype, reference):
    """Create a zero array on the same backend as reference."""

    try:
        return namespace.zeros(
            shape, dtype=dtype, device=apc.device(reference)
        )
    except TypeError:
        return namespace.zeros(shape, dtype=dtype)


def _coefficients_array_api(
    m: float | complex,
    a: float,
    L: int,
    namespace,
    reference,
):
    """Array API implementation of Mie coefficients."""

    dtype = _complex_dtype(namespace, m, a)

    m = _asarray(m, namespace, dtype, reference)
    a = _asarray(a, namespace, dtype, reference)

    if L == 0:
        return (
            _empty(namespace, (0,), dtype, reference),
            _empty(namespace, (0,), dtype, reference),
        )

    A = []
    B = []

    for l in range(1, L + 1):
        Sx = _ricbesj_array_api(l, a, namespace)
        dSx = _dricbesj_array_api(l, a, namespace)
        Smx = _ricbesj_array_api(l, m * a, namespace)
        dSmx = _dricbesj_array_api(l, m * a, namespace)
        xix = _ricbesh_array_api(l, a, namespace)
        dxix = _dricbesh_array_api(l, a, namespace)

        A.append(
            (m * Smx * dSx - Sx * dSmx)
            / (m * Smx * dxix - xix * dSmx)
        )
        B.append(
            (Smx * dSx - m * Sx * dSmx)
            / (Smx * dxix - m * xix * dSmx)
        )

    return namespace.stack(A), namespace.stack(B)


def _stratified_coefficients_array_api(
    m: list[complex],
    a: list[float],
    L: int,
    namespace,
    reference,
):
    """Array API implementation of stratified Mie coefficients."""

    dtype = _complex_dtype(namespace, m, a)

    m = _asarray_vector(m, namespace, dtype, reference)
    a = _asarray_vector(a, namespace, dtype, reference)
    n_layers = a.shape[0]

    if n_layers == 1:
        return _coefficients_array_api(
            m[0], a[0], L, namespace, reference
        )

    if L == 0:
        return (
            _empty(namespace, (0,), dtype, reference),
            _empty(namespace, (0,), dtype, reference),
        )

    an = []
    bn = []

    for n in range(L):
        A_rows = []
        C_rows = []
        zero = _zeros(namespace, (), dtype, reference)

        for i in range(2 * n_layers):
            for j in range(2 * n_layers):
                p = (j + 1) // 2
                q = i // 2

                A_ij = zero
                C_ij = zero

                if (p - q == 0) or (p - q == 1):
                    if i % 2 == 0:
                        if (
                            j < 2 * n_layers - 1
                            and (j == 0 or j % 2 == 1)
                        ):
                            A_ij = _dricbesj_array_api(
                                n + 1, m[p] * a[q], namespace
                            )
                        elif j % 2 == 0:
                            A_ij = _dricbesy_array_api(
                                n + 1, m[p] * a[q], namespace
                            )
                        else:
                            A_ij = _dricbesj_array_api(
                                n + 1, a[q], namespace
                            )

                        if j != 2 * n_layers - 1:
                            C_ij = m[p] * A_ij
                        else:
                            C_ij = A_ij
                    else:
                        if (
                            j < 2 * n_layers - 1
                            and (j == 0 or j % 2 == 1)
                        ):
                            C_ij = _ricbesj_array_api(
                                n + 1, m[p] * a[q], namespace
                            )
                        elif j % 2 == 0:
                            C_ij = _ricbesy_array_api(
                                n + 1, m[p] * a[q], namespace
                            )
                        else:
                            C_ij = _ricbesj_array_api(
                                n + 1, a[q], namespace
                            )

                        if j != 2 * n_layers - 1:
                            A_ij = m[p] * C_ij
                        else:
                            A_ij = C_ij

                A_rows.append(A_ij)
                C_rows.append(C_ij)

        shape = (2 * n_layers, 2 * n_layers)
        A = namespace.reshape(namespace.stack(A_rows), shape)
        C = namespace.reshape(namespace.stack(C_rows), shape)

        B = A * 1
        B[-2, -1] = _dricbesh_array_api(n + 1, a[-1], namespace)
        B[-1, -1] = _ricbesh_array_api(n + 1, a[-1], namespace)
        an.append(namespace.linalg.det(A) / namespace.linalg.det(B))

        D = C * 1
        D[-2, -1] = _dricbesh_array_api(n + 1, a[-1], namespace)
        D[-1, -1] = _ricbesh_array_api(n + 1, a[-1], namespace)
        bn.append(namespace.linalg.det(C) / namespace.linalg.det(D))

    return namespace.stack(an), namespace.stack(bn)


def _harmonics_array_api(
    x,
    L: int,
    namespace,
    reference,
):
    """Array API implementation of Mie harmonics."""

    PI = []
    TAU = []

    if L == 0:
        shape = (0, *x.shape)
        return (
            _empty(namespace, shape, x.dtype, reference),
            _empty(namespace, shape, x.dtype, reference),
        )

    if L >= 1:
        PI.append(namespace.ones_like(x))
        TAU.append(x)

    if L >= 2:
        PI.append(3 * x)
        TAU.append(6 * x * x - 3)

    for i in range(3, L + 1):
        PI.append(
            (2 * i - 1) / (i - 1) * x * PI[i - 2]
            - i / (i - 1) * PI[i - 3]
        )
        TAU.append(i * x * PI[i - 1] - (i + 1) * PI[i - 2])

    return namespace.stack(PI), namespace.stack(TAU)


#TODO ***??*** revise coefficients - torch, docstring, unit test
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
    m : float or complex
        The relative refractive index of the particle n_particle / n_medium.
    a : float
        The radius of the particle (> 0).
    L : int
        The maximum order of the spherical harmonics to be calculated.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays of complex numbers, A and B, which
        are the Mie scattering coefficients up to (and including) order L.

    """

    namespace, reference = _array_api_namespace(m, a)

    if namespace is not None:
        return _coefficients_array_api(m, a, L, namespace, reference)

    A = np.zeros((L,), dtype=np.complex128)
    B = np.zeros((L,), dtype=np.complex128)

    for l in range(1, L + 1):
        Sx = ricbesj(l, a)
        dSx = dricbesj(l, a)
        Smx = ricbesj(l, m * a)
        dSmx = dricbesj(l, m * a)
        xix = ricbesh(l, a)
        dxix = dricbesh(l, a)

        A[l - 1] = (
            (m * Smx * dSx - Sx * dSmx) 
            / 
            (m * Smx * dxix - xix * dSmx)
        )
        B[l - 1] = (
            (Smx * dSx - m * Sx * dSmx) 
            / 
            (Smx * dxix - m * xix * dSmx)
        )

    return A, B


#TODO ***??*** revise stratified_coefficients - torch, docstring, unit test
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
    m : List[float or complex]
        The relative refractive indices of the particle layers
        (n_particle / n_medium).
    a : List[float]
        The radii of the particle layers (> 0).
    L : int
        The maximum order of the spherical harmonics to be calculated.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing arrays of coefficients an and bn, up to (and
        including) order L.

    """
    namespace, reference = _array_api_namespace(m, a)

    if namespace is not None:
        return _stratified_coefficients_array_api(
            m, a, L, namespace, reference
        )

    n_layers = len(a)

    if n_layers == 1:
        return coefficients(m[0], a[0], L)

    an = np.zeros((L,), dtype=np.complex128)
    bn = np.zeros((L,), dtype=np.complex128)

    for n in range(L):
        A = np.zeros((2 * n_layers, 2 * n_layers), dtype=np.complex128)
        C = np.zeros((2 * n_layers, 2 * n_layers), dtype=np.complex128)

        for i in range(2 * n_layers):
            for j in range(2 * n_layers):
                p = np.floor((j + 1) / 2).astype(np.int32)
                q = np.floor((i / 2)).astype(np.int32)

                if not ((p - q == 0) or (p - q == 1)):
                    continue

                if np.mod(i, 2) == 0:
                    if (j < 2 * n_layers - 1) and ((j == 0) or
                                                   (np.mod(j, 2) == 1)):
                        A[i, j] = dricbesj(n + 1, m[p] * a[q])
                    elif np.mod(j, 2) == 0:
                        A[i, j] = dricbesy(n + 1, m[p] * a[q])
                    else:
                        A[i, j] = dricbesj(n + 1, a[q])

                    C[i, j] = (
                        m[p] * A[i, j]
                        if j != 2 * n_layers - 1
                        else A[i, j]
                    )
                else:
                    if (j < 2 * n_layers - 1) and ((j == 0) or
                                                   (np.mod(j, 2) == 1)):
                        C[i, j] = ricbesj(n + 1, m[p] * a[q])
                    elif np.mod(j, 2) == 0:
                        C[i, j] = ricbesy(n + 1, m[p] * a[q])
                    else:
                        C[i, j] = ricbesj(n + 1, a[q])

                    A[i, j] = (
                        m[p] * C[i, j]
                        if j != 2 * n_layers - 1 
                        else C[i, j]
                    )

        B = A.copy()
        B[-2, -1] = dricbesh(n + 1, a[-1])
        B[-1, -1] = ricbesh(n + 1, a[-1])
        an[n] = np.linalg.det(A) / np.linalg.det(B)

        D = C.copy()
        D[-2, -1] = dricbesh(n + 1, a[-1])
        D[-1, -1] = ricbesh(n + 1, a[-1])
        bn[n] = np.linalg.det(C) / np.linalg.det(D)

    return an, bn


#TODO ***??*** revise harmonics - torch, docstring, unit test
def harmonics(
    x: NDArray,
    L: int,
) -> tuple[NDArray, NDArray]:
    """Calculate the spherical harmonics of the Mie field.

    The harmonics are calculated up to order L using an iterative method.

    Parameters
    ----------
    x : np.ndarray
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

    L : int
        The order up to which to evaluate the harmonics.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing arrays of harmonics PI and TAU of 
        shape (L, *x.shape).

    """

    namespace, reference = _array_api_namespace(x)

    if namespace is not None:
        return _harmonics_array_api(x, L, namespace, reference)

    PI = np.zeros((L, *x.shape))
    TAU = np.zeros((L, *x.shape))

    if L >= 1:
        PI[0, :] = 1
        TAU[0, :] = x

    if L >= 2:
        PI[1, :] = 3 * x
        TAU[1, :] = 6 * x * x - 3

    for i in range(3, L + 1):
        PI[i - 1] = (
            (2 * i - 1) / (i - 1) * x * PI[i - 2] - i / (i - 1) * PI[i - 3]
        )
        TAU[i - 1] = i * x * PI[i - 1] - (i + 1) * PI[i - 2]

    return PI, TAU
