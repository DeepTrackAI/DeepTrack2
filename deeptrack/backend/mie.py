"""Mie scattering calculations

This module provides functions to perform Mie scattering calculations, 
including computation of spherical harmonics coefficients and related 
operations.

Module Structure
-----------------
Functions:
- `mie_coefficients`: Computes coefficients for spherical harmonics.
- `stratified_mie_coefficients`: Computes coefficients for stratified spherical 
                                 harmonics.
- `mie_harmonics`: Evaluates spherical harmonics of the Mie field.

Example
-------
Calculate Mie coefficients for a single-layer particle:

>>> from mie import mie_coefficients
>>> relative_refract_index = 1.5 + 0.01j
>>> particle_radius = 0.5
>>> max_order = 5

Compute the coefficients:

>>> A, B = mie_coefficients(relative_refract_index, particle_radius, max_order)
>>> print("A coefficients:", A)
>>> print("B coefficients:", B)
"""

from typing import Tuple
import numpy as np
from ._config import cupy
from . import (
    ricbesh,
    ricbesy,
    ricbesj,
    besselj,
    dricbesh,
    dricbesj,
    dricbesy,
    dbesselj,
)


def mie_coefficients(
    m: float or complex, a: float, L: int
) -> Tuple[np.ndarray, np.ndarray]:
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
            (m * Smx * dSx - Sx * dSmx) / (m * Smx * dxix - xix * dSmx)
        )
        B[l - 1] = (
            (Smx * dSx - m * Sx * dSmx) / (Smx * dxix - m * xix * dSmx)
        )

    return A, B


def stratified_mie_coefficients(
    m: list[complex], a: list[float], L: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the Mie scattering coefficients for stratified spherical 
    particles.

    Calculates the terms up to (and including) order L using Riccati-Bessel
    polynomials.

    Parameters
    ----------
    m : list of float or complex
        The relative refractive indices of the particle layers
        (n_particle / n_medium).
    a : list of float
        The radii of the particle layers (> 0).
    L : int
        The maximum order of the spherical harmonics to be calculated.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing arrays of coefficients an and bn, up to (and
        including) order L.

    """
    n_layers = len(a)

    if n_layers == 1:
        return mie_coefficients(m[0], a[0], L)

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
                    A[i, j] = (
                        dricbesj(n + 1, m[p] * a[q])
                        if np.mod(j, 2) == 0
                        else dricbesy(n + 1, m[p] * a[q])
                    )
                    C[i, j] = (
                        m[p] * A[i, j] if j != 2 * n_layers - 1 else A[i, j]
                    )
                else:
                    C[i, j] = (
                        ricbesj(n + 1, m[p] * a[q])
                        if np.mod(j, 2) == 0
                        else ricbesy(n + 1, m[p] * a[q])
                    )
                    A[i, j] = (
                        m[p] * C[i, j] if j != 2 * n_layers - 1 else C[i, j]
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

def mie_harmonics(
    x: np.ndarray, L: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the spherical harmonics of the Mie field.

    The harmonics are calculated up to order L using an iterative method.

    Parameters
    ----------
    x : np.ndarray
        The cosine of the angle defined by the line passing through the origin
        parallel to the propagation direction and the evaluation point, with 
        the corner at the origin.
    L : int
        The order up to which to evaluate the harmonics.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing arrays of harmonics PI and TAU of 
        shape (L, *x.shape).

    """
    if isinstance(x, cupy.ndarray):
        PI = cupy.zeros((L, *x.shape))
        TAU = cupy.zeros((L, *x.shape))
    else:
        PI = np.zeros((L, *x.shape))
        TAU = np.zeros((L, *x.shape))

    PI[0, :] = 1
    PI[1, :] = 3 * x
    TAU[0, :] = x
    TAU[1, :] = 6 * x * x - 3

    for i in range(3, L + 1):
        PI[i - 1] = (
            (2 * i - 1) / (i - 1) * x * PI[i - 2] - i / (i - 1) * PI[i - 3]
        )
        TAU[i - 1] = i * x * PI[i - 1] - (i + 1) * PI[i - 2]

    return PI, TAU