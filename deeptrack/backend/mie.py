""" Perform Mie-specific calculations
"""

import numpy as np

from typing import Tuple
from deeptrack.backend import (
    ricbesh,
    ricbesy,
    ricbesj,
    besselj,
    dricbesh,
    dricbesj,
    dricbesy,
    dbesselj,
)


def mie_coefficients(m: float or complex, a: float, L: int) -> Tuple[np.ndarray]:
    """Calculate the coefficients of the spherical harmonics.

    Calculates the terms up to (and including) order L using riccati-bessel polynomials.

    Parameters
    ----------
    m : float or complex float
        The relative refractive index of the particle n_particle / n_medium
    a : float
        The radius of the particle (> 0)
    L : int
        The order

    Returns
    -------
    ndarray, ndarray :
        Tuple of ndarrays, containing coefficient terms up to but not including order L.
    """

    A = np.zeros((L,)) * 1j
    B = np.zeros((L,)) * 1j

    for l in range(1, L + 1):

        Sx = ricbesj(l, a)
        dSx = dricbesj(l, a)
        Smx = ricbesj(l, m * a)
        dSmx = dricbesj(l, m * a)
        xix = ricbesh(l, a)
        dxix = dricbesh(l, a)

        A[l - 1] = (m * Smx * dSx - Sx * dSmx) / (m * Smx * dxix - xix * dSmx)
        B[l - 1] = (Smx * dSx - m * Sx * dSmx) / (Smx * dxix - m * xix * dSmx)

    return A, B


def stratified_mie_coefficients(m, a, L):
    """Calculate the coefficients of the stratified spherical harmonics.

    Calculates the terms up to (and including) order L using riccati-bessel polynomials.

    Parameters
    ----------
    k : float
        The wave-vector in the medium
    m : list of float or complex float
        The relative refractive index of the particle n_particle / n_medium
    a : list of float
        The radius of the particle (> 0)
    L : int
        The order

    Returns
    -------
    ndarray, ndarray :
        Tuple of ndarrays, containing coefficient terms up to but not including order L.
    """

    n_layers = len(a)

    if n_layers == 1:
        return mie_coefficients(m[0], a[0], L)

    an = np.zeros((L,)) * 1j
    bn = np.zeros((L,)) * 1j

    for n in range(L):

        A = np.zeros((2 * n_layers, 2 * n_layers)) * 1j
        C = np.zeros((2 * n_layers, 2 * n_layers)) * 1j

        for i in range(2 * n_layers):
            for j in range(2 * n_layers):

                p = np.floor((j + 1) / 2).astype(np.int32)
                q = np.floor((i / 2)).astype(np.int32)

                if not ((p - q == 0) or (p - q == 1)):
                    continue
                # print(p, q, i, j)
                if np.mod(i, 2) == 0:

                    if (j < 2 * n_layers - 1) and ((j == 0) or (np.mod(j, 2) == 1)):
                        A[i, j] = dricbesj(n + 1, m[p] * a[q])
                    elif np.mod(j, 2) == 0:
                        A[i, j] = dricbesy(n + 1, m[p] * a[q])
                    else:
                        A[i, j] = dricbesj(n + 1, a[q])

                    if j != 2 * n_layers - 1:
                        C[i, j] = m[p] * A[i, j]
                    else:
                        C[i, j] = A[i, j]

                else:
                    if (j < 2 * n_layers - 1) and ((j == 0) or (np.mod(j, 2) == 1)):
                        C[i, j] = ricbesj(n + 1, m[p] * a[q])
                    elif np.mod(j, 2) == 0:
                        C[i, j] = ricbesy(n + 1, m[p] * a[q])
                    else:
                        C[i, j] = ricbesj(n + 1, a[q])

                    if j != 2 * n_layers - 1:
                        A[i, j] = m[p] * C[i, j]
                    else:
                        A[i, j] = C[i, j]
        A = np.real(A)
        C = np.real(C)

        B = A + 0j
        B[-2, -1] = dricbesh(n + 1, a[-1])
        B[-1, -1] = ricbesh(n + 1, a[-1])
        an[n] = np.linalg.det(A) / np.linalg.det(B)

        D = C + 0j
        D[-2, -1] = dricbesh(n + 1, a[-1])
        D[-1, -1] = ricbesh(n + 1, a[-1])
        bn[n] = np.linalg.det(C) / np.linalg.det(D)

    return an, bn


def mie_harmonics(x: np.ndarray, L: int) -> Tuple[np.ndarray]:
    """Calculates the spherical harmonics of the mie field.

    The harmonics are calculated up to order L using the iterative method.

    Parameters
    ----------
    x : ndarray
        The cosine of the angle defined by the line passing through origo parallel
        to the propagation direction and the evaluation point, with the corner at origo.
    L : int
        The order up to which to evaluate the harmonics. The L:th

    Returns
    -------
    ndarray, ndarray
        Tuple of ndarray of shape (L, *x.shape)

    """

    PI = np.zeros((L, *x.shape))
    TAU = np.zeros((L, *x.shape))

    PI[0, :] = 1
    PI[1, :] = 3 * x
    TAU[0, :] = x
    TAU[1, :] = 6 * x * x - 3

    for i in range(3, L + 1):
        PI[i - 1] = (2 * i - 1) / (i - 1) * x * PI[i - 2] - i / (i - 1) * PI[i - 3]
        TAU[i - 1] = i * x * PI[i - 1] - (i + 1) * PI[i - 2]

    return PI, TAU