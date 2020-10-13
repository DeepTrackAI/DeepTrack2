import numpy as np

from typing import Tuple
from deeptrack.backend import ricbesh, ricbesj, besselj, dricbesh, dricbesj, dbesselj


def mie_coefficients(
    k: float, m: float or complex, a: float, L: int
) -> Tuple[np.ndarray]:
    """Calculate the coefficients of the spherical harmonics.

    Calculates the terms up to (and including) order L using riccati-bessel polynomials.

    Parameters
    ----------
    k : float
        The wave-vector in the medium
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

        Sx = ricbesj(l, k * a)
        dSx = dricbesj(l, k * a)
        Smx = ricbesj(l, k * m * a)
        dSmx = dricbesj(l, k * m * a)
        xix = ricbesh(l, k * a)
        dxix = dricbesh(l, k * a)

        A[l - 1] = (m * Smx * dSx - Sx * dSmx) / (m * Smx * dxix - xix * dSmx)
        B[l - 1] = (Smx * dSx - m * Sx * dSmx) / (Smx * dxix - m * xix * dSmx)

    return A, B


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