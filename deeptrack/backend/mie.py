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

import numpy as np
from numpy.typing import NDArray

from .polynomials import (
    ricbesh,
    ricbesy,
    ricbesj,
    dricbesh,
    dricbesj,
    dricbesy,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _is_torch_array(x) -> bool:
    """Return whether x is, or contains, a torch tensor."""

    if not TORCH_AVAILABLE:
        return False

    if torch.is_tensor(x):
        return True

    if isinstance(x, (list, tuple)):
        return any(_is_torch_array(v) for v in x)

    return False


def _first_torch_array(*values):
    """Return the first torch tensor in values, searching nested sequences."""

    for value in values:
        if torch.is_tensor(value):
            return value

        if isinstance(value, (list, tuple)):
            found = _first_torch_array(*value)
            if found is not None:
                return found

    return None


def _torch_complex_dtype(*values):
    """Return a complex dtype compatible with the torch inputs."""

    for value in values:
        if torch.is_tensor(value):
            if value.dtype in (torch.float64, torch.complex128):
                return torch.complex128

        if isinstance(value, (list, tuple)):
            dtype = _torch_complex_dtype(*value)
            if dtype == torch.complex128:
                return dtype

    return torch.complex64


def _as_torch_scalar(value, dtype, device):
    """Convert value to a scalar torch tensor on device with dtype."""

    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device)

    return torch.as_tensor(value, dtype=dtype, device=device)


def _as_torch_vector(value, dtype, device):
    """Convert a tensor or sequence of scalars to a one-dimensional tensor."""

    if torch.is_tensor(value):
        return value.to(dtype=dtype, device=device).reshape(-1)

    return torch.stack(
        [
            _as_torch_scalar(element, dtype=dtype, device=device).reshape(())
            for element in value
        ]
    )


def _ricbesj_torch(l: int, x):
    """Differentiable torch Riccati-Bessel polynomial of the first kind."""

    if l == 0:
        return torch.sin(x)

    previous = torch.sin(x)
    current = torch.sin(x) / x - torch.cos(x)

    for order in range(1, l):
        previous, current = current, (2 * order + 1) / x * current - previous

    return current


def _dricbesj_torch(l: int, x):
    """Differentiable torch derivative of ricbesj."""

    return _ricbesj_torch(l - 1, x) - l / x * _ricbesj_torch(l, x)


def _ricbesy_torch(l: int, x):
    """Differentiable torch Riccati-Bessel polynomial of the second kind."""

    if l == 0:
        return torch.cos(x)

    previous = torch.cos(x)
    current = torch.cos(x) / x + torch.sin(x)

    for order in range(1, l):
        previous, current = current, (2 * order + 1) / x * current - previous

    return current


def _dricbesy_torch(l: int, x):
    """Differentiable torch derivative of ricbesy."""

    return _ricbesy_torch(l - 1, x) - l / x * _ricbesy_torch(l, x)


def _ricbesh_torch(l: int, x):
    """Differentiable torch Riccati-Bessel polynomial of the third kind."""

    return _ricbesj_torch(l, x) - 1j * _ricbesy_torch(l, x)


def _dricbesh_torch(l: int, x):
    """Differentiable torch derivative of ricbesh."""

    return _dricbesj_torch(l, x) - 1j * _dricbesy_torch(l, x)


def _coefficients_torch(
    m: float | complex | "torch.Tensor",
    a: float | "torch.Tensor",
    L: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch implementation of Mie coefficients."""

    reference = _first_torch_array(m, a)
    device = reference.device
    dtype = _torch_complex_dtype(m, a)

    m = _as_torch_scalar(m, dtype=dtype, device=device)
    a = _as_torch_scalar(a, dtype=dtype, device=device)

    if L == 0:
        empty = torch.empty((0,), dtype=dtype, device=device)
        return empty, empty.clone()

    A = []
    B = []

    for l in range(1, L + 1):
        Sx = _ricbesj_torch(l, a)
        dSx = _dricbesj_torch(l, a)
        Smx = _ricbesj_torch(l, m * a)
        dSmx = _dricbesj_torch(l, m * a)
        xix = _ricbesh_torch(l, a)
        dxix = _dricbesh_torch(l, a)

        A.append(
            (m * Smx * dSx - Sx * dSmx)
            / (m * Smx * dxix - xix * dSmx)
        )
        B.append(
            (Smx * dSx - m * Sx * dSmx)
            / (Smx * dxix - m * xix * dSmx)
        )

    return torch.stack(A), torch.stack(B)


def _stratified_coefficients_torch(
    m: list[complex] | "torch.Tensor",
    a: list[float] | "torch.Tensor",
    L: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch implementation of stratified Mie coefficients."""

    reference = _first_torch_array(m, a)
    device = reference.device
    dtype = _torch_complex_dtype(m, a)

    m = _as_torch_vector(m, dtype=dtype, device=device)
    a = _as_torch_vector(a, dtype=dtype, device=device)
    n_layers = a.numel()

    if n_layers == 1:
        return _coefficients_torch(m[0], a[0], L)

    if L == 0:
        empty = torch.empty((0,), dtype=dtype, device=device)
        return empty, empty.clone()

    an = []
    bn = []

    for n in range(L):
        A_rows = []
        C_rows = []
        zero = torch.zeros((), dtype=dtype, device=device)

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
                            A_ij = _dricbesj_torch(n + 1, m[p] * a[q])
                        elif j % 2 == 0:
                            A_ij = _dricbesy_torch(n + 1, m[p] * a[q])
                        else:
                            A_ij = _dricbesj_torch(n + 1, a[q])

                        if j != 2 * n_layers - 1:
                            C_ij = m[p] * A_ij
                        else:
                            C_ij = A_ij
                    else:
                        if (
                            j < 2 * n_layers - 1
                            and (j == 0 or j % 2 == 1)
                        ):
                            C_ij = _ricbesj_torch(n + 1, m[p] * a[q])
                        elif j % 2 == 0:
                            C_ij = _ricbesy_torch(n + 1, m[p] * a[q])
                        else:
                            C_ij = _ricbesj_torch(n + 1, a[q])

                        if j != 2 * n_layers - 1:
                            A_ij = m[p] * C_ij
                        else:
                            A_ij = C_ij

                A_rows.append(A_ij)
                C_rows.append(C_ij)

        A = torch.stack(A_rows).reshape(2 * n_layers, 2 * n_layers)
        C = torch.stack(C_rows).reshape(2 * n_layers, 2 * n_layers)

        B = A.clone()
        B[-2, -1] = _dricbesh_torch(n + 1, a[-1])
        B[-1, -1] = _ricbesh_torch(n + 1, a[-1])
        an.append(torch.linalg.det(A) / torch.linalg.det(B))

        D = C.clone()
        D[-2, -1] = _dricbesh_torch(n + 1, a[-1])
        D[-1, -1] = _ricbesh_torch(n + 1, a[-1])
        bn.append(torch.linalg.det(C) / torch.linalg.det(D))

    return torch.stack(an), torch.stack(bn)


def _harmonics_torch(
    x: "torch.Tensor",
    L: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch implementation of Mie harmonics."""

    PI = []
    TAU = []

    if L == 0:
        shape = (0, *x.shape)
        return (
            torch.empty(shape, dtype=x.dtype, device=x.device),
            torch.empty(shape, dtype=x.dtype, device=x.device),
        )

    if L >= 1:
        PI.append(torch.ones_like(x))
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

    return torch.stack(PI), torch.stack(TAU)


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

    if _is_torch_array(m) or _is_torch_array(a):
        return _coefficients_torch(m, a, L)

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
    if _is_torch_array(m) or _is_torch_array(a):
        return _stratified_coefficients_torch(m, a, L)

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

    if _is_torch_array(x):
        return _harmonics_torch(x, L)

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
