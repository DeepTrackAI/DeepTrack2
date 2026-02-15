"""Radial center calculation.

This module provides a robust implementation of the radial symmetry center
estimator introduced by Parthasarathy (2011-2012).

The estimator computes local intensity gradients on a half-pixel grid and
solves a weighted least-squares problem to find the point that best matches
radial symmetry.

Key Features
------------
- **Gradient-Based Least-Squares Estimation**

    Uses intensity gradients evaluated on a half-pixel grid and solves a
    weighted least-squares system to estimate the center.

- **Numerical Safeguards**

    Handles common degeneracies (e.g., constant images, singular systems)
    by returning `nan` coordinates instead of raising obscure runtime errors.

- **Optional Coordinate Swapping**

    Can swap the returned `(x, y)` coordinate order for convenience.

Module Structure
----------------
Functions:

- `radialcenter(I, invert_xy) -> tuple[float, float]`

    Estimates the center of radial symmetry of a 2D intensity distribution.

Examples
--------
>>> from deeptrack.extras.radialcenter import radialcenter

Estimate the center of a 2D Gaussian:

>>> import numpy as np
>>>
>>> lin = np.linspace(-10, 10, 101)
>>> xg, yg = np.meshgrid(lin, lin, indexing="xy")
>>> img = np.exp(-0.5 * (xg**2 + yg**2))
>>>
>>> x, y = radialcenter(img)
>>> (round(x, 3), round(y, 3))
(50.0, 50.0)

References
----------
- Raghuveer Parthasarathy, University of Oregon (2011–2012).
- Python implementation by Benjamin Midtvedt, University of Gothenburg (2020).

License
-------
GNU General Public License v3 or later (GPL-3.0-or-later), per the original
distribution by Parthasarathy.

"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["radialcenter"]


def radialcenter(
    I: Any,
    invert_xy: bool = False,
) -> tuple[float, float]:
    """Calculate the center of a 2D intensity distribution.

    The method considers, for each half-pixel midpoint, a line passing through
    that point with slope parallel to the local intensity gradient. It then
    finds the point that minimizes a weighted sum of squared perpendicular
    distances to all such lines (weighted least squares).

    Parameters
    ----------
    I: Any
        2D intensity distribution (e.g. a grayscale image). The input is
        converted to a NumPy array. Extra singleton dimensions are removed.
    invert_xy: bool, optional
        If `True`, return `(y, x)` instead of `(x, y)`. Defaults to `False`.

    Returns
    -------
    tuple[float, float]
        The estimated center coordinate `(x, y)` in pixel units, where
        `(0, 0)` corresponds to the left/top-most pixel. The returned
        coordinates are floating-point and may fall between pixels.
        If the center cannot be estimated (e.g., constant image or singular
        system), returns `(nan, nan)` (or swapped if `invert_xy=True`).

    Notes
    -----
    This function requires SciPy for the 2D convolution used to smooth
    derivatives.

    """

    # Local import to avoid hard import-time dependency costs if unused.
    import scipy.signal  # pylint: disable=import-outside-toplevel

    arr = np.asarray(I)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            "radialcenter expects a 2D array after squeezing, got shape "
            f"{arr.shape}."
        )

    ny, nx = arr.shape
    if ny < 2 or nx < 2:
        raise ValueError(
            "radialcenter requires an array of shape at least (2, 2), got "
            f"{arr.shape}."
        )

    # Grid midpoint coordinates:
    # x: -(nx-1)/2+0.5 ... (nx-1)/2-0.5, repeated ny-1 times
    # y: -(ny-1)/2+0.5 ... (ny-1)/2-0.5, repeated nx-1 times
    xm_onerow = np.arange(
        -(nx - 1) / 2.0 + 0.5,
        (nx - 1) / 2.0 + 0.5,
        dtype=float,
    )[None, :]
    xm = np.repeat(xm_onerow, ny - 1, axis=0)

    ym_onecol = np.arange(
        -(ny - 1) / 2.0 + 0.5,
        (ny - 1) / 2.0 + 0.5,
        dtype=float,
    )[:, None]  # Note that y increases "downward."
    ym = np.repeat(ym_onecol, nx - 1, axis=1)
    
    # Derivatives along 45-degree shifted coordinates (u and v).
    dIdu = arr[: ny - 1, 1:nx] - arr[1:ny, : nx - 1]
    dIdv = arr[: ny - 1, : nx - 1] - arr[1:ny, 1:nx]

    # Smooth derivatives to reduce noise.
    kernel = np.ones((3, 3), dtype=float) / 9.0
    fdu = scipy.signal.convolve2d(dIdu, kernel, mode="same")
    fdv = scipy.signal.convolve2d(dIdv, kernel, mode="same")

    # Gradient magnitude squared.
    dImag2 = fdu * fdu + fdv * fdv

    sdI2 = float(np.sum(dImag2))
    if not np.isfinite(sdI2) or sdI2 <= 0.0:
        out = (float("nan"), float("nan"))
        return out[::-1] if invert_xy else out

    # Slope in x-y coordinates (accounting for y increasing downward).
    with np.errstate(divide="ignore", invalid="ignore"):
        m = -(fdv + fdu) / (fdu - fdv)

    # Replace NaNs and infs robustly.
    m = np.where(np.isfinite(m), m, 0.0)
    m = np.where(np.isinf(m), 1e6, m)

    # Line intercepts for lines passing through each midpoint: y = m x + b.
    b = ym - m * xm

    # Centroid of gradient energy.
    xcentroid = float(np.sum(dImag2 * xm) / sdI2)
    ycentroid = float(np.sum(dImag2 * ym) / sdI2)

    # Weighting: gradient magnitude squared divided by distance to centroid.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt((xm - xcentroid) ** 2 + (ym - ycentroid) ** 2)
        w = dImag2 / r

    # Avoid infinities when r == 0 at the centroid.
    w = np.where(np.isfinite(w), w, 0.0)

    # Weighted least squares.
    wm2p1 = w / (m * m + 1.0)
    sw = float(np.sum(wm2p1))
    mwm2p1 = m * wm2p1
    smmw = float(np.sum(m * mwm2p1))
    smw = float(np.sum(mwm2p1))

    # b*weights sums (note: b, m, w are 2D arrays).
    smbw = float(np.sum(b * mwm2p1))
    sbw = float(np.sum(b * wm2p1))

    det = smw * smw - smmw * sw
    if not np.isfinite(det) or det == 0.0:
        out = (float("nan"), float("nan"))
        return out[::-1] if invert_xy else out

    # Center relative to image center.
    xc_rel = (smbw * sw - smw * sbw) / det
    yc_rel = (smbw * smw - smmw * sbw) / det

    # Convert to pixel coordinates with (0, 0) at top-left.
    xc = float(xc_rel + (nx + 1) / 2.0 - 1.0)
    yc = float(yc_rel + (ny + 1) / 2.0 - 1.0)

    return (yc, xc) if invert_xy else (xc, yc)
