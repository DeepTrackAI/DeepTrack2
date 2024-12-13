"""Radial center calculation function

This module provides a function to calculate the center location
of an intensity distribution.

Module Structure
----------------
Functions:
- `radialcenter`: Calculates the center of a 2D intensity distribution.

Example
-------
Calculate center of an image containing randomly generated Gaussian blur.

>>> from deeptrack.extras import radialcenter as rc
>>> gaussian_blur = np.random.normal(0, 0.005, (100, 100))
>>> x, y = rc.radialcenter(gaussian_blur)
>>> print(f"Center of distribution = {x}, {y}")


  Python implementation by Benjamin Midtvedt, University of Gothenburg, 2020
  Copyright 2011-2012, Raghuveer Parthasarathy, The University of Oregon

  Disclaimer / License  
    This program is free software: you can redistribute it and/or 
      modify it under the terms of the GNU General Public License as 
      published by the Free Software Foundation, either version 3 of the 
      License, or (at your option) any later version.
    This set of programs is distributed in the hope that it will be useful, 
    but WITHOUT ANY WARRANTY; without even the implied warranty of 
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
    General Public License for more details.
    You should have received a copy of the GNU General Public License 
    (gpl.txt) along with this program. 
     If not, see <http://www.gnu.org/licenses/>.

  Raghuveer Parthasarathy
  The University of Oregon
  August 21, 2011 (begun)
  last modified Apr. 6, 2012 (minor change)
  Copyright 2011-2012, Raghuveer Parthasarathy
"""

# import * to keep syntax similar to matlab
from numpy import *
import scipy.signal

def radialcenter(I, invert_xy=False):
  """Calculates the center of a 2D intensity distribution.

  Considers lines passing through each half-pixel point with slope
  parallel to the gradient of the intensity at that point. Considers the
  distance of closest approach between these lines and the coordinate
  origin, and determines (analytically) the origin that minimizes the
  weighted sum of these distances-squared.

  Parameters
  ----------
  I : np.ndarray
    2D intensity distribution (i.e. a grayscale image)
    Size need not be an odd number of pixels along each dimension

  Returns
  -------
  float, float
      Coordinate pair x, y of the center of radial symmetry, 
      px, from px #1 = left/topmost pixel.
      So a shape centered in the middle of a 2*N+1 x 2*N+1
      square (e.g. from make2Dgaussian.m with x0=y0=0) will return
      a center value at x0=y0=N+1.

      Note that y increases with increasing row number (i.e. "downward")

  """
  I = squeeze(I)
  Ny, Nx = I.shape[:2]

  # Grid coordinates are -n:n, where Nx (or Ny) = 2*n+1.
  # Grid midpoint coordinates are -n+0.5:n-0.5.
  # The two lines below replace:
  #    xm = repmat(-(Nx-1)/2.0+0.5:(Nx-1)/2.0-0.5,Ny-1,1);
  # And are faster (by a factor of >15!).
  # The idea is taken from the repmat source code.
  xm_onerow = arange(-(Nx - 1) / 2.0 + 0.5, (Nx - 1) / 2.0 + 0.5)
  xm_onerow = reshape(xm_onerow, (1, xm_onerow.size))
  xm = xm_onerow[(0,) * (Ny - 1), :]

  # Similarly replacing:
  #    ym = repmat((-(Ny-1)/2.0+0.5:(Ny-1)/2.0-0.5)', 1, Nx-1).
  ym_onecol = arange(
      -(Ny - 1) / 2.0 + 0.5, (Ny - 1) / 2.0 + 0.5
  )  # Note that y increases "downward."
  ym_onecol = reshape(ym_onecol, (ym_onecol.size, 1))
  ym = ym_onecol[:, (0,) * (Nx - 1)]

  # Calculate derivatives along 45-degree shifted coordinates (u and v).
  # Note that y increases "downward" (increasing row number) -- we'll deal
  # with this when calculating "m" below.
  dIdu = I[: Ny - 1, 1:Nx] - I[1:Ny, : Nx - 1]
  dIdv = I[: Ny - 1, : Nx - 1] - I[1:Ny, 1:Nx]

  # Apply a smoothing filter.
  h = ones((3, 3)) / 9
  fdu = scipy.signal.convolve2d(dIdu, h, "same")
  fdv = scipy.signal.convolve2d(dIdv, h, "same")

  # Gradient magnitude, squared.
  dImag2 = fdu * fdu + fdv * fdv

  # Slope of the gradient.
  # Note that we need a 45-degree rotation of
  # the u,v components to express the slope in the x-y coordinate system.
  # The negative sign "flips" the array to account for y increasing
  # "downward."
  m = -(fdv + fdu) / (fdu - fdv)
  m[isnan(m)] = 0

  # Handle infinite slopes by setting them to a large value.
  isinfbool = isinf(m)
  m[isinfbool] = 1000000

  # Shorthand "b," which also happens to be the
  # y intercept of the line of slope m that goes through each grid midpoint.
  b = ym - m * xm

  # Weighting: Weight by square of gradient magnitude and inverse
  # distance to gradient intensity centroid.
  sdI2 = sum(dImag2)
  xcentroid = sum(dImag2 * xm) / sdI2
  ycentroid = sum(dImag2 * ym) / sdI2
  w = dImag2 / sqrt(
      (xm - xcentroid) * (xm - xcentroid) + (ym - ycentroid) * (ym - ycentroid)
  )

  # Least squares solution to determine the radial symmetry center.
  # Inputs m, b, w are defined on a grid.
  # w are the weights for each point.
  wm2p1 = w / (m * m + 1)
  sw = sum(wm2p1)
  mwm2pl = m * wm2p1
  smmw = sum(m * mwm2pl)
  smw = sum(mwm2pl)
  smbw = sum(sum(b * mwm2pl))
  sbw = sum(sum(b * wm2p1))
  det = smw * smw - smmw * sw
  xc = (smbw * sw - smw * sbw) / det
  # Relative to image center.
  yc = (smbw * smw - smmw * sbw) / det
  # Relative to image center.

  # Adjust coordinates relative to the image center.
  xc = xc + (Nx + 1) / 2.0 - 1
  yc = yc + (Ny + 1) / 2.0 - 1

  if invert_xy:
      return yc, xc
  else:
      return xc, yc
