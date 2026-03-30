"""Classes that implement light-scattering objects.

This module provides implementations of scattering objects
with geometries that are commonly observed in experimental setups 
such as ellipsoids, spheres, or point-particles.

These scatterer objects are primarily used in combination with the `Optics`
module to simulate how a (e.g. brightfield) microscope would resolve the 
object for a given optical setup (NA, wavelength, Refractive Index etc.).

Key Features
------------
- **Customizable geometries**

    The initialization parameters allow the user to choose proportions and 
    positioning of the scatterer in the image. It is also possible to combine 
    multiple scatterers and overlay them, e.g. two ellipses orthogonal
    to each other would form a plus-shape or combining two spheres
    (one small, one large) to simulate a core-shell particle.
    
- **Defocusing**

    As the `z` parameter represents the scatterers position in relation to the
    focal point of the microscope, the user can simulate defocusing by setting
    this parameter to be non-zero.
    
- **Mie scatterers**

    Implements Mie-theory scatterers that calculates harmonics up to a desired
    order with functions and utilities from `deeptrack.backend.mie`. Includes
    the case of a spherical Mie scatterer, and a stratified spherical
    scatterer which is a sphere with several concentric shells of
    uniform refractive index.
    
Module Structure
----------------
Classes:

- `Scatterer`: Abstract base class for scatterers.

    This abstract class stores positional information about the scatterer
    and implements a method to convert the position to voxel units,
    as well as the a methods to upsample and crop.

- `PointParticle`: Generates point particles with the size of 1 pixel.

    Represented as a numpy array or a torch tensor of ones.

- `Ellipse`: Generates 2-D elliptical particles.

- `Sphere`: Generates 3-D spheres.

- `Ellipsoid`: Generates 3-D ellipsoids.

- `MieScatterer`: Mie scatterer base class.

- `MieSphere`: Extends `MieScatterer` to the spherical case.

- `MieStratifiedSphere`:  Extends `MieScatterer` to the stratified sphere case.

    A stratified sphere is a sphere with several concentric shells of uniform
    refractive index.

Examples
--------

Create a ellipse scatterer and resolve it through a microscope:

>>> import numpy as np

>>> import deeptrack as dt

>>> optics = dt.Fluorescence(
...            NA=0.7,
...            wavelength=680e-9,
...            resolution=1e-6,
...            magnification=10,
...            output_region=(0, 0, 64, 64),
...        )

>>> scatterer = dt.Ellipse(
...      intensity=100,
...      position_unit="pixel",
...      position=(32, 32),
...      radius=(1e-6, 0.5e-6),
...      rotation=np.pi / 4,
...      upsample=4,
...  )

>>> imaged_scatterer = optics(scatterer)
>>> imaged_scatterer.plot(cmap="gray")

Combine multiple scatterers to image a core-shell particle:

>>> import numpy as np

>>> import deeptrack as dt

>>> optics = dt.Fluorescence(
...    NA=1.4,
...    wavelength=638.0e-9,
...    refractive_index_medium=1.33,
...    output_region=[0, 0, 64, 64],
...    magnification=1,
...    resolution=100e-9,
...    return_field=False,
... )

>>> inner_sphere = dt.Ellipsoid(
...    position=(32, 32),
...    z=-500e-9, # Defocus slightly.
...    radius=450e-9,
...    intensity=100,
... )

>>> outer_sphere = dt.Ellipsoid(
...    position=inner_sphere.position,
...    z=inner_sphere.z,
...    radius=inner_sphere.radius * 2,
...    intensity= inner_sphere.intensity * -0.25,
... )

>>> combined_scatterer = inner_sphere >> outer_sphere
>>> imaged_scatterer = optics(combined_scatterer)
>>> imaged_scatterer.plot(cmap="gray")

Create a stratified Mie sphere and resolve it through a microscope:

>>> import numpy as np

>>> import deeptrack as dt

>>> optics = dt.Brightfield(
...    NA=0.7,
...    wavelength=680e-9,
...    resolution=1e-6,
...    magnification=5,
...    output_region=(0, 0, 64, 64),
...    return_field=True,
...    upscale=4,
... )

>>> scatterer = dt.MieStratifiedSphere(
...    radius=np.array([0.5e-6, 3e-6]),
...    refractive_index=[1.45 + 0.1j, 1.52],
...    position_unit="pixel",
...    position=(128, 128),
...    aperature_angle=0.1,
... )

>>> imaged_scatterer = optics(scatterer) # Creates an array of complex numbers.
>>> abs_imaged_scatterer = dt.Abs(imaged_scatterer)

>>> abs_imaged_scatterer.plot()

"""

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT321

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import warnings

import array_api_compat as apc
import numpy as np
from numpy.typing import NDArray
from pint import Quantity
from dataclasses import dataclass

from deeptrack.holography import get_propagation_matrix
from deeptrack.backend.units import (
    ConversionTable,
    get_active_scale,
    get_active_voxel_size,
)
from deeptrack.backend import mie
from deeptrack.math import AveragePooling
from deeptrack.features import Feature, StructuralFeature, MERGE_STRATEGY_APPEND
from deeptrack.wrappers import Wrapper
from deeptrack.image import pad_image_to_fft #TODO ***??***  pad_image_to_fft should be moved
from deeptrack import units_registry as u

from deeptrack.backend import xp

__all__ = [
    "Scatterer",
    "PointParticle",
    "Ellipse",
    "Sphere",
    "Ellipsoid",
    "MieScatterer",
    "MieSphere",
    "MieStratifiedSphere",
    "Unpolarized",
    "Incoherent",
]


if TYPE_CHECKING:
    import torch


#TODO ***??*** revise Scatterer - torch, typing, docstring, unit test
class Scatterer(Feature):
    """Base abstract class for scatterers.

    A scatterer is defined by a 3-dimensional volume of voxels.
    To each voxel corresponds an occupancy factor, i.e., how much
    of that voxel does the scatterer occupy. However, this number is not
    necessarily limited to the [0, 1] range. It can be any number, and its
    interpretation is left to the optical device that images the scatterer.

    This abstract class implements the `_process_properties` method to convert
    the position to voxel units, as well as the `_process_and_get` method to
    upsample the calculation and crop empty slices.

    Attributes
    ----------
    position:   tuple[float, float] | tuple[float, float, float]
        The position of the particle, length 2 or 3. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    value: float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
        
    position_unit: "meter" or "pixel"
        The unit of the provided position property.

    upsample_axes: tuple of ints
        Sets the axes along which the calculation is upsampled (default is
        None, which implies all axes are upsampled).
        
    crop_zeros: bool
        Whether to remove slices in which all elements are zero.
        
    """

    __list_merge_strategy__ = MERGE_STRATEGY_APPEND ### Not clear why needed
    __distributed__ = False
    __conversion_table__ = ConversionTable(
        position=(u.pixel, u.pixel),
        z=(u.zpixel, u.zpixel),
        voxel_size=(u.meter, u.meter),
    )

    def __init__(
        self,
        position: tuple[float, float] | tuple[float, float, float] = (32.0, 32.0),
        z: float = 0.0,
        value: float = 1.0,
        position_unit: str = "pixel",
        upsample: int = 1,
        voxel_size=None,
        pixel_size=None,
        **kwargs,
    ) -> None:
        # Ignore warning to help with comparison with arrays.
        # if upsample != 1:  # noqa: F632
        #     warnings.warn(
        #         f"Setting upsample != 1 is deprecated. "
        #         f"Please, instead use dt.Upscale(f, factor={upsample})"
        #     )

        self._processed_properties = False

        super().__init__(
            position=position,
            z=z,
            value=value,
            position_unit=position_unit,
            upsample=upsample,
            voxel_size=voxel_size,
            pixel_size=pixel_size,
            _position_sampler=lambda: position,
            **kwargs,
        )

    def _antialias_volume(self, volume, factor: int):
        """Geometry-only supersampling anti-aliasing.

        Assumes `volume` was generated on a grid oversampled by `factor`
        and downsamples it back by average pooling.
        """
        if factor == 1:
            return volume

        # average pooling conserves fractional occupancy
        return AveragePooling(
            factor
        )(volume)


    def _process_properties(
        self,
        properties: dict
    ) -> dict:

        # Rescales the position property.
        properties = super()._process_properties(properties)
        self._processed_properties = True
        return properties
        
    def _process_and_get(
        self,
        *args,
        voxel_size: np.ndarray,
        upsample: int,
        upsample_axes=None,
        crop_empty=True,
        **kwargs
    ) -> list[np.ndarray | torch.Tensor]:


        # Post processes the created object to handle upsampling,
        # as well as cropping empty slices.
        if not self._processed_properties:

            warnings.warn(
                "Overridden _process_properties method does not call super. "
                + "This is likely to result in errors if used with "
                + "Optics.upscale != 1."
            )


        voxel_size = xp.asarray(get_active_voxel_size(), dtype=float)

        apply_supersampling = upsample > 1 and isinstance(self, VolumeScatterer)

        if upsample > 1 and not apply_supersampling:
            warnings.warn(
                "Geometry supersampling (upsample) is ignored for "
                "FieldScatterers.",
                UserWarning,
            )

        if apply_supersampling:
            voxel_size /= float(upsample)

        new_image = super(Scatterer, self)._process_and_get(
            *args,
            voxel_size=voxel_size,
            upsample=upsample,
            **kwargs,
        )[0]

        if apply_supersampling:
            new_image = self._antialias_volume(new_image, factor=upsample)


        # if new_image.size == 0:
        if new_image.numel() == 0 if apc.is_torch_array(new_image) else new_image.size == 0:
            warnings.warn(
                "Scatterer created that is smaller than a pixel. "
                + "This may yield inconsistent results."
                + " Consider using upsample on the scatterer,"
                + " or upscale on the optics.",
                Warning,
            )

        # Crops empty slices
        if crop_empty:
            # new_image = new_image[~np.all(new_image == 0, axis=(1, 2))]
            # new_image = new_image[:, ~np.all(new_image == 0, axis=(0, 2))]
            # new_image = new_image[:, :, ~np.all(new_image == 0, axis=(0, 1))]
            mask_z = ~xp.all(new_image == 0, axis=(1, 2))
            mask_y = ~xp.all(new_image == 0, axis=(0, 2))
            mask_x = ~xp.all(new_image == 0, axis=(0, 1))

            new_image = new_image[mask_z][:, mask_y][:, :, mask_x]

        # # Copy properties
        # props = kwargs.copy()
        return [self._wrap_output(new_image, kwargs)]

    def _wrap_output(self, array, props):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _wrap_output()"
        )


class VolumeScatterer(Scatterer):
    """Abstract scatterer producing ScatteredVolume outputs."""
    def _wrap_output(self, array, props) -> ScatteredVolume:
        return ScatteredVolume(
            array=array,
            properties=props.copy(),
        )


class FieldScatterer(Scatterer):
    def _wrap_output(self, array, props) -> ScatteredField:
        return ScatteredField(
            array=array,
            properties=props.copy(),
        )


#TODO ***??*** revise PointParticle - torch, typing, docstring, unit test
class PointParticle(VolumeScatterer):
    """Generate a diffraction-limited point particle.

    A point particle is approximated by the size of a single pixel or voxel.
    For subpixel positioning, the position is interpolated linearly.

    Parameters
    ----------
    position:  tuple[float, float] | tuple[float, float, float] = (32.0, 32.0)
        Particle position in 2D or 3D. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    value: float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
        
    """

    def __init__(
        self: PointParticle,
        **kwargs: Any,
    ):
        """

        """
        kwargs.pop("upsample", None)
        super().__init__(upsample=1, upsample_axes=(), **kwargs)

    def get(
        self: PointParticle,
        *ignore,
        **kwarg: Any,
    ) -> np.ndarray | torch.Tensor:
        """Evaluate and return the scatterer volume."""

        scale = xp.asarray(get_active_scale(), dtype=xp.float32)

        return xp.ones((1, 1, 1), dtype=scale.dtype) * xp.prod(scale)


#TODO ***??*** revise Ellipse - torch, typing, docstring, unit test
class Ellipse(VolumeScatterer):
    """Generates an elliptical disk scatterer

    Parameters
    ----------
    radius: float | tuple[float, float]
        Radius of the ellipse in meters. If only one value,
        assume circular.
        
    rotation: float
        Orientation angle of the ellipse in the camera plane in radians.
        
    position: tuple[float, float] | tuple[float, float, float]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    value: float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
        
    upsample: int
        Upsamples the calculations of the pixel occupancy fraction.
        
    transpose: bool
        If True, the ellipse is transposed as to align the first axis of the
        radius with the first axis of the created volume. This is applied
        before rotation.

    """


    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        radius: float = 1e-6,
        rotation: float = 0,
        transpose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            radius=radius, rotation=rotation, transpose=transpose, **kwargs
        )

    def _process_properties(
        self,
        properties: dict
    ) -> dict:
        """Preprocess the input to the method .get()

        Ensures that the radius is an array of length 2. If the radius
        is a single value, the particle is made circular
        """

        properties = super()._process_properties(properties)

        # Ensure radius is of length 2
        radius = properties["radius"]
        r = xp.asarray(radius) if hasattr(xp, "asarray") else xp.array(radius)

        if r.ndim == 0:
            r = xp.stack([r, r])
        else:
            n = r.shape[0]
            if n == 1:
                # If only one value, assume circle.
                # radius = (radius[0], radius[0])
                r = xp.stack([r.reshape(()), r.reshape(())])
            else:
                r = r[:2]
        
        properties["radius"] = r

        return properties

    def get(
        self,
        *ignore,
        radius: np.ndarray | torch.Tensor | float,
        rotation: float,
        voxel_size: np.ndarray | torch.Tensor,
        transpose: bool,
        **kwargs
    ) -> np.ndarray | torch.Tensor:
        """Abstract method to initialize the ellipse scatterer"""
        rotation = xp.asarray(rotation)
        if not transpose:
            radius = xp.stack([radius[1], radius[0]])

        # Create a grid to calculate on.
        rad = radius[:2]
        ceil = int(xp.ceil(xp.max(rad) / xp.min(voxel_size[:2])))
        rad_ceil = int(
            xp.ceil(xp.max(radius) / xp.min(voxel_size)).item()
        )
        Y, X = xp.meshgrid(
            xp.arange(-rad_ceil, rad_ceil) * voxel_size[1],
            xp.arange(-rad_ceil, rad_ceil) * voxel_size[0],
        )

        cos = xp.cos(-rotation)
        sin = xp.sin(-rotation)
        Xt = X * cos + Y * sin
        Yt = -X * sin + Y * cos

        # Evaluate ellipse.
        mask = xp.asarray(
            (Xt * Xt) / (rad[0] * rad[0]) +
            (Yt * Yt) / (rad[1] * rad[1]) < 1,
            dtype=xp.float32,
        )
        mask = xp.expand_dims(mask, axis=-1)
        return mask



#TODO ***??*** revise Sphere - torch, typing, docstring, unit test
class Sphere(VolumeScatterer):
    """Generates a spherical scatterer

    Parameters
    ----------
    radius: float
        Radius of the sphere in meters.
        
        position: tuple[float, float] | tuple[float, float, float]
        The position of the particle, length 2 or 3. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    value: float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
        
    upsample: int
        Upsamples the calculations of the pixel occupancy fraction.
        
    """

    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
    )

    def __init__(
        self,
        radius: float = 1e-6,
        **kwargs
    ) -> None:
        super().__init__(radius=radius, **kwargs)

    def get(
        self,
        image: np.ndarray | torch.Tensor,
        radius: float,
        voxel_size: np.ndarray | torch.Tensor,
        **kwargs
    ) -> np.ndarray | torch.Tensor:
        """Abstract method to initialize the sphere scatterer"""

        # Create a grid to calculate on.
        rad = xp.asarray(radius) * xp.ones(3) / xp.asarray(voxel_size)
        rad_ceil = xp.ceil(rad)
        if hasattr(rad_ceil, "astype"):
            rad_ceil = rad_ceil.astype(int)
        else:
            rad_ceil = rad_ceil.to(dtype=xp.int64)
            
        x = xp.arange(-rad_ceil[0], rad_ceil[0])
        y = xp.arange(-rad_ceil[1], rad_ceil[1])
        z = xp.arange(-rad_ceil[2], rad_ceil[2])
        
        X, Y, Z = xp.meshgrid(
            (y / rad[1]) ** 2,
            (x / rad[0]) ** 2,
            (z / rad[2]) ** 2,
            indexing="xy",   # important for torch consistency
        )

        mask = xp.asarray(
            X + Y + Z <= 1,
            dtype=xp.float32,
        )
        return mask


#TODO ***??*** revise Ellipsoid - torch, typing, docstring, unit test
class Ellipsoid(VolumeScatterer):
    """Generates an ellipsoidal scatterer

    Parameters
    ----------
        radius: float | tuple[float, float, float]
        Radius of the ellipsoid in meters. If only one value,
        assume spherical.
        
    rotation: float
        Rotation of the ellipsoid in about the x, y and z axis.
        
    position: tuple[float, float] | tuple[float, float, float]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    value: float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
        
    upsample: int
        Upsamples the calculations of the pixel occupancy fraction.
        
    transpose: bool
        If True, the ellipse is transposed as to align the first axis
        of the radius with the first axis of the created volume.
        This is applied before rotation.
        
    """

    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        radius: float = 1e-6,
        rotation: float = 0,
        transpose: float = False,
        **kwargs,
    ) -> None:
        super().__init__(
            radius=radius, rotation=rotation, transpose=transpose, **kwargs
        )

    def _process_properties(
        self,
        propertydict: dict
    ) -> dict:
        """Preprocess the input to the method .get()

        Ensures that the radius and the rotation properties both are arrays of
        length 3.

        If the radius is a single value, the particle is made a sphere
        If the radius are two values, the smallest value is appended as the
        third value

        The rotation vector is padded with zeros until it is of length 3
        """

        propertydict = super()._process_properties(propertydict)

        # Ensure radius has three values.
        r = xp.asarray(propertydict["radius"])
        if r.ndim == 0:
            r = xp.stack([r])

        n = r.shape[0]
        if n == 1:
            # If only one value, assume sphere.
            # radius = (*radius,) * 3
            r = xp.stack([r.reshape(()), r.reshape(()), r.reshape(())])
        elif n == 2:
            # If two values, duplicate the minor axis.
            # radius = (*radius, np.min(radius[-1]))
            r = xp.stack([r[0], r[1], xp.minimum(r[0], r[1])])
        elif n == 3:
            # If three values, convert to tuple for consistency.
            # radius = (*radius,)
            r = r[:3]
        propertydict["radius"] = r

        # Ensure rotation has three values.
        rot = xp.asarray(propertydict["rotation"])
        if rot.ndim == 0:
            # rot = xp.array([rot])
            rot = xp.stack([rot])

        n = rot.shape[0]
        if n == 1:
            # If only one value, pad with two zeros.
            # rotation = (*rotation, 0, 0)
            rot = xp.stack([rot.reshape(()), xp.asarray(0.0), xp.asarray(0.0)])
        elif n == 2:
            # If two values, pad with one zero.
            # rotation = (*rotation, 0)
            rot = xp.stack([rot[0], rot[1], xp.asarray(0.0)])
        elif n == 3:        
            # If three values, convert to tuple for consistency.
            # rotation = (*rotation,)
            rot = rot[:3]
        propertydict["rotation"] = rot

        return propertydict

    def get(
        self,
        image: np.ndarray | torch.Tensor,
        radius: np.ndarray | torch.Tensor | float,
        rotation: np.ndarray | torch.Tensor | float,
        voxel_size: np.ndarray | torch.Tensor | float,
        transpose: bool,
        **kwargs
    ) -> np.ndarray | torch.Tensor:
        """Abstract method to initialize the ellipsoid scatterer"""

        radius = xp.asarray(radius)
        rotation = xp.asarray(rotation)
        voxel_size = xp.asarray(voxel_size)

        if not transpose:
            # Swap the first and second value of the radius vector.
            radius = xp.stack([radius[1], radius[0], radius[2]])


        rad_ceil = int(
            xp.ceil(xp.max(radius) / xp.min(voxel_size)).item()
        )

        # Create grid to calculate on.
        x = xp.arange(-rad_ceil, rad_ceil) * voxel_size[0]
        y = xp.arange(-rad_ceil, rad_ceil) * voxel_size[1]
        z = xp.arange(-rad_ceil, rad_ceil) * voxel_size[2]
        Y, X, Z = xp.meshgrid(y, x, z)

        # Rotate the grid.
        cos = xp.cos(rotation)
        sin = xp.sin(rotation)
        XR = (
            (cos[0] * cos[1] * X)
            + (cos[0] * sin[1] * sin[2] - sin[0] * cos[2]) * Y
            + (cos[0] * sin[1] * cos[2] + sin[0] * sin[2]) * Z
        )
        YR = (
            (sin[0] * cos[1] * X)
            + (sin[0] * sin[1] * sin[2] + cos[0] * cos[2]) * Y
            + (sin[0] * sin[1] * cos[2] - cos[0] * sin[2]) * Z
        )
        ZR = (-sin[1] * X) + cos[1] * sin[2] * Y + cos[1] * cos[2] * Z

        mask = xp.asarray(
            (XR / radius[0]) ** 2 +
            (YR / radius[1]) ** 2 +
            (ZR / radius[2]) ** 2 < 1,
            dtype=xp.float32,
        )
        return mask


#TODO ***??*** revise MieScatterer - torch, typing, docstring, unit test
class MieScatterer(FieldScatterer):
    """Base class for Mie-theory scatterers.

    This class implements scattering from spherical particles using Mie
    theory. New scatterer types can be created by subclassing `MieScatterer`
    and providing a function that returns the Mie coefficients.
    The coefficient function should return the harmonic coefficients up to
    order `L`. Specifically, it should be a wrapper that receives the current
    feature properties and returns a callable. That callable must take a
    single integer argument `L` and return the coefficients `(an, bn)` up to
    that order.
    See `deeptrack.backend.mie.coefficients` for an example implementation.

    Attributes
    ----------
    coefficients : Callable[int] -> tuple[ndarray, ndarray]
        Function that returns the harmonic coefficients.

    offset_z : "auto" | float
        Distance from the particle in the z direction where the field is
        evaluated. If `"auto"`, this is calculated from the pixel size and
        `collection_angle`.

    collection_angle : "auto" | float
        Maximum collection angle in radians. If `"auto"`, this is computed
        from the objective NA (assuming the objective is the limiting
        aperture).

    input_polarization : float | Quantity | str
        Polarization angle of the incident illumination in radians. If a float
        (or `Quantity`), it specifies the orientation of a linear polarizer
        before the sample. If set to `"circular"`, circular polarization is 
        approximated by assigning equal weights to the two orthogonal 
        scattering components.

    output_polarization : float | Quantity
        Angle of a polarization analyzer placed after the sample, in radians.
        If a float (or `Quantity`), the detected field is projected onto the 
        corresponding linear polarization direction.

    L : int | str
        Number of terms used to evaluate the Mie series. If `"auto"`,
        the number of terms is determined automatically.

    position : tuple[float, float] | tuple[float, float, float]
        Particle position. If three values are provided, the third
        corresponds to the axial position relative to the camera plane.

    z : float
        Axial particle position if `position` is two-dimensional.

    return_fft : bool
        If True, the feature returns the Fourier transform of the field
        rather than the spatial field itself.

    coherence_length : float | None
        Temporal coherence length of the illumination in meters. If None,
        illumination is assumed to be fully coherent.

    amp_factor : float
        Scaling factor applied to the scattered field amplitude.

    phase_shift_correction : bool
        If True, applies a phase correction to the field according to
        arr *= exp(1j * k * z + 1j * π / 2)
        This correction is used in ISCAT simulations.

    mode : {"geometric", "hybrid"}
        Determines how the scattered field is constructed before propagation.

        Both modes use the same Mie coefficients but differ in how the
        scattered field is represented prior to propagation through the
        optical system.
        - "geometric"
          Evaluates the scattered field as a spherical wave on a virtual
          plane located at ``offset_z`` from the particle. The field includes
          the geometric propagation factor ``exp(i k R) / R`` and is sampled
          on a finite spatial grid before being propagated through the optical
          system. Because the field is computed on a finite plane, the result
          can be sensitive to the simulated field-of-view.
        - "hybrid"
          Constructs the scattered field using the Mie scattering amplitudes
          S1`` and ``S2`` mapped to spatial frequencies corresponding to the
          objective pupil. The field is then propagated to the detector.
          This approach is less sensitive to the simulated field-of-view and
          generally more numerically stable.

    pupil : None | ndarray
        Optional pupil function applied to the scattered field. This can be
        used to simulate aberrations or other modifications of the optical
        system.
        
    """


    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
        polarization_angle=(u.radian, u.radian),
        collection_angle=(u.radian, u.radian),
        wavelength=(u.meter, u.meter),
        offset_z=(u.meter, u.meter),
        coherence_length=(u.meter, u.meter),
    )

    def __init__(
        self,
        coefficients,
        input_polarization: float=0,
        output_polarization: float=0,
        offset_z: str="auto",
        collection_angle: str = "auto",
        L: str = "auto",
        refractive_index_medium: float=None,
        wavelength: float=None,
        NA: float=None,
        padding=(0,) * 4,
        output_region=None,
        polarization_angle: float | None=None,
        working_distance: float=1000000,
        position_objective: tuple[float, float]=(0, 0),
        return_fft: bool=False,
        coherence_length: float=None,
        illumination_angle: float=0,
        amp_factor: float=1,
        phase_shift_correction: bool=False,
        mode: str = "geometric",
        pupil=None,
        **kwargs,
    ) -> None:
        self.mode = mode
        self.pupil = pupil
        if polarization_angle is not None:
            warnings.warn(
                "polarization_angle is deprecated. " 
                "Please use input_polarization instead"
            )
            input_polarization = polarization_angle
        kwargs.pop("crop_empty", None)

        super().__init__(
            is_field=True, # remove
            crop_empty=False,
            L=L,
            offset_z=offset_z,
            input_polarization=input_polarization,
            output_polarization=output_polarization,
            collection_angle=collection_angle,
            coefficients=coefficients,
            refractive_index_medium=refractive_index_medium,
            wavelength=wavelength,
            NA=NA,
            padding=padding,
            output_region=output_region,
            polarization_angle=polarization_angle,
            working_distance=working_distance,
            position_objective=position_objective,
            return_fft=return_fft,
            coherence_length=coherence_length,
            illumination_angle=illumination_angle,
            amp_factor=amp_factor,
            phase_shift_correction=phase_shift_correction,
            mode=mode,
            pupil=pupil,
            **kwargs,
        )

    def _process_properties(
        self,
        properties: dict
    ) -> dict:

        properties = super()._process_properties(properties)

        # --- polarization validation ---
        inp = properties.get("input_polarization", None)
        out = properties.get("output_polarization", None)

        if inp is None:
            raise ValueError(
                "input_polarization must be specified for coherent scattering. "
                "Use the Incoherent feature to model unpolarized illumination."
            )

        if out is None:
            raise ValueError(
                "output_polarization=None (no analyzer) is not supported in coherent mode. "
                "Use the Incoherent feature to model detection without analyzer."
            )

        if properties["L"] == "auto":
            try:
                v = (
                    2 * np.pi *
                    np.max(properties["radius"]) / properties["wavelength"]
                )

                properties["L"] = int(np.floor((v + 4 * (v ** (1 / 3)) + 1)))
            except (ValueError, TypeError):
                pass
        if properties["collection_angle"] == "auto":
            properties["collection_angle"] = np.arcsin(
                properties["NA"] / properties["refractive_index_medium"]
            )

        if properties["offset_z"] == "auto":
            size = (
                np.array(properties["output_region"][2:])
                - properties["output_region"][:2]
            )
            xSize, ySize = size

            # offset_z should be calculated with the physical size of the image
            # not the fft-padded size
            min_edge_size=np.min([xSize,ySize])
            properties["offset_z"] = (
                min_edge_size
                * 0.45
                * min(get_active_voxel_size()[:2])
                / np.tan(properties["collection_angle"])
            )
        return properties

    def get_xy_size(
        self,
        output_region: tuple[int, int, int, int],
        padding: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        """Computes the x and y dimensions of the output region with padding.

        Parameters
        ----------
        output_region: tuple[int, int, int, int]
            The coordinates defining the output region.

        padding: tuple[int, int, int, int]
            The padding applied in each direction.

        Returns
        -------
        tuple[int, int]
            The total size in x and y directions.

        """
        return (
            output_region[2] - output_region[0] + padding[0] + padding[2],
            output_region[3] - output_region[1] + padding[1] + padding[3],
        )


    def get_XY(
        self,
        shape: tuple[int, int],
        voxel_size: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates meshgrid for X and Y given the shape and voxel size.

        Parameters
        ----------
        shape: tuple[int, int]
            The dimensions of the output region.

        voxel_size: tuple[float, float]
            The size of each voxel in meters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The meshgrid of X and Y coordinates.

        """
        x = np.arange(shape[0]) - shape[0] / 2 
        y = np.arange(shape[1]) - shape[1] / 2 
        return np.meshgrid(x * voxel_size[0], y * voxel_size[1], indexing="ij")
    

    def get_detector_mask(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Creates a mask based on a circular aperture.

        Parameters
        ----------
        X: np.ndarray
            X-coordinates of the field.

        Y: np.ndarray
            Y-coordinates of the field.

        radius: float
            The radius of the detector aperture.

        Returns
        -------
        np.ndarray
            A boolean mask.

        """

        return np.sqrt(X ** 2 + Y ** 2) < radius

    def _plane_in_polar_coords_geometric(
        self,
        shape: tuple[int, int],
        voxel_size: np.ndarray,
        plane_position: np.ndarray,
        illumination_angle: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, Y = self.get_XY(shape, voxel_size)

        X = X + plane_position[0]
        Y = Y + plane_position[1]
        Z = plane_position[2]

        R2_squared = X**2 + Y**2
        R3 = np.sqrt(R2_squared + Z**2)

        cos_theta = Z / R3
        illumination_cos_theta = np.cos(np.arccos(cos_theta) + illumination_angle)
        phi = np.arctan2(Y, X)

        return R3, cos_theta, illumination_cos_theta, phi
    
    def _plane_in_polar_coords_hybrid(
        self,
        shape: tuple[int, int],
        voxel_size: np.ndarray,
        plane_position: np.ndarray,
        illumination_angle: float,
        k: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, Y = self.get_XY(shape, voxel_size)

        X = X + plane_position[0]
        Y = Y + plane_position[1]
        Z = plane_position[2]

        R2_squared = X**2 + Y**2
        R3 = np.sqrt(R2_squared + Z**2)

        Q = np.sqrt(R2_squared)/voxel_size[0]**2*2*np.pi/shape[0]
        sin_theta=Q/(k)
        pupil_mask=sin_theta<1
        cos_theta=np.zeros(sin_theta.shape)
        cos_theta[pupil_mask]=np.sqrt(1-sin_theta[pupil_mask]**2)

        illumination_cos_theta = (
            np.cos(np.arccos(cos_theta) + illumination_angle)
            )
        phi = np.arctan2(Y, X)

        return R3, cos_theta, illumination_cos_theta, phi, pupil_mask

    def _polarization_coefficients(
        self: MieScatterer,
        phi: np.ndarray,
        illumination_cos_theta: np.ndarray,
        input_polarization,
        output_polarization,
    ):

        if isinstance(input_polarization, (float, int, str, Quantity)):
            if isinstance(input_polarization, Quantity):
                input_polarization = input_polarization.to("rad").magnitude

            if isinstance(input_polarization, (float, int)):
                S1_coef = np.sin(phi + input_polarization)
                S2_coef = np.cos(phi + input_polarization)

            elif isinstance(input_polarization, str) and input_polarization == "circular":
                S1_coef = 1 / np.sqrt(2)
                S2_coef = 1j / np.sqrt(2)
            else:
                raise TypeError(f"Unsupported input_polarization: {input_polarization}")

        if isinstance(output_polarization, (float, int, Quantity)):
            if isinstance(output_polarization, Quantity):
                output_polarization = output_polarization.to("rad").magnitude

            S1_coef *= np.sin(phi + output_polarization)
            S2_coef *= np.cos(phi + output_polarization) * illumination_cos_theta

        return S1_coef, S2_coef
    
    def _mie_scattering(
        self,
        L: int,
        illumination_cos_theta: np.ndarray,
        coefficients,
    ):
        A, B = coefficients(L)
        PI, TAU = mie.harmonics(illumination_cos_theta, L)

        E = [(2 * i + 1) / (i * (i + 1)) for i in range(1, L + 1)]

        S1 = sum(E[i] * A[i] * PI[i] + E[i] * B[i] * TAU[i] for i in range(L))
        S2 = sum(E[i] * B[i] * PI[i] + E[i] * A[i] * TAU[i] for i in range(L))

        return S1, S2
    
    def _common_setup(
        self,
        position,
        padding,
        output_region,
        wavelength,
        refractive_index_medium,
        collection_angle,
        z,
        working_distance,
        position_objective,
    ):
        xSize, ySize = self.get_xy_size(output_region, padding)
        voxel_size = get_active_voxel_size()
        scale = get_active_scale()

        arr = pad_image_to_fft(np.zeros((xSize, ySize))).astype(complex)

        position = np.array(position) * scale[: len(position)] * voxel_size[: len(position)]
        z = z * voxel_size[2] * scale[2]

        pupil_physical_size = working_distance * np.tan(collection_angle) * 2
        k = 2 * np.pi / wavelength * refractive_index_medium

        relative_position = np.array(
            (
                position_objective[0] - position[0],
                position_objective[1] - position[1],
                working_distance - z,
            )
        )

        return arr, voxel_size, position, z, pupil_physical_size, k, relative_position

    def get(self, *args, mode=None, **kwargs):
        mode = self.mode if mode is None else mode

        if mode == "geometric":
            return self._solve_geometric(*args, **kwargs)
        if mode == "hybrid":
            return self._solve_hybrid(*args, **kwargs)
        if mode == "fourier":
            raise NotImplementedError("Pure Fourier mode not implemented yet.")

        raise ValueError(f"Unknown mode: {mode}")
    
    def _solve_geometric(
        self,
        inp,
        position,
        voxel_size,
        padding,
        wavelength,
        refractive_index_medium,
        L,
        collection_angle,
        input_polarization,
        output_polarization,
        coefficients,
        offset_z,
        z,
        working_distance,
        position_objective,
        return_fft,
        coherence_length,
        output_region,
        illumination_angle,
        amp_factor,
        phase_shift_correction,
        pupil=None,
        **kwargs,
    ):
        arr, voxel_size, position, z, pupil_physical_size, k, relative_position = self._common_setup(
            position, padding, output_region, wavelength, refractive_index_medium,
            collection_angle, z, working_distance, position_objective
        )

        ratio = offset_z / (working_distance - z)

        R3_field, cos_theta_field, illumination_angle_field, phi_field = \
            self._plane_in_polar_coords_geometric(
                arr.shape, voxel_size, relative_position * ratio, illumination_angle
            )
        
        cos_phi_field = np.cos(phi_field)
        sin_phi_field = np.sin(phi_field)

        x_farfield = (
            position[0]
            + R3_field * np.sqrt(1 - cos_theta_field**2) * cos_phi_field / ratio
        )
        y_farfield = (
            position[1]
            + R3_field * np.sqrt(1 - cos_theta_field**2) * sin_phi_field / ratio
        )

        pupil_mask = (
            (x_farfield - position_objective[0]) ** 2
            + (y_farfield - position_objective[1]) ** 2
            < (pupil_physical_size / 2) ** 2
        )
        cos_theta_field = cos_theta_field[pupil_mask]
            

        R3_field = R3_field[pupil_mask]
        phi_field = phi_field[pupil_mask]
        illumination_angle_field = illumination_angle_field[pupil_mask]

        S1_coef, S2_coef = self._polarization_coefficients(
            phi_field, illumination_angle_field, input_polarization, output_polarization
        )
        S1, S2 = self._mie_scattering(L, illumination_angle_field, coefficients)

        arr[pupil_mask] = (
            -1j / (k * R3_field)
            * np.exp(1j * k * R3_field)
            * (S2 * S2_coef + S1 * S1_coef)
        ) / amp_factor

        # For phase shift correction (a multiplication of the field
        # by exp(1j * k * z)).
        if phase_shift_correction:
            arr *= np.exp(1j * k * z + 1j * np.pi / 2)

        # For partially coherent illumination.
        if coherence_length:
            sigma = z * np.sqrt((coherence_length / z + 1) ** 2 - 1)
            sigma = sigma * (offset_z / z)

            mask = np.zeros_like(arr)
            y, x = np.ogrid[
                -mask.shape[0] // 2 : mask.shape[0] // 2,
                -mask.shape[1] // 2 : mask.shape[1] // 2,
            ]
            mask = np.exp(-0.5 * (x ** 2 + y ** 2) / ((sigma) ** 2))
            arr = arr * mask

        fourier_field = np.fft.fft2(arr)

        propagation_matrix = get_propagation_matrix(
            fourier_field.shape,
            pixel_size=voxel_size[:2],
            wavelength=wavelength / refractive_index_medium,
            to_z=(-offset_z - z),
            dy=(
                relative_position[0] * ratio
                + position[0]
                + (padding[0] - arr.shape[0] / 2) * voxel_size[0]
            ),
            dx=(
                relative_position[1] * ratio
                + position[1]
                + (padding[2] - arr.shape[1] / 2) * voxel_size[1]
            ),
        )

        fourier_field *= propagation_matrix * np.exp(-1j * k * offset_z)

        if return_fft:
            return fourier_field[..., np.newaxis]
        return np.fft.ifft2(fourier_field)[..., np.newaxis]
    
    def _solve_hybrid(
        self,
        inp,
        position,
        voxel_size,
        padding,
        wavelength,
        refractive_index_medium,
        L,
        collection_angle,
        input_polarization,
        output_polarization,
        coefficients,
        offset_z,
        z,
        working_distance,
        position_objective,
        return_fft,
        coherence_length,
        output_region,
        illumination_angle,
        amp_factor,
        phase_shift_correction,
        pupil=None,
        **kwargs,
    ):
        arr, voxel_size, position, z, pupil_physical_size, k, relative_position = self._common_setup(
            position, padding, output_region, wavelength, refractive_index_medium,
            collection_angle, z, working_distance, position_objective
        )

        ratio = offset_z / (working_distance - z)

        
        R3_field, cos_theta_field, illumination_angle_field, phi_field, pupil_mask = \
            self._plane_in_polar_coords_hybrid(
                arr.shape, voxel_size, relative_position * ratio, illumination_angle, k
            )

        cos_phi_field = np.cos(phi_field)
        sin_phi_field = np.sin(phi_field)

        x_farfield = (
            position[0]
            + R3_field * np.sqrt(1 - cos_theta_field**2) * cos_phi_field / ratio
        )
        y_farfield = (
            position[1]
            + R3_field * np.sqrt(1 - cos_theta_field**2) * sin_phi_field / ratio
        )

        phi_valid = phi_field[pupil_mask]
        illum_valid = illumination_angle_field[pupil_mask]

        S1_coef, S2_coef = self._polarization_coefficients(
            phi_valid, illum_valid, input_polarization, output_polarization
        )
        S1, S2 = self._mie_scattering(L, illum_valid, coefficients)

        arr[pupil_mask] = (S2 * S2_coef + S1 * S1_coef) / amp_factor

        # For phase shift correction (a multiplication of the field
        # by exp(1j * k * z)).
        if phase_shift_correction:
            arr *= np.exp(1j * k * z + 1j * np.pi / 2)

        # For partially coherent illumination.
        if coherence_length:
            sigma = z * np.sqrt((coherence_length / z + 1) ** 2 - 1)
            sigma = sigma * (offset_z / z)

            mask = np.zeros_like(arr)
            y, x = np.ogrid[
                -mask.shape[0] // 2 : mask.shape[0] // 2,
                -mask.shape[1] // 2 : mask.shape[1] // 2,
            ]
            mask = np.exp(-0.5 * (x ** 2 + y ** 2) / ((sigma) ** 2))
            arr = arr * mask

        if pupil is not None and len(pupil) > 0:
            c0 = arr.shape[0] // 2
            c1 = arr.shape[1] // 2
            h0 = pupil.shape[0] // 2
            h1 = pupil.shape[1] // 2
            arr[c0 - h0:c0 + h0, c1 - h1:c1 + h1] *= pupil

        fourier_field = np.fft.ifft2(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr))))

        propagation_matrix = get_propagation_matrix(
            fourier_field.shape,
            pixel_size=voxel_size[:2],
            wavelength=wavelength / refractive_index_medium,
            to_z=(-z),
            dy=(
                relative_position[0] * ratio
                + position[0]
                + (padding[0] - arr.shape[0] / 2) * voxel_size[0]
            ),
            dx=(
                relative_position[1] * ratio
                + position[1]
                + (padding[1] - arr.shape[1] / 2) * voxel_size[1]
            ),
        )

        fourier_field *= propagation_matrix

        if return_fft:
            return fourier_field[..., np.newaxis]
        return np.fft.ifft2(fourier_field)[..., np.newaxis]


#TODO ***??*** revise MieSphere - torch, typing, docstring, unit test
class MieSphere(MieScatterer):
    """Scattered field by a sphere

    Should be calculated on at least a 64 by 64 grid. Use padding in the
    optics if necessary.

    Calculates the scattered field by a spherical particle in a homogenous
    medium, as predicted by Mie theory. Note that the induced phase shift is
    calculated in comparison to the `refractive_index_medium` property of the
    optical device.

    Parameters
    ----------
    radius: float
        Radius of the mie particle in meter.
        
    refractive_index: float
        Refractive index of the particle
        
    L: int | str
        The number of terms used to evaluate the mie theory. If `"auto"`,
        it determines the number of terms automatically.
        
    position: tuple[float, float] | tuple[float, float, float]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    offset_z: "auto" | float
        Distance from the particle in the z direction the field is evaluated.
        If "auto", this is calculated from the pixel size and
        `collection_angle`.
        
    collection_angle: "auto" | float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective
        is the limiting aperature).
        
    input_polarization: float | Quantity
        Defines the polarization angle of the input. For simulating circularly
        polarized light we recommend a coherent sum of two simulated fields.
        For unpolarized light we recommend a incoherent sum of two simulated
        fields.
        
    output_polarization: float | Quantity | None
        If None, the output light is not polarized. Otherwise defines the
        angle of the polarization filter after the sample. For off-axis,
        keep the same as input_polarization.
        
    """


    def __init__(
        self,
        radius: float = 1e-6,
        refractive_index: float = 1.45,
        **kwargs,
    ) -> None:
        def coeffs(
            radius: float,
            refractive_index: float,
            refractive_index_medium: float,
            wavelength: float
        ):

            if isinstance(radius, Quantity):
                radius = radius.to("m").magnitude
            if isinstance(wavelength, Quantity):
                wavelength = wavelength.to("m").magnitude

            def inner(L):
                return mie.coefficients(
                    refractive_index / refractive_index_medium,
                    radius * 2 * np.pi / wavelength * refractive_index_medium,
                    L,
                )

            return inner

        super().__init__(
            coefficients=coeffs,
            radius=radius,
            refractive_index=refractive_index,
            **kwargs,
        )


#TODO ***??*** revise MieStratifiedSphere - torch, typing, docstring, unit test
class MieStratifiedSphere(MieScatterer):
    """Scattered field by a stratified sphere

    A stratified sphere is a sphere with several concentric shells of uniform
    refractive index.

    Should be calculated on at least a 64 by 64 grid. Use padding in the
    optics if necessary

    Calculates the scattered field in a homogenous medium, as predicted by
    Mie theory. Note that the induced phase shift is calculated in comparison
    to the `refractive_index_medium` property of the optical device.

    Parameters
    ----------
    radius: list[float]
        The radius of each cell in increasing order.
        
    refractive_index: list[float]
        Refractive index of each cell in the same order as `radius`.
        
    L: int | str
        The number of terms used to evaluate the mie theory. If `"auto"`,
        it determines the number of terms automatically.
        
    position: tuple[float, float] | tuple[float, float, float]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    offset_z: "auto" | float
        Distance from the particle in the z direction the field is evaluated.
        If "auto", this is calculated from the pixel size and
        `collection_angle`.
        
    collection_angle: "auto" | float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective
        is the limiting aperature).
        
    input_polarization: float | Quantity
        Defines the polarization angle of the input. For simulating circularly
        polarized light we recommend a coherent sum of two simulated fields.
        For unpolarized light we recommend a incoherent sum of two
        simulated fields.
        
    output_polarization: float | Quantity | None
        If None, the output light is not polarized. Otherwise defines the angle
        of the polarization filter after the sample. For off-axis, keep the
        same as input_polarization.
        
    """


    def __init__(
        self,
        radius: tuple[float, ...] = (1e-6,),
        refractive_index: tuple[float, ...] = (1.45,),
        **kwargs,
    ) -> None:
        def coeffs(
            radius: int | str,
            refractive_index: float,
            refractive_index_medium: float,
            wavelength: float
        ):
            assert np.all(
                radius[1:] >= radius[:-1]
            ), ("Radius of the shells of a stratified sphere should be "
               "monotonically increasing")

            def inner(
                L: int
            ):
                return mie.stratified_coefficients(
                    np.array(refractive_index) / refractive_index_medium,
                    np.array(radius) * 2 * np.pi / wavelength
                    *refractive_index_medium,
                    L,
                )

            return inner

        super().__init__(
            coefficients=coeffs,
            radius=radius,
            refractive_index=refractive_index,
            **kwargs,
        )


@dataclass
class ScatteredVolume(Wrapper):
    """Voxelized volume produced by a VolumeScatterer."""

    @property
    def pos3d(self: ScatteredVolume) -> np.ndarray:
        return np.array([*self.position, self.z], dtype=float)

    @property
    def position(self: ScatteredVolume) -> np.ndarray:
        pos = self.properties.get("position", None)
        if pos is None:
            return None
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 2 and pos.shape[0] == 1:
            pos = pos[0]
        return pos


@dataclass
class ScatteredField(Wrapper):
    """Complex field produced by a FieldScatterer."""
    pass

    
class Incoherent(StructuralFeature):
    """Average intensities over orthogonal polarization states.

    This meta-feature evaluates a child feature for a set of polarization
    configurations and returns the incoherent (intensity) average.

    By default, unpolarized states are approximated by averaging over two
    orthogonal linear polarizations (0 and π/2).

    """

    __distributed__ = False

    def __init__(
        self: Incoherent,
        feature: Feature,
        input_unpolarized: bool = True,
        output_unpolarized: bool = True,
        **kwargs: Any,
    ):
        """Initializes the Incoherent feature.
        
        Parameters
        ----------
        feature: Feature
            The child feature to evaluate for different polarization states.
        input_unpolarized: bool, optional
            If True, the input light is treated as unpolarized, and the feature
            will be evaluated for two orthogonal input polarization states (0 and π/2).
        output_unpolarized: bool, optional
            If True, the output light is treated as unpolarized, and the feature
            will be evaluated for two orthogonal output polarization states (0 and π/2).
        **kwargs: dict
            Additional keyword arguments passed to the parent StructuralFeature.
                
        """

        super().__init__(
            input_unpolarized=input_unpolarized,
            output_unpolarized=output_unpolarized,
            **kwargs,
        )
        self.feature = self.add_feature(feature)

    @staticmethod
    def _states(base, unpolarized):
        """Return polarization states to sample."""

        if unpolarized:
            return (0.0, np.pi / 2)
        return (0.0 if base is None else base,)

    def get(
        self: Incoherent,
        inputs: Any,
        input_unpolarized: bool,
        output_unpolarized: bool,
        _ID: tuple = (),
        **kwargs: Any,
    ) -> Any:
        """Evaluates the feature for different polarization states and returns 
            the incoherent average.
        
        Parameters
        ----------
        inputs: Any
            The input to the feature, passed through to the child feature.
        input_unpolarized: bool
            Whether the input light is unpolarized.
        output_unpolarized: bool
            Whether the output light is unpolarized.
        _ID: tuple, optional
            The identifier for the current feature evaluation, passed through to the child feature.
        **kwargs: dict
            Additional keyword arguments passed to the child feature.   

        Returns
        -------
        Any
            The incoherent average of the feature evaluated over the specified polarization states.
        """
        # Fast path: no averaging needed
        if not input_unpolarized and not output_unpolarized:
            return self.feature(_ID=_ID, **kwargs)

        base_input = kwargs.get("input_polarization", 0.0)
        base_output = kwargs.get("output_polarization", 0.0)

        input_states = self._states(base_input, input_unpolarized)
        output_states = self._states(base_output, output_unpolarized)

        # print('IN', input_states, input_unpolarized)
        # print('OUT', output_states, output_unpolarized)

        intensity_sum = None
        count = 0

        for pin in input_states:
            for pout in output_states:
                result = self.feature(
                    _ID=_ID,
                    **kwargs,
                    input_polarization=pin,
                    output_polarization=pout,
                )

                I = np.abs(result) ** 2

                if intensity_sum is None:
                    intensity_sum = np.array(I, copy=True)
                else:
                    intensity_sum += I

                count += 1

        return intensity_sum / count