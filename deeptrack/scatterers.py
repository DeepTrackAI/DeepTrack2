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
from dataclasses import dataclass, field

from deeptrack.holography import get_propagation_matrix
from deeptrack.backend.units import (
    ConversionTable,
    get_active_scale,
    get_active_voxel_size,
)
from deeptrack.backend import mie
from deeptrack.math import AveragePooling
from deeptrack.features import Feature, MERGE_STRATEGY_APPEND
from deeptrack.image import pad_image_to_fft
from deeptrack.types import ArrayLike
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
    position:  ArrayLike[float, float (, float)]
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
        position: ArrayLike[float] = (32, 32),
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
        voxel_size: ArrayLike[int],
        upsample: int,
        upsample_axes=None,
        crop_empty=True,
        **kwargs
    ) -> list[np.ndarray]:
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
    position:  ArrayLike[float, float (, float)]
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

        scale = xp.asarray(get_active_scale(), dtype=float)

        return xp.ones((1, 1, 1), dtype=scale.dtype) * xp.prod(scale)


#TODO ***??*** revise Ellipse - torch, typing, docstring, unit test
class Ellipse(VolumeScatterer):
    """Generates an elliptical disk scatterer

    Parameters
    ----------
    radius: float | ArrayLike[float, (,float)]
        Radius of the ellipse in meters. If only one value,
        assume circular.
        
    rotation: float
        Orientation angle of the ellipse in the camera plane in radians.
        
    position: ArrayLike[float]
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
        radius = np.array(properties["radius"])
        if radius.ndim == 0:
            radius = np.array((properties["radius"], properties["radius"]))
        elif radius.size == 1:
            radius = np.array((*radius,) * 2)
        else:
            radius = radius[:2]
        properties["radius"] = radius

        return properties

    def get(
        self,
        *ignore,
        radius: ArrayLike[float] | float,
        rotation: float,
        voxel_size: float,
        transpose: float,
        **kwargs
    ) -> ArrayLike[float]:
        """Abstract method to initialize the ellipse scatterer"""
        if not transpose:
            radius = radius[::-1]
            # rotation = rotation[::-1]
        # Create a grid to calculate on.
        rad = radius[:2]
        ceil = int(np.ceil(np.max(rad) / np.min(voxel_size[:2])))
        Y, X = np.meshgrid(
            np.arange(-ceil, ceil) * voxel_size[1],
            np.arange(-ceil, ceil) * voxel_size[0],
        )

        # Rotate the grid.
        if rotation != 0:
            Xt = X * np.cos(-rotation) + Y * np.sin(-rotation)
            Yt = -X * np.sin(-rotation) + Y * np.cos(-rotation)
            X = Xt
            Y = Yt

        # Evaluate ellipse.
        mask = (
            (X * X) / (rad[0] * rad[0]) +
            (Y * Y) / (rad[1] * rad[1]) < 1
            ).astype(float)
        mask = np.expand_dims(mask, axis=-1)
        return mask


#TODO ***??*** revise Sphere - torch, typing, docstring, unit test
class Sphere(VolumeScatterer):
    """Generates a spherical scatterer

    Parameters
    ----------
    radius: float
        Radius of the sphere in meters.
        
    position: ArrayLike[float, float (, float)]
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
        image: np.ndarray,
        radius: float,
        voxel_size: float,
        **kwargs
    ) -> ArrayLike[float]:
        """Abstract method to initialize the sphere scatterer"""

        # Create a grid to calculate on.
        rad = radius * np.ones(3) / voxel_size
        rad_ceil = np.ceil(rad)
        x = np.arange(-rad_ceil[0], rad_ceil[0])
        y = np.arange(-rad_ceil[1], rad_ceil[1])
        z = np.arange(-rad_ceil[2], rad_ceil[2])
        
        X, Y, Z = np.meshgrid(
            (y / rad[1]) ** 2,
            (x / rad[0]) ** 2,
            (z / rad[2]) ** 2
        )

        mask = (X + Y + Z <= 1).astype(float)
        return mask


#TODO ***??*** revise Ellipsoid - torch, typing, docstring, unit test
class Ellipsoid(VolumeScatterer):
    """Generates an ellipsoidal scatterer

    Parameters
    ----------
    radius: float | ArrayLike[float (, float, float)]
        Radius of the ellipsoid in meters. If only one value,
        assume spherical.
        
    rotation: float
        Rotation of the ellipsoid in about the x, y and z axis.
        
    position: ArrayLike[float, float (, float)]
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
        radius = np.array(propertydict["radius"])
        if radius.ndim == 0:
            radius = np.array([radius])
        if radius.size == 1:
            
            # If only one value, assume sphere.
            radius = (*radius,) * 3
        elif radius.size == 2:
            
            # If two values, duplicate the minor axis.
            radius = (*radius, np.min(radius[-1]))
        elif radius.size == 3:
            
            # If three values, convert to tuple for consistency.
            radius = (*radius,)
        propertydict["radius"] = radius

        # Ensure rotation has three values.
        rotation = np.array(propertydict["rotation"])
        if rotation.ndim == 0:
            rotation = np.array([rotation])
        if rotation.size == 1:
            
            # If only one value, pad with two zeros.
            rotation = (*rotation, 0, 0)
        elif rotation.size == 2:
            
            # If two values, pad with one zero.
            rotation = (*rotation, 0)
        elif rotation.size == 3:
            
            # If three values, convert to tuple for consistency.
            rotation = (*rotation,)
        propertydict["rotation"] = rotation

        return propertydict

    def get(
        self,
        image: np.ndarray,
        radius: float,
        rotation: ArrayLike[float] | float,
        voxel_size: float,
        transpose: bool,
        **kwargs
    ) -> ArrayLike[float]:
        """Abstract method to initialize the ellipsoid scatterer"""
        if not transpose:
            
            # Swap the first and second value of the radius vector.
            radius = (radius[1], radius[0], radius[2])

        # radius_in_pixels = np.array(radius) / np.array(voxel_size)
        # max_rad = np.max(radius_in_pixels)
        rad_ceil = np.ceil(np.max(radius) / np.min(voxel_size))

        # Create grid to calculate on.
        x = np.arange(-rad_ceil, rad_ceil) * voxel_size[0]
        y = np.arange(-rad_ceil, rad_ceil) * voxel_size[1]
        z = np.arange(-rad_ceil, rad_ceil) * voxel_size[2]
        Y, X, Z = np.meshgrid(y, x, z)

        # Rotate the grid.
        cos = np.cos(rotation)
        sin = np.sin(rotation)
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

        mask = (
            (XR / radius[0]) ** 2 +
            (YR / radius[1]) ** 2 +
            (ZR / radius[2]) ** 2 < 1
        ).astype(float)
        return mask


#TODO ***??*** revise MieScatterer - torch, typing, docstring, unit test
class MieScatterer(FieldScatterer):
    """Base implementation of a Mie particle.

    New Mie-theory scatterers can be implemented by extending this class, and
    passing a function that calculates the coefficients of the harmonics up to
    order `L`. To be precise, the feature expects a wrapper function that takes
    the current values of the properties, as well as a inner function that
    takes an integer as the only parameter, and calculates the coefficients up
    to that integer. The return format is expected to be a tuple with two
    values, corresponding to `an` and `bn`.
    See `deeptrack.backend.mie.coefficients` for an example.

    Attributes
    ----------
    coefficients: Callable[int] -> tuple[ndarray, ndarray]
    
        Function that returns the harmonics coefficients.
        
    offset_z: "auto" | float
    
        Distance from the particle in the z direction the field is evaluated.
        If "auto", this is calculated from the pixel size and
        `collection_angle`.
        
    collection_angle: "auto" | float
    
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective is
        the limiting aperature).
        
    input_polarization: float | Quantity
    
        Defines the polarization angle of the input. For simulating circularly
        polarized light we recommend a coherent sum of two simulated fields. 
        For unpolarized light we recommend a incoherent sum of two simulated
        fields. If defined as "circular", the coefficients are set to 1/2.
        
    output_polarization: float | Quantity | None
    
        If None, the output light is not polarized. Otherwise defines the
        angle of the polarization filter after the sample. For off-axis, keep
        the same as input_polarization. If defined as "circular", the
        coefficients are multiplied by 1. I.e. no change.
        
    L: int | str
    
        The number of terms used to evaluate the mie theory. If `"auto"`,
        it determines the number of terms automatically.
        
    position: ArrayLike[float, float (, float)]
    
        The position of the particle, length 2 or 3. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
        
    z: float
    
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
        
    return_fft: bool
    
        If True, the feature returns the fft of the field, rather than the
        field itself.
        
    coherence_length: float
    
        The temporal coherence length of a partially coherent light given in
        meters. If None, the illumination is assumed to be coherent.
        
    amp_factor: float
    
        A factor that scales the amplification of the field. 
        This is useful for scaling the field to the correct intensity.
        Default is 1.
        
    phase_shift_correction: bool
    
        If True, the feature applies a phase shift correction to the output
        field. This is necessary for ISCAT simulations. 
        The correction depends on the k-vector and z according to the formula: 
        arr*=np.exp(1j * k * z + 1j * np.pi / 2)
        
    """


    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
        polarization_angle=(u.radian, u.radian),
        collection_angle=(u.radian, u.radian),
        wavelength=(u.meter, u.meter),
        offset_z=(u.meter, u.meter),
        coherence_length=(u.meter, u.pixel),
    )

    def __init__(
        self,
        coefficients,
        input_polarization: int=0,
        output_polarization: int=0,
        offset_z: str="auto",
        collection_angle: str = "auto",
        L: str = "auto",
        refractive_index_medium: float=None,
        wavelength: float=None,
        NA: float=None,
        padding=(0,) * 4,
        output_region=None,
        polarization_angle: float=None,
        working_distance: float=1000000,  # Value to avoid numerical issues.
        position_objective: tuple[float, float]=(0, 0),
        return_fft: bool=False,
        coherence_length: float=None,
        illumination_angle: float=0,
        amp_factor: float=1,
        phase_shift_correction: bool=False,
        # pupil: ArrayLike=[], # Daniel
        **kwargs,
    ) -> None:
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
            # pupil=pupil, # Daniel
            **kwargs,
        )

    def _process_properties(
        self,
        properties: dict
    ) -> dict:

        properties = super()._process_properties(properties)

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
            arr = pad_image_to_fft(np.zeros((xSize, ySize))).astype(complex)
            min_edge_size = np.min(arr.shape)
            properties["offset_z"] = (
                min_edge_size
                * 0.45
                * min(properties["voxel_size"][:2])
                / np.tan(properties["collection_angle"])
            )
        return properties

    def get_xy_size(
        self,
        output_region: ArrayLike[int],
        padding: ArrayLike[int]
    ) -> ArrayLike[int]:
        """Computes the x and y dimensions of the output region with padding.

        Parameters
        ----------
        output_region: ArrayLike[int]
            The coordinates defining the output region.

        padding: ArrayLike[int]
            The padding applied in each direction.

        Returns
        -------
        ArrayLike[int]
            The total size in x and y directions.

        """
        return (
            output_region[2] - output_region[0] + padding[0] + padding[2],
            output_region[3] - output_region[1] + padding[1] + padding[3],
        )

    def get_XY(
        self,
        shape: ArrayLike[float],
        voxel_size: ArrayLike[float]
    ) -> ArrayLike[int] :
        """Generates meshgrid for X and Y given the shape and voxel size.

        Parameters
        ----------
        shape: ArrayLike[float]
            The dimensions of the output region.

        voxel_size: ArrayLike[float]
            The size of each voxel in meters.

        Returns
        -------
        ArrayLike[int]
            The meshgrid of X and Y coordinates.

        """
        x = np.arange(shape[0]) - shape[0] / 2
        y = np.arange(shape[1]) - shape[1] / 2
        return np.meshgrid(x * voxel_size[0], y * voxel_size[1], indexing="ij")

    def get_detector_mask(
        self,
        X: float,
        Y: float,
        radius: float
    ) -> ArrayLike[bool]:
        """Creates a mask based on a circular aperture.

        Parameters
        ----------
        X: float
            X-coordinates of the field.

        Y: float
            Y-coordinates of the field.

        radius: float
            The radius of the detector aperture.

        Returns
        -------
        ArrayLike[bool]
            A boolean mask.

        """

        return np.sqrt(X ** 2 + Y ** 2) < radius

    def get_plane_in_polar_coords(
        self,
        shape: int,
        voxel_size: ArrayLike[float],
        plane_position: float,
        illumination_angle: float,
        # k: float, # Daniel
    ) -> tuple[float, float, float, float]:
        """Computes the coordinates of the plane in polar form."""

        X, Y = self.get_XY(shape, voxel_size)

        # The X, Y coordinates of the pupil relative to the particle.
        X = X + plane_position[0]
        Y = Y + plane_position[1]
        Z = plane_position[2]  # Might be +z or -z.

        R2_squared = X ** 2 + Y ** 2
        R3 = np.sqrt(R2_squared + Z ** 2)  # Might be +z instead of -z.
        
        # # DANIEL
        # Q = np.sqrt(R2_squared)/voxel_size[0]**2*2*np.pi/shape[0]
        # # is dimensionally ok?
        # sin_theta=Q/(k)
        # pupil_mask=sin_theta<1
        # cos_theta=np.zeros(sin_theta.shape)
        # cos_theta[pupil_mask]=np.sqrt(1-sin_theta[pupil_mask]**2)

        # Fet the angles.
        cos_theta = Z / R3
        
        illumination_cos_theta = (
            np.cos(np.arccos(cos_theta) + illumination_angle)
            )
        phi = np.arctan2(Y, X)

        return R3, cos_theta, illumination_cos_theta, phi#, pupil_mask # Daniel

    def get(
        self,
        inp,
        position: ArrayLike[float, float],
        voxel_size: ArrayLike[float],
        padding: ArrayLike[int],
        wavelength: float,
        refractive_index_medium: float,
        L: int | str,
        collection_angle: float,
        input_polarization: float,
        output_polarization: float,
        coefficients,
        offset_z: float,
        z: float,
        working_distance: float,
        position_objective: float,
        return_fft: bool,
        coherence_length: float,
        output_region: ArrayLike[int],
        illumination_angle: float,
        amp_factor: float,
        phase_shift_correction: bool,
        # pupil: ArrayLike, # Daniel
        **kwargs,
    ) -> ArrayLike[float]:
        """Abstract method to initialize the Mie scatterer"""
        
        # Get size of the output.
        xSize, ySize = self.get_xy_size(output_region, padding)
        voxel_size = get_active_voxel_size()
        scale = get_active_scale()
        arr = pad_image_to_fft(np.zeros((xSize, ySize))).astype(complex)
        position = np.array(position) * scale[: len(position)] * voxel_size[: len(position)]

        pupil_physical_size = working_distance * np.tan(collection_angle) * 2

        z = z * voxel_size[2]

        ratio = offset_z / (working_distance - z)

        # Wave vector.
        k = 2 * np.pi / wavelength * refractive_index_medium

        # Position of objective relative particle.
        relative_position = np.array(
            (
                position_objective[0] - position[0],
                position_objective[1] - position[1],
                working_distance - z,
            )
        )

        # Get field evaluation plane at offset_z. # , pupil_mask # Daniel
        R3_field, cos_theta_field, illumination_angle_field, phi_field =\
        self.get_plane_in_polar_coords(
            arr.shape, voxel_size,
            relative_position * ratio,
            illumination_angle,
            # k # Daniel
        )
        
        cos_phi_field, sin_phi_field = np.cos(phi_field), np.sin(phi_field)

        # x and y position of a beam passing through field evaluation plane
        # on the objective.
        x_farfield = (
            position[0] +
            R3_field * np.sqrt(1 - cos_theta_field ** 2) *
            cos_phi_field / ratio
        )
        y_farfield = (
            position[1] +
            R3_field * np.sqrt(1 - cos_theta_field ** 2) *
            sin_phi_field / ratio
        )

        # If the beam is within the pupil. Remove if Daniel
        pupil_mask = (x_farfield - position_objective[0]) ** 2 + (
            y_farfield - position_objective[1]
        ) ** 2 < (pupil_physical_size / 2) ** 2

        R3_field = R3_field[pupil_mask]
        cos_theta_field = cos_theta_field[pupil_mask]
        phi_field = phi_field[pupil_mask]

        illumination_angle_field=illumination_angle_field[pupil_mask]
        
        if isinstance(input_polarization, (float, int, str, Quantity)):
            if isinstance(input_polarization, Quantity):
                input_polarization = input_polarization.to("rad")
                input_polarization = input_polarization.magnitude

            if isinstance(input_polarization, (float, int)): 
                S1_coef = np.sin(phi_field + input_polarization) 
                S2_coef = np.cos(phi_field + input_polarization)

            # If input polarization is circular set the coefficients to 1/2.
            elif isinstance(input_polarization, (str)):
                if input_polarization == "circular":
                    S1_coef = 1/2
                    S2_coef = 1/2

        if isinstance(output_polarization, (float, int, Quantity)):
            if isinstance(input_polarization, Quantity):
                output_polarization = output_polarization.to("rad")
                output_polarization = output_polarization.magnitude

            S1_coef *= np.sin(phi_field + output_polarization)

            S2_coef *= (
                np.cos(phi_field + output_polarization)
            * illumination_angle_field
            )

        # Harmonics.
        A, B = coefficients(L)
        PI, TAU = mie.harmonics(illumination_angle_field, L)

        # Normalization factor.
        E = [(2 * i + 1) / (i * (i + 1)) for i in range(1, L + 1)]

        # Scattering terms.
        S1 = sum(
            [E[i] * A[i] * PI[i] + E[i] * B[i] * TAU[i] for i in range(0, L)]
        )

        S2 = sum(
            [E[i] * B[i] * PI[i] + E[i] * A[i] * TAU[i] for i in range(0, L)]
        )
        
        # Daniel
        # arr[pupil_mask] = (S2 * S2_coef + S1 * S1_coef)/amp_factor
        arr[pupil_mask] = (
            -1j
            / (k * R3_field)
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

        # Not sure if needed... CM
        # if len(pupil)>0:
        #     c_pix=[arr.shape[0]//2,arr.shape[1]//2] 

        #     arr[c_pix[0]-pupil.shape[0]//2:c_pix[0]+pupil.shape[0]//2,c_pix[1]-pupil.shape[1]//2:c_pix[1]+pupil.shape[1]//2]*=pupil
        
        # Daniel
        # fourier_field = -np.fft.ifft2(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr)))) 
        fourier_field = np.fft.fft2(arr)

        propagation_matrix = get_propagation_matrix(
            fourier_field.shape,
            pixel_size=voxel_size[:2], # this needs a double check
            wavelength=wavelength / refractive_index_medium,
            # to_z=(-z), # Daniel
            to_z=(-offset_z - z),
            dy=(
                relative_position[0] * ratio
                + position[0]
                + (padding[0] - arr.shape[0] / 2) * voxel_size[0]
            ),
            dx=(
                relative_position[1] * ratio
                + position[1]
                + (padding[2] - arr.shape[1] / 2) * voxel_size[1] # check if padding is top, bottom, left, right
            ),
        )

        fourier_field = (
            fourier_field * propagation_matrix * np.exp(-1j * k * offset_z) # Remove last part (from exp)) if Daniel
        )

        if return_fft:
            return fourier_field[..., np.newaxis]
        else:
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
        
    position: ArrayLike[float, float (, float)]
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
        
    position: ArrayLike[float, float (, float)]
    
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
        radius: ArrayLike[float] = [1e-6],
        refractive_index: ArrayLike[float] = [1.45],
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
class ScatteredBase:
    """Base class for scatterers (volumes and fields)."""

    array: np.ndarray | torch.Tensor
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying array."""
        return self.array.ndim

    @property
    def shape(self) -> int:
        """Number of dimensions of the underlying array."""
        return self.array.shape

    @property
    def pos3d(self) -> np.ndarray:
        return np.array([*self.position, self.z], dtype=float)

    @property
    def position(self) -> np.ndarray:
        pos = self.properties.get("position", None)
        if pos is None:
            return None
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 2 and pos.shape[0] == 1:
            pos = pos[0]
        return pos

    def copy(
        self,
        *,
        array=None,
        properties=None,
    ) -> ScatteredBase:
        """Return a shallow copy of the ScatteredBase.

        Parameters
        ----------
        array : np.ndarray | torch.Tensor | None
            Optional replacement for the internal array.
            If None, the existing array is reused.
        properties : dict | None
            Optional replacement for properties.
            If None, a shallow copy of the current properties is used.

        Returns
        -------
        ScatteredBase
            A new ScatteredBase instance.
        """
        return ScatteredBase(
            array=self.array if array is None else array,
            properties=self.properties.copy() if properties is None else properties,
        )


    def as_array(self) -> ArrayLike:
        """Return the underlying array.

        Notes
        -----
        The raw array is also directly available as ``scatterer.array``.
        This method exists mainly for API compatibility and clarity.

        """
        
        return self.array

    def get_property(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, self.properties.get(key, default))


@dataclass
class ScatteredVolume(ScatteredBase):
    """Voxelized volume produced by a VolumeScatterer."""
    pass


@dataclass
class ScatteredField(ScatteredBase):
    """Complex field produced by a FieldScatterer."""
    pass