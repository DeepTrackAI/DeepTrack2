"""Classes that implement light-scattering objects.

This module provides implementations of scattering objects with geometries
commonly encountered in experimental microscopy, such as ellipsoids, spheres,
and point particles.

These scatterers are primarily used together with the `Optics` module to
simulate how an optical system (e.g., brightfield or fluorescence microscopy)
images an object under a given configuration (NA, wavelength, refractive index,
etc.).

Scatterers produce either voxelized volumes (for geometrical optics models)
or complex fields (for wave-optical models such as Mie scattering).

Volume-based scatterers are evaluated on a discrete grid defined by the active
optics configuration, and can be supersampled (`upsample`) for improved
accuracy. Upsampling does not change the physical size of the scatterer, but
rather the resolution at which it is evaluated. Field-based scatterers are
evaluated directly as complex fields without supersampling.

`Upsample` should not be confused with `Optics.upscale`, which applies to the
entire imaging pipeline and can be used to improve the accuracy of the optics
model itself.

Key Features
------------
- **Customizable Geometries**

    Initialization parameters allow full control over shape, size, and spatial
    positioning. Multiple scatterers can be combined and overlaid using feature
    composition. For example, two orthogonal ellipses can form a cross, or two
    concentric spheres can represent a core–shell particle.

- **Defocusing**

    The `z` parameter defines the axial position relative to the focal plane,
    enabling simulation of defocused imaging by assigning nonzero values.

- **Fluorescence discretization**

    Some scatterers include measure corrections to ensure consistent
    fluorescence scaling under discretization. Point-like emitters are scaled
    by voxel volume, planar emitters by axial voxel size, while volumetric
    emitters require no additional correction beyond their voxelized support.

- **Mie Scatterers**

    Includes Mie-theory-based scatterers that compute scattering harmonics up
    to a specified order using utilities from `deeptrack.backend.mie`.
    Supported implementations include homogeneous spheres and stratified
    spheres with multiple concentric layers of distinct refractive indices.

- **Backend Compatibility**

    Geometry-based scatterers support both NumPy and PyTorch arrays.
    Mie-based scatterers currently rely on NumPy implementations and
    do not fully support PyTorch execution.

Module Structure
----------------
Classes:

- `Scatterer`: Abstract base class for all scatterers.
    Stores positional information and implements utilities for coordinate
    conversion, upsampling, and cropping.
- `VolumeScatterer`: Base class for scatterers that generate voxelized volumes.
    Produces `ScatteredVolume` outputs representing spatial occupancy.
- `FieldScatterer`: Base class for scatterers that generate complex fields.
    Produces `ScatteredField` outputs representing optical fields.
- `PointParticle`: Generates diffraction-limited point particles.
- `Ellipse`: Generates 2-D elliptical particles.
- `Sphere`: Generates 3-D spheres.
- `Ellipsoid`: Generates 3-D ellipsoids.
- `MieScatterer`: Mie scatterer base class.
- `MieSphere`: Extends `MieScatterer` to the spherical case.
- `MieStratifiedSphere`:  Extends `MieScatterer` to the stratified sphere case.
    A stratified sphere consists of concentric shells with distinct refractive
    indices.
- `Incoherent`: A wrapper to treat coherent scatterers as incoherent sources.

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
...    aperture_angle=0.1,
... )

>>> imaged_scatterer = optics(scatterer) # Creates an array of complex numbers.
>>> abs_imaged_scatterer = dt.Abs(imaged_scatterer)

>>> abs_imaged_scatterer.plot()

"""

from __future__ import annotations

import warnings
from typing import Any, TYPE_CHECKING

import array_api_compat as apc
import numpy as np
from pint import Quantity
from dataclasses import dataclass

from deeptrack.optical.holography import get_propagation_matrix
from deeptrack.backend.units import (
    ConversionTable,
    get_active_scale,
    get_active_voxel_size,
)
from deeptrack.backend import config, mie, TORCH_AVAILABLE, xp
from deeptrack.optical.math import AveragePooling, pad_image_to_fft
from deeptrack.features import (
    Feature,
    StructuralFeature,
    MERGE_STRATEGY_APPEND,
)
from deeptrack.wrappers import Wrapper
from deeptrack import units_registry as u

if TORCH_AVAILABLE:
    import torch


def _asarray(value, dtype=None):
    """Convert values through xp while preserving existing tensor gradients."""

    is_current_backend_array = (
        config.get_backend() == "numpy"
        and apc.is_numpy_array(value)
        or config.get_backend() == "torch"
        and apc.is_torch_array(value)
    )

    if is_current_backend_array:
        return xp.astype(value, dtype) if dtype is not None else value

    if dtype is not None:
        return xp.asarray(value, dtype=dtype)
    return xp.asarray(value)


def _asarray_vector(value, dtype=None):
    """Convert a vector-like value without detaching tensor elements."""

    if isinstance(value, (list, tuple)) and any(
        apc.is_array_api_obj(element) for element in value
    ):
        if TORCH_AVAILABLE and any(torch.is_tensor(e) for e in value):
            elements = []
            for e in value:
                if not torch.is_tensor(e):
                    e = torch.tensor(e, dtype=dtype)
                elif dtype is not None:
                    e = e.to(dtype=dtype)
                elements.append(e.reshape(()))
            return torch.stack(elements)
        return xp.stack(
            [xp.reshape(_asarray(element, dtype), ()) for element in value]
        )

    return xp.reshape(_asarray(value, dtype), (-1,))


__all__ = [
    "Scatterer",
    "PointParticle",
    "Ellipse",
    "Sphere",
    "Ellipsoid",
    "MieScatterer",
    "MieSphere",
    "MieStratifiedSphere",
    "Incoherent",
]


if TYPE_CHECKING:
    import torch


class Scatterer(Feature):
    """Base abstract class for scatterers.

    A `Scatterer` defines an object or optical source term to be evaluated on a
    discrete spatial grid. Depending on the subclass, the result may represent
    either a voxelized volume (`VolumeScatterer`) or a complex field
    (`FieldScatterer`).

    Parameters
    ----------
    position: tuple[float, float] | tuple[float, float, float], optional
        The position of the particle, length 2 or 3. Third index is optional,
        and represents the position in the direction normal to the camera
        plane. Default is (32.0, 32.0).
    z: float, optional
        The position in the direction normal to the camera plane. Used if
        `position` is of length 2. Default is 0.0.
    value: float, optional
        A default value of the characteristic of the particle. Used by optics
        unless a more direct property is set (eg. `refractive_index` for
        `Brightfield` and `intensity` for `Fluorescence`). Default is 1.0.
    position_unit: str, optional
        The unit of the provided position property. Can be "meter" or "pixel".
        Default is "pixel".
    upsample: int, optional
        Geometry supersampling factor for volume-based scatterers. The
        scatterer is evaluated on a finer grid and downsampled by average
        pooling. Ignored by field-based scatterers.
    upsample_axes: tuple of int, optional
        Deprecated. Previously selected the axes along which supersampling was
        applied. This parameter is now ignored.
    voxel_size: array-like, optional
        The size of the voxels in meters. If not provided, it is obtained from
        the active optics configuration.
    pixel_size: array-like, optional
        The size of the pixels in meters. If not provided, it is obtained from
        the active optics configuration.
    **kwargs: Any
        Additional feature properties forwarded to the parent `Feature` class.

    Methods
    -------
    `_antialias_volume(volume, factor) -> array`
        Geometry-only supersampling anti-aliasing.
    `_process_properties(properties) -> dict`
        Preprocess the input to the method `.get()`. This method is called
        before the scatterer is evaluated.
    `_process_and_get(...) -> list[array]`
        Post-processes the created object.
    `_wrap_output(array, props) -> ScatteredVolume or ScatteredField`
        Wraps the output of the scatterer in the appropriate class.

    Notes
    -----
    For developers extending the class hierarchy:
    __list_merge_strategy__: str
        The strategy for merging lists of properties when multiple scatterers
        are combined. Default is "append", which concatenates the lists.
    __distributed__: bool
        Determines whether `.get(image, **kwargs)` is applied to each element
        of the input list independently (`__distributed__ = True`) or to the
        list as a whole (`__distributed__ = False`).
    __conversion_table__: ConversionTable
        A table defining the physical units of the scatterer's properties and
        how to convert them to the internal units used for calculations.

    """

    __list_merge_strategy__ = MERGE_STRATEGY_APPEND
    __distributed__ = False
    __conversion_table__ = ConversionTable(
        position=(u.pixel, u.pixel),
        z=(u.zpixel, u.zpixel),
        voxel_size=(u.meter, u.meter),
    )

    def __init__(
        self: Scatterer,
        position: tuple[float, float] | tuple[float, float, float] = (
            32.0,
            32.0,
        ),
        z: float = 0.0,
        value: float = 1.0,
        position_unit: str = "pixel",
        upsample: int = 1,
        voxel_size: tuple | None = None,
        pixel_size: tuple | None = None,
        **kwargs,
    ):
        """Initialize the scatterer with the given properties."""

        upsample_axes = kwargs.pop("upsample_axes", None)

        if upsample_axes is not None:
            warnings.warn(
                "`upsample_axes` is deprecated and will be removed in a "
                "future release. Supersampling is now applied uniformly to "
                "all applicable axes.",
                DeprecationWarning,
                stacklevel=2,
            )

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

    def _antialias_volume(
        self: Scatterer,
        volume: np.ndarray | torch.Tensor,
        factor: int,
    ) -> np.ndarray | torch.Tensor:
        """Geometry-only supersampling anti-aliasing.

        Assumes `volume` was generated on a grid oversampled by `factor`
        and downsamples it back by average pooling.

        Parameters
        ----------
        volume: np.ndarray or torch.Tensor
            The oversampled volume to be downsampled.
        factor: int
            The factor by which the volume is oversampled.

        Returns
        -------
        np.ndarray or torch.Tensor
            The downsampled volume after anti-aliasing.

        """

        if factor == 1:
            return volume

        # Avoid pooling along dimensions smaller than the pooling factor
        # (e.g., Z=1 for 2D scatterers like Ellipse)
        shape = volume.shape
        pool = tuple(factor if s >= factor else 1 for s in shape)
        # average pooling conserves fractional occupancy
        return AveragePooling(pool)(volume)

    def _process_properties(
        self: Scatterer,
        properties: dict,
    ) -> dict:
        """Preprocess the input to the method `.get()`

        This method is called before the scatterer is evaluated, and can be
        used to preprocess the input properties.

        Parameters
        ----------
        properties: dict
            The properties of the scatterer, which are passed to the method
            `.get()`. This method can modify the properties before they are
            used for evaluation.

        Returns
        -------
        dict
            The processed properties to be used for evaluation.

        """

        # Rescales the position property.
        properties = super()._process_properties(properties)
        self._processed_properties = True
        return properties

    def _process_and_get(
        self: Scatterer,
        *args: Any,
        voxel_size: np.ndarray,
        upsample: int,
        upsample_axes: tuple | None = None,
        crop_empty: bool = True,
        **kwargs: Any,
    ) -> list[np.ndarray | torch.Tensor]:
        """Post-processes the created object.

        Post-process the created object to handle upsampling, as well as
        cropping empty slices.

        Parameters
        ----------
        *args: Any
            Positional arguments passed to the method. Not used in this
            implementation.
        voxel_size: array
            Voxel size supplied by the feature pipeline. Field scatterers use
            this value directly; volume scatterers use the active optics
            context to keep geometry evaluation aligned with upsampling.
        upsample: int
            Geometry supersampling factor for volume-based scatterers. Ignored
            by field-based scatterers.
        upsample_axes: tuple of ints, optional
            Deprecated. Previously selected the axes along which supersampling
            was applied. This parameter is now ignored, and supersampling is
            applied uniformly to all applicable axes when `upsample` > 1.
        crop_empty: bool, optional
            Whether to remove slices in which all elements are zero. This can
            be used to reduce the size of the created scatterer, which can be
            beneficial for memory and computational efficiency when the
            scatterer is small compared to the voxel size. Default is True.

        Returns
        -------
        list of array or tensor
            The created scatterer after post-processing.

        """

        if not self._processed_properties:

            warnings.warn(
                "Overridden _process_properties method does not call super. "
                + "This is likely to result in errors if used with "
                + "Optics.upscale != 1."
            )

        if isinstance(self, FieldScatterer) and voxel_size is not None:
            voxel_size = _asarray(voxel_size, dtype=xp.float64)
        else:
            voxel_size = xp.asarray(get_active_voxel_size(), dtype=float)

        apply_supersampling = upsample > 1 and isinstance(
            self, VolumeScatterer
        )

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

        if (
            new_image.numel() == 0
            if apc.is_torch_array(new_image)
            else new_image.size == 0
        ):
            warnings.warn(
                "Scatterer created that is smaller than a pixel. "
                + "This may yield inconsistent results."
                + " Consider using upsample on the scatterer,"
                + " or upscale on the optics.",
                Warning,
            )

        # Crops empty slices
        if crop_empty:
            mask_z = ~xp.all(new_image == 0, axis=(1, 2))
            mask_y = ~xp.all(new_image == 0, axis=(0, 2))
            mask_x = ~xp.all(new_image == 0, axis=(0, 1))

            new_image = new_image[mask_z][:, mask_y][:, :, mask_x]

        return [self._wrap_output(new_image, kwargs)]

    def _wrap_output(
        self: Scatterer, array: np.ndarray | torch.Tensor, props: dict
    ) -> ScatteredVolume | ScatteredField:
        """Wraps the output of the scatterer in the appropriate class.

        This method must be implemented by subclasses to wrap the output in the
        appropriate type (`ScatteredVolume` or `ScatteredField`).

        Parameters
        ----------
        array: np.ndarray or torch.Tensor
            The array or tensor representing the scatterer volume or field.
        props: dict
            The properties of the scatterer, which are passed to the
            constructor of the ScatteredVolume or ScatteredField class.

        Returns
        -------
        ScatteredVolume or ScatteredField
            The wrapped scatterer output.

        """

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _wrap_output()"
        )


class VolumeScatterer(Scatterer):
    """Abstract scatterer producing ScatteredVolume outputs."""

    def _wrap_output(
        self: VolumeScatterer,
        array: np.ndarray | torch.Tensor,
        props: dict,
    ) -> ScatteredVolume:
        """Abstract scatterer producing ScatteredVolume outputs.

        This method wraps the output of the scatterer in a ScatteredVolume
        object, which is used to represent the spatial occupancy of the
        scatterer. The properties of the scatterer are passed to the
        constructor of the ScatteredVolume class.

        Parameters
        ----------
        array: np.ndarray or torch.Tensor
            The array or tensor representing the scatterer volume.
        props: dict
            The properties of the scatterer, which are passed to the
            constructor of the ScatteredVolume class.

        Returns
        -------
        ScatteredVolume
            The wrapped scatterer output.

        """

        return ScatteredVolume(
            array=array,
            properties=props.copy(),
        )


class FieldScatterer(Scatterer):
    """Abstract scatterer producing ScatteredField outputs."""

    def _wrap_output(
        self: FieldScatterer,
        array: np.ndarray | torch.Tensor,
        props: dict,
    ) -> ScatteredField:
        """Abstract scatterer producing ScatteredField outputs.

        This method wraps the output of the scatterer in a ScatteredField
        object, which is used to represent the complex field produced by the
        scatterer. The properties of the scatterer are passed to the
        constructor of the ScatteredField class.

        Parameters
        ----------
        array: np.ndarray or torch.Tensor
            The array or tensor representing the scatterer field.
        props: dict
            The properties of the scatterer, which are passed to the
            constructor of the ScatteredField class.

        Returns
        -------
        ScatteredField
            The wrapped scatterer output.

        """

        return ScatteredField(
            array=array,
            properties=props.copy(),
        )


class PointParticle(VolumeScatterer):
    """Generate a diffraction-limited point particle.

    A point particle is represented by a single voxel. Subpixel positioning is
    handled at the optics level.

    For fluorescence imaging, a point particle is a zero-dimensional emitter
    represented on a discrete voxel grid. To preserve the correct emitted
    measure under discretization, the returned voxel is scaled by the voxel
    volume.

    Parameters
    ----------
    position:  tuple[float, float] | tuple[float, float, float] = (32.0, 32.0)
        Particle position in 2D or 3D. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float, optional
        The position in the direction normal to the camera plane. Used if
        `position` is of length 2.
    value : float, optional
        A default value of the characteristic of the particle. Used by
        `optics` unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).

    """

    def __init__(
        self: PointParticle,
        **kwargs: Any,
    ):
        """Initialize the point particle scatterer."""

        kwargs.pop("upsample", None)
        super().__init__(upsample=1, **kwargs)

    def get(
        self: PointParticle,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Return the voxelized point particle.

        The point particle is represented by a single voxel. For fluorescence
        imaging, this voxel is scaled by the voxel volume so that the discrete
        source has the correct measure under changes in grid resolution.

        Parameters
        ----------
        *args: Any
            Positional arguments passed to the method. Not used in this
            implementation.
        **kwargs: Any
            Keyword arguments passed to the method. Not used in this
            implementation.

        Returns
        -------
        np.ndarray or torch.Tensor
            A (1, 1, 1) array or tensor representing the point particle.

        """

        scale = xp.asarray(get_active_scale(), dtype=xp.float32)
        mask = xp.ones((1, 1, 1), dtype=scale.dtype) * xp.prod(scale)
        return mask


class Ellipse(VolumeScatterer):
    """Generate a 2D elliptical scatterer.

    Build a 2D ellipse on a voxel grid, defined by its radii and rotation.
    The ellipse is represented as a planar object embedded in a 3D voxel grid,
    with support on a single z-slice. For fluorescence imaging, the discrete
    mask is therefore scaled by the axial voxel size to account for the missing
    thickness of the continuous emitter.

    Supports both NumPy and PyTorch backends.

    Parameters
    ----------
    radius: float | tuple[float, float]
        Radius of the ellipse in meters. If a single value is provided, a
        circular shape is assumed.
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
        If True, the radius components are aligned with the (y, x) axes before
        rotation.

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

    def _process_properties(self, properties: dict) -> dict:
        """Preprocess the input to the method .get()

        Ensures that the radius is an array of length 2. If the radius
        is a single value, the particle is made circular.

        """

        properties = super()._process_properties(properties)

        radius = properties["radius"]
        r = xp.asarray(radius)

        if r.ndim == 0:
            r = xp.stack([r, r])
        elif r.shape[0] == 1:
            r = xp.stack([r[0], r[0]])
        else:
            r = r[:2]

        properties["radius"] = r

        return properties

    def get(
        self: Ellipse,
        *args: Any,
        radius: np.ndarray | torch.Tensor | float,
        rotation: float,
        voxel_size: np.ndarray | torch.Tensor,
        transpose: bool,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Evaluate the ellipse on a voxel grid.

        The ellipse is defined by its radii and rotation and evaluated on a
        grid with spacing given by `voxel_size`.

        For fluorescence imaging, the returned planar mask is scaled by the
        axial voxel size so that the discrete source has the correct measure
        under changes in z-resolution.

        Parameters
        ----------
        radius : array-like
            Radii of the ellipse along the principal axes.
        rotation : float
            Rotation angle in radians.
        voxel_size : array-like
            Size of voxels along each axis.
        transpose : bool
            Whether to align radii with (y, x) axes before rotation.

        Returns
        -------
        np.ndarray or torch.Tensor
            An array representing the elliptical mask.

        """

        rotation = xp.asarray(rotation)

        # swap to match (y, x) convention
        if not transpose:
            radius = xp.stack([radius[1], radius[0]])

        # Create a grid to calculate on.
        rad = radius[:2]
        rad_ceil = int(xp.ceil(xp.max(rad) / xp.min(voxel_size)).item())
        Y, X = xp.meshgrid(
            xp.arange(-rad_ceil, rad_ceil) * voxel_size[1],
            xp.arange(-rad_ceil, rad_ceil) * voxel_size[0],
            indexing="xy",
        )

        cos = xp.cos(-rotation)
        sin = xp.sin(-rotation)
        Xt = X * cos + Y * sin
        Yt = -X * sin + Y * cos

        # Evaluate ellipse.
        mask = xp.asarray(
            (Xt * Xt) / (rad[0] * rad[0]) + (Yt * Yt) / (rad[1] * rad[1]) < 1,
            dtype=xp.float32,
        )
        mask = xp.expand_dims(mask, axis=-1)

        scale = xp.asarray(get_active_scale(), dtype=xp.float32)
        # The returned value is scaled to preserve intensity
        # under discretization.
        mask = mask * scale[2]
        return mask


class Sphere(VolumeScatterer):
    """Generate a spherical scatterer.

    `Sphere` is a true volumetric scatterer. Its support spans a 3D voxelized
    region, so the correct spatial measure is already represented by the extent
    of the discrete mask. No additional fluorescence measure correction is
    required.

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

    def __init__(self, radius: float = 1e-6, **kwargs):
        """Initialize the sphere scatterer."""

        super().__init__(radius=radius, **kwargs)

    def get(
        self,
        *args: Any,
        radius: float,
        voxel_size: np.ndarray | torch.Tensor,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Evaluate the sphere on a voxel grid.

        The sphere is defined by its radius, and evaluated on a grid with
        spacing given by `voxel_size`. The returned value is a 3D array where
        each voxel is assigned 1 if inside the sphere and 0 otherwise..

        Parameters
        ----------
        args: Any
            Positional arguments passed to the method. Not used in this
            implementation.
        radius : float
            Radius of the sphere in meters.
        voxel_size : array-like
            Size of voxels along each axis.
        kwargs: Any
            Keyword arguments passed to the method.

        Returns
        -------
        np.ndarray or torch.Tensor
            A 3D array representing the spherical mask.

        """

        # Create a grid to calculate on.
        voxel_size = xp.asarray(voxel_size)
        rad = xp.asarray(radius) / voxel_size
        rad = xp.broadcast_to(rad, (3,))
        rad_ceil = xp.ceil(rad)

        x = xp.arange(-rad_ceil[0], rad_ceil[0])
        y = xp.arange(-rad_ceil[1], rad_ceil[1])
        z = xp.arange(-rad_ceil[2], rad_ceil[2])

        X, Y, Z = xp.meshgrid(
            (y / rad[1]) ** 2,
            (x / rad[0]) ** 2,
            (z / rad[2]) ** 2,
            indexing="xy",
        )

        mask = xp.asarray(
            X + Y + Z <= 1,
            dtype=xp.float32,
        )
        return mask


class Ellipsoid(VolumeScatterer):
    """Generates an ellipsoidal scatterer.

    `Ellipsoid` is a true volumetric scatterer. Its support spans a 3D
    voxelized region, so the correct spatial measure is already represented by
    the extent of the discrete mask. No additional fluorescence measure
    correction is required.

    Parameters
    ----------
    args: Any
        Positional arguments passed to the method. Not used in this
        implementation.
    radius: float | tuple[float, float, float]
        Radius of the ellipsoid in meters. If only one value,
        assume spherical.
    rotation: float | tuple[float, float, float]
        Rotation angles (rx, ry, rz) applied in XYZ order.
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
    kwargs: Any
        Keyword arguments passed to the method.

    """

    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        radius: (
            float | tuple[float, float] | tuple[float, float, float]
        ) = 1e-6,
        rotation: float | tuple[float, float] | tuple[float, float, float] = 0,
        transpose: bool = False,
        **kwargs,
    ):
        """Initialize the ellipsoid scatterer."""

        super().__init__(
            radius=radius, rotation=rotation, transpose=transpose, **kwargs
        )

    def _process_properties(self, propertydict: dict) -> dict:
        """Preprocess the input to the method `.get()`

        Ensures that the radius and the rotation properties both are arrays of
        length 3.

        If the radius is a single value, the particle is made a sphere
        If the radius are two values, the smallest value is appended as the
        third value

        The rotation vector is padded with zeros until it is of length 3.

        Parameters
        ----------
        propertydict: dict
            The properties of the scatterer, which are preprocessed and passed
            to the `get` method.

        Returns
        -------
        dict
            The preprocessed properties of the scatterer.

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
            rot = rot[:3]
        propertydict["rotation"] = rot

        return propertydict

    def get(
        self: Ellipsoid,
        *args: Any,
        radius: np.ndarray | float,
        rotation: np.ndarray | float,
        voxel_size: np.ndarray | float,
        transpose: bool,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Evaluate the ellipsoid on a voxel grid.

        The ellipsoid is defined by its radii and rotation, and evaluated on a
        grid with spacing given by `voxel_size`. The returned value is a 3D
        array where each voxel is assigned 1 if inside the ellipsoid and 0
        otherwise.

        Parameters
        ----------
        args: Any
            Positional arguments passed to the method. Not used in this
            implementation.
        radius : array-like
            Radii of the ellipsoid along the principal axes.
        rotation : array-like of length 3
            Rotation angles (rx, ry, rz) applied in XYZ order.
        voxel_size : array-like
            Size of voxels along each axis.
        transpose : bool
            Whether to align the first axis of the radius with the first axis
            of the created volume before rotation.
        kwargs: Any
            Keyword arguments passed to the method.

        Returns
        -------
        np.ndarray or torch.Tensor
            A (X, Y, Z) array representing the ellipsoidal mask.


        """

        radius = xp.asarray(radius)
        rotation = xp.asarray(rotation)
        voxel_size = xp.asarray(voxel_size)

        if not transpose:
            # Swap the first and second value of the radius vector.
            radius = xp.stack([radius[1], radius[0], radius[2]])

        rad_ceil = int(xp.ceil(xp.max(radius) / xp.min(voxel_size)).item())

        # Create grid to calculate on.
        x = xp.arange(-rad_ceil, rad_ceil) * voxel_size[0]
        y = xp.arange(-rad_ceil, rad_ceil) * voxel_size[1]
        z = xp.arange(-rad_ceil, rad_ceil) * voxel_size[2]
        Y, X, Z = xp.meshgrid(y, x, z, indexing="xy")

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
            (XR / radius[0]) ** 2
            + (YR / radius[1]) ** 2
            + (ZR / radius[2]) ** 2
            <= 1,
            dtype=xp.float32,
        )
        return mask


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

    Parameters
    ----------
    coefficients: callable
        Factory function receiving the current feature properties and returning
        a callable `f(L)` that yields the Mie coefficients `(an, bn)` up to
        order `L`.
    offset_z: "auto" | float
        Distance from the particle in the z direction where the field is
        evaluated. If `"auto"`, this is calculated from the pixel size and
        `collection_angle`.
    collection_angle: "auto" | float
        Maximum collection angle in radians. If `"auto"`, this is computed
        from the objective NA (assuming the objective is the limiting
        aperture).
    input_polarization: float | Quantity | str
        Polarization angle of the incident illumination in radians. If a float
        (or `Quantity`), it specifies the orientation of a linear polarizer
        before the sample. If set to `"circular"`, circular polarization is
        approximated by assigning equal weights to the two orthogonal
        scattering components. `None` is not supported in coherent mode. Use
        `Incoherent` to model unpolarized illumination.
    output_polarization: float | Quantity
        Angle of a polarization analyzer placed after the sample, in radians.
        If a float (or `Quantity`), the detected field is projected onto the
        corresponding linear polarization direction. `None` is not supported in
        coherent mode. Use `Incoherent` to model detection without analyzer.
    L: int | str
        Number of terms used to evaluate the Mie series. If `"auto"`,
        the number of terms is determined automatically.
    position: tuple[float, float] | tuple[float, float, float]
        Particle position. If three values are provided, the third
        corresponds to the axial position relative to the camera plane.
    z: float
        Axial particle position if `position` is two-dimensional.
    return_fft: bool
        If True, the feature returns the Fourier transform of the field
        rather than the spatial field itself.
    coherence_length: float | None
        Temporal coherence length of the illumination in meters. If None,
        illumination is assumed to be fully coherent.
    amp_factor: float
        Scaling factor applied to the scattered field amplitude.
    phase_shift_correction: bool
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
          `S1` and `S2` mapped to spatial frequencies corresponding to the
          objective pupil. The field is then propagated to the detector.
          This approach is less sensitive to the simulated field-of-view and
          generally more numerically stable.
    pupil: None | ndarray
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
        self: MieScatterer,
        coefficients: callable,
        input_polarization: float | Quantity | str = 0,
        output_polarization: float | Quantity = 0,
        offset_z: str | float = "auto",
        collection_angle: str | float = "auto",
        L: str | int = "auto",
        refractive_index_medium: float | None = None,
        wavelength: float | None = None,
        NA: float | None = None,
        padding: tuple[int, int, int, int] = (0,) * 4,
        output_region: tuple[int, int, int, int] | None = None,
        polarization_angle: float | None = None,
        working_distance: float = 1000000,
        position_objective: tuple[float, float] = (0, 0),
        return_fft: bool = False,
        coherence_length: float | None = None,
        illumination_angle: float = 0,
        amp_factor: float = 1,
        phase_shift_correction: bool = False,
        mode: str = "geometric",
        pupil: np.ndarray | None = None,
        **kwargs: Any,
    ):
        """Initialize the Mie scatterer.

        Parameters
        ----------
        coefficients: callable
            Factory function receiving the current feature properties and
            returning a callable `f(L)` that yields the Mie coefficients
            `(an, bn)` up to order `L`.
        input_polarization: float | Quantity | str
            Polarization angle of the incident illumination in radians. If a
            float (or `Quantity`), it specifies the orientation of a linear
            polarizer before the sample. If set to `"circular"`, circular
            polarization is approximated by assigning equal weights to the two
            orthogonal scattering components. `None` is not supported in
            coherent mode. Use `Incoherent` to model unpolarized illumination.
        output_polarization: float | Quantity
            Angle of a polarization analyzer placed after the sample, in
            radians. If a float (or `Quantity`), the detected field is
            projected onto the corresponding linear polarization direction.
            `None` is not supported in coherent mode. Use `Incoherent` to
            model detection without analyzer.
        offset_z: "auto" | float
            Distance from the particle in the z direction where the field is
            evaluated. If `"auto"`, this is calculated from the pixel size and
            `collection_angle`.
        collection_angle: "auto" | float
            Maximum collection angle in radians. If `"auto"`, this is computed
            from the objective NA (assuming the objective is the limiting
            aperture).
        L: "auto" | int
            Number of terms used to evaluate the Mie series. If `"auto"`,
            the number of terms is determined automatically.
        refractive_index_medium: float | None
            Refractive index of the surrounding medium. Required for automatic
            determination of `L` and `collection_angle`.
        wavelength: float | None
            Wavelength of the illumination in meters. Required for automatic
            determination of `L`.
        NA: float | None
            Numerical aperture of the objective. Required for automatic
            determination of `collection_angle`.
        padding: tuple[int, int, int, int]
            Padding applied to the output field in (left, top, right, bottom)
            order.
        output_region: tuple[int, int, int, int] | None
            The region of the output field to return, defined as (x_start,
            y_start, x_end, y_end). If None, the entire field is returned.
        polarization_angle: float
            Deprecated alias for `input_polarization`. Please use
            `input_polarization` instead.
        working_distance: float
            Distance from the objective to the focal plane in meters. Used for
            calculating the phase curvature of the field at the objective
            pupil.
        position_objective: tuple[float, float]
            Lateral position of the objective relative to the particle in
            meters. Used for calculating the phase curvature of the field at
            the objective pupil.
        return_fft: bool
            If True, the feature returns the Fourier transform of the field
            rather than the spatial field itself.
        coherence_length: float | None
            Temporal coherence length of the illumination in meters. If None,
            illumination is assumed to be fully coherent.
        illumination_angle: float
            Angle of illumination relative to the optical axis in radians. Used
            for calculating the phase curvature of the field at the objective
            pupil.
        amp_factor: float
            Scaling factor applied to the scattered field amplitude.
        phase_shift_correction: bool
            If True, applies a phase correction to the field according to
            arr *= exp(1j * k * z + 1j * π / 2). This correction is used in
            ISCAT simulations.
        mode : {"geometric", "hybrid"}
            Determines how the scattered field is constructed before
            propagation. Both modes use the same Mie coefficients but differ in
            how the scattered field is represented prior to propagation through
            the optical system.
            - "geometric"
              Evaluates the scattered field as a spherical wave on a virtual
              plane located at ``offset_z`` from the particle. The field
              includes the geometric propagation factor ``exp(i k R) / R`` and
              is sampled on a finite spatial grid before being propagated
              through the optical system. Because the field is computed on a
              finite plane, the result can be sensitive to the simulated
              field-of-view.
            - "hybrid"
              Constructs the scattered field using the Mie scattering
              amplitudes `S1` and `S2` mapped to spatial frequencies
              corresponding to the objective pupil. The field is then
              propagated to the detector. This approach is less sensitive to
              the simulated field-of-view and generally more numerically
              stable.
        pupil: None | ndarray
            Optional pupil function applied to the scattered field. This can be
            used to simulate aberrations or other modifications of the optical
            system.

        """

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
        self: MieScatterer,
        properties: dict,
    ) -> dict:
        """Validate and infer Mie-scatterer properties.

        This method enforces coherent-mode polarization requirements and
        resolves automatic values for `L`, `collection_angle`, and `offset_z`
        from the current optical configuration.

        Parameters
        ----------
        properties: dict
            Scatterer properties after base preprocessing.

        Returns
        -------
        dict
            Processed property dictionary.

        """

        properties = super()._process_properties(properties)

        # --- polarization validation ---
        inp = properties.get("input_polarization", None)
        out = properties.get("output_polarization", None)

        if inp is None:
            raise ValueError(
                "input_polarization must be specified for coherent "
                "scattering. Use the Incoherent feature to model unpolarized "
                "illumination."
            )

        if out is None:
            raise ValueError(
                "output_polarization=None (no analyzer) is not supported in "
                "coherent mode. Use the Incoherent feature to model detection "
                "without analyzer."
            )

        if properties["L"] == "auto":
            try:
                radius_for_l = properties["radius"]
                if TORCH_AVAILABLE and torch.is_tensor(radius_for_l):
                    radius_for_l = radius_for_l.detach().cpu().numpy()
                wavelength_for_l = properties["wavelength"]
                if TORCH_AVAILABLE and torch.is_tensor(wavelength_for_l):
                    wavelength_for_l = (
                        wavelength_for_l.detach().cpu().numpy()
                    )

                v = (
                    2
                    * np.pi
                    * np.max(radius_for_l)
                    / wavelength_for_l
                )

                properties["L"] = int(np.floor((v + 4 * (v ** (1 / 3)) + 1)))
            except (ValueError, TypeError, RuntimeError):
                pass
        if properties["collection_angle"] == "auto":
            collection_arg = (
                properties["NA"] / properties["refractive_index_medium"]
            )
            if config.get_backend() == "torch":
                collection_arg = _asarray(collection_arg, dtype=xp.float64)
            properties["collection_angle"] = xp.asin(collection_arg)

        if properties["offset_z"] == "auto":
            size = (
                np.array(properties["output_region"][2:])
                - properties["output_region"][:2]
            )
            xSize, ySize = size

            # offset_z should be calculated with the physical size of the image
            # not the fft-padded size
            min_edge_size = np.min([xSize, ySize])
            collection_angle = properties["collection_angle"]
            if config.get_backend() == "torch":
                collection_angle = _asarray(
                    collection_angle,
                    dtype=xp.float64,
                )
            voxel_size = properties.get("voxel_size")
            if voxel_size is None:
                voxel_size = get_active_voxel_size()
            properties["offset_z"] = (
                min_edge_size
                * 0.45
                * xp.min(_asarray(voxel_size, dtype=xp.float64)[:2])
                / xp.tan(collection_angle)
            )
        return properties

    def get_xy_size(
        self: MieScatterer,
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

    def get_xy_grid(
        self: MieScatterer, shape: tuple[int, int], voxel_size: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates meshgrid for X and Y given the shape and voxel size.

        Parameters
        ----------
        shape: tuple[int, int]
            The dimensions of the output region.

        voxel_size: array-like of float
            The size of each voxel in meters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The meshgrid of X and Y coordinates.

        """

        x = xp.arange(shape[0], dtype=xp.float64)
        y = xp.arange(shape[1], dtype=xp.float64)
        x = x - shape[0] / 2
        y = y - shape[1] / 2
        return xp.meshgrid(x * voxel_size[0], y * voxel_size[1], indexing="ij")

    def get_detector_mask(
        self: MieScatterer,
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

        return xp.sqrt(X**2 + Y**2) < radius

    def _plane_in_polar_coords_geometric(
        self: MieScatterer,
        shape: tuple[int, int],
        voxel_size: np.ndarray,
        plane_position: np.ndarray,
        illumination_angle: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the polar coordinates of virtual plane for geometric mode.

        In geometric mode, the virtual plane is defined in the spatial domain
        at a distance `offset_z` from the particle. The coordinates are
        calculated based on the plane position, voxel size, and illumination
        angle, and are used to compute the spherical wave representation of the
        scattered field on the virtual plane, which is then propagated through
        the optical system. The coordinates include the distance from the
        particle to each point on the plane (R3), the cosine of the angle
        between the illumination direction and the local normal at each point
        (cos_theta), the cosine of the angle between the illumination direction
        and the local normal adjusted by the illumination angle
        (illumination_cos_theta), and the azimuthal angle in the plane of the
        virtual field (phi).

        Parameters
        ----------
        shape: tuple[int, int]
            The dimensions of the output region.
        voxel_size: array-like of float
            The size of each voxel in meters.
        plane_position: array-like of float
            The position of the virtual plane in (x, y, z) coordinates.
        illumination_angle: float
            The angle of illumination in radians.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The polar coordinates (R3, cos_theta, illumination_cos_theta, phi)
            of the virtual plane.

        """

        X, Y = self.get_xy_grid(shape, voxel_size)

        X = X + plane_position[0]
        Y = Y + plane_position[1]
        Z = plane_position[2]

        R2_squared = X**2 + Y**2
        R3 = xp.sqrt(R2_squared + Z**2)

        cos_theta = Z / R3
        if float(illumination_angle) == 0:
            illumination_cos_theta = cos_theta
        else:
            illumination_cos_theta = xp.cos(
                xp.acos(cos_theta) + illumination_angle
            )
        phi = xp.atan2(Y, X)

        return R3, cos_theta, illumination_cos_theta, phi

    def _plane_in_polar_coords_hybrid(
        self: MieScatterer,
        shape: tuple[int, int],
        voxel_size: np.ndarray,
        plane_position: np.ndarray,
        illumination_angle: float,
        k: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the polar coordinates of virtual plane for hybrid mode.

        In hybrid mode, the virtual plane is defined in the spatial frequency
        domain corresponding to the objective pupil. The coordinates are
        calculated based on the plane position, voxel size, and illumination
        angle, and are used to map the Mie scattering amplitudes onto the
        pupil-frequency representation.

        Parameters
        ----------
        shape: tuple[int, int]
            The dimensions of the output region.
        voxel_size: array-like of float
            The size of each voxel in meters.
        plane_position: array-like of float
            The position of the virtual plane in (x, y, z) coordinates.
        illumination_angle: float
            The angle of illumination in radians.
        k: float
            The wavenumber of the illumination, calculated as
            2 * π / wavelength * refractive_index_medium.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The polar coordinates (R3, cos_theta, illumination_cos_theta, phi)
            of the virtual plane, and a boolean mask indicating which points
            are within the objective pupil.

        """

        X, Y = self.get_xy_grid(shape, voxel_size)

        X = X + plane_position[0]
        Y = Y + plane_position[1]
        Z = plane_position[2]

        R2_squared = X**2 + Y**2
        R3 = xp.sqrt(R2_squared + Z**2)

        Q = xp.sqrt(R2_squared) / voxel_size[0] ** 2 * 2 * np.pi / shape[0]
        sin_theta = Q / (k)
        pupil_mask = sin_theta < 1
        cos_theta = xp.sqrt(
            xp.maximum(xp.zeros_like(sin_theta), 1 - sin_theta**2)
        )
        cos_theta = xp.where(
            pupil_mask,
            cos_theta,
            xp.zeros_like(cos_theta),
        )

        if float(illumination_angle) == 0:
            illumination_cos_theta = cos_theta
        else:
            illumination_cos_theta = xp.cos(
                xp.acos(cos_theta) + illumination_angle
            )
        phi = xp.atan2(Y, X)

        return R3, cos_theta, illumination_cos_theta, phi, pupil_mask

    def _polarization_coefficients(
        self,
        phi: np.ndarray,
        illumination_cos_theta: np.ndarray,
        input_polarization: float | int | str | Quantity,
        output_polarization: float | int | Quantity,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the polarization coefficients for the scattered field.

        Parameters
        ----------
        phi: np.ndarray
            The azimuthal angle in the plane of the virtual field.
        illumination_cos_theta: np.ndarray
            The cosine of the angle between the illumination direction and the
            local normal at each point in the virtual field.
        input_polarization: float | int | str | Quantity
            The polarization state of the incident illumination. Can be a float
            representing the angle of linear polarization, the string
            "circular" for circular polarization, or a Quantity with angle
            units.
        output_polarization: float | int | Quantity
            The angle of the polarization analyzer for detection. Can be a
            float representing the angle of linear polarization, or a Quantity
            with angle units.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The coefficients S1_coef and S2_coef that weight the scattering
            amplitudes S1 and S2 based on the input and output polarization
            states.

        """

        if isinstance(input_polarization, Quantity):
            input_polarization = input_polarization.to("rad").magnitude

        if isinstance(input_polarization, str):
            if input_polarization != "circular":
                raise TypeError(
                    f"Unsupported input_polarization: {input_polarization}"
                )
            S1_coef = 1 / np.sqrt(2)
            S2_coef = 1j / np.sqrt(2)
        else:
            input_polarization = _asarray(
                input_polarization,
                dtype=xp.float64,
            )
            S1_coef = xp.sin(phi + input_polarization)
            S2_coef = xp.cos(phi + input_polarization)

        if isinstance(output_polarization, Quantity):
            output_polarization = output_polarization.to("rad").magnitude

        output_polarization = _asarray(
            output_polarization,
            dtype=xp.float64,
        )
        S1_coef = S1_coef * xp.sin(phi + output_polarization)
        S2_coef = (
            S2_coef
            * xp.cos(phi + output_polarization)
            * illumination_cos_theta
        )

        return S1_coef, S2_coef

    def _mie_scattering(
        self: MieScatterer,
        L: int,
        illumination_cos_theta: np.ndarray,
        coefficients: callable,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the Mie scattering amplitudes S1 and S2.

        Parameters
        ----------
        L: int
            The number of terms used to evaluate the Mie series.
        illumination_cos_theta: np.ndarray
            The cosine of the angle between the illumination direction and the
            local normal at each point in the virtual field.
        coefficients: callable
            Callable such that `coefficients(L)` returns the Mie coefficients
            `(an, bn)` up to order `L`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The scattering amplitudes S1 and S2.

        """

        A, B = coefficients(L)
        PI, TAU = mie.harmonics(illumination_cos_theta, L)

        E = [(2 * i + 1) / (i * (i + 1)) for i in range(1, L + 1)]

        S1 = sum(E[i] * A[i] * PI[i] + E[i] * B[i] * TAU[i] for i in range(L))
        S2 = sum(E[i] * B[i] * PI[i] + E[i] * A[i] * TAU[i] for i in range(L))

        return S1, S2

    def _common_setup(
        self: MieScatterer,
        position: tuple[float, float, float],
        voxel_size: np.ndarray,
        padding: tuple[int, int, int, int],
        output_region: tuple[int, int, int, int],
        wavelength: float,
        refractive_index_medium: float,
        collection_angle: float,
        z: float,
        working_distance: float,
        position_objective: tuple[float, float, float],
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray
    ]:
        """Performs common setup steps for both geometric and hybrid modes.

        This method computes the initial field array, voxel size, scaled
        position, pupil physical size, wavenumber, and relative position of the
        particle to the objective. These calculations are shared between the
        geometric and hybrid modes, so they are factored out into a common
        method to avoid code duplication.

        Parameters
        ----------
        position: tuple[float, float, float]
            The position of the particle in (x, y, z) coordinates.
        voxel_size: np.ndarray
            The physical voxel size in meters.
        padding: int
            The padding applied to the output region.
        output_region: tuple[int, int]
            The coordinates defining the output region.
        wavelength: float
            The wavelength of the illumination in meters.
        refractive_index_medium: float
            The refractive index of the medium surrounding the particle.
        collection_angle: float
            The maximum collection angle in radians.
        z: float
            The axial position of the particle relative to the camera plane.
        working_distance: float
            The working distance of the objective lens in meters.
        position_objective: tuple[float, float, float]
            The position of the objective lens in (x, y, z) coordinates.

        Returns
        -------
        tuple[array, array, array, float, float, float, array]
            A tuple containing the initialized field array, voxel size, scaled
            position, pupil physical size, wavenumber, and relative position of
            the particle to the objective.

        """

        xSize, ySize = self.get_xy_size(output_region, padding)
        voxel_size = _asarray(
            voxel_size,
            dtype=xp.float64,
        )
        scale = xp.asarray(
            get_active_scale(),
            dtype=xp.float64,
        )

        arr = pad_image_to_fft(
            xp.zeros((xSize, ySize), dtype=xp.complex128)
        )

        position = _asarray_vector(
            position,
            dtype=xp.float64,
        )
        position = (
            position
            * scale[: len(position)]
            * voxel_size[: len(position)]
        )
        wavelength = _asarray(wavelength, dtype=xp.float64)
        refractive_index_medium = _asarray(
            refractive_index_medium,
            dtype=xp.float64,
        )
        collection_angle = _asarray(
            collection_angle,
            dtype=xp.float64,
        )
        working_distance = _asarray(
            working_distance,
            dtype=xp.float64,
        )
        z = _asarray(z, dtype=xp.float64)
        z = z * voxel_size[2] * scale[2]
        position_objective = _asarray_vector(
            position_objective,
            dtype=xp.float64,
        )

        pupil_physical_size = working_distance * xp.tan(collection_angle) * 2
        k = 2 * np.pi / wavelength * refractive_index_medium

        relative_position = xp.stack(
            [
                position_objective[0] - position[0],
                position_objective[1] - position[1],
                working_distance - z,
            ]
        )

        return (
            arr,
            voxel_size,
            position,
            z,
            pupil_physical_size,
            k,
            relative_position,
        )

    def get(
        self: MieScatterer,
        *args,
        mode=None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate the Mie scatterer field based on the specified mode.

        This method dispatches the field calculation to either the geometric or
        hybrid implementation based on the `mode` argument. If `mode` is not
        provided, it defaults to the mode specified during initialization.

        Parameters
        ----------
        args: Any
            Positional arguments passed to the method.
        mode: str | None
            The mode to use for field calculation. Can be "geometric" or
            "hybrid". If None, the mode specified during initialization is
            used.
        kwargs: Any
            Keyword arguments passed to the method.

        Returns
        -------
        np.ndarray
            The calculated scattered field based on the specified mode.

        """

        mode = self.mode if mode is None else mode

        if mode == "geometric":
            return self._solve_geometric(*args, **kwargs)
        if mode == "hybrid":
            return self._solve_hybrid(*args, **kwargs)
        if mode == "fourier":
            raise NotImplementedError("Pure Fourier mode not implemented yet.")

        raise ValueError(f"Unknown mode: {mode}")

    def _solve_geometric(
        self: MieScatterer,
        inp: Any,
        position: np.ndarray,
        voxel_size: np.ndarray,
        padding: tuple[int, int, int, int],
        wavelength: float,
        refractive_index_medium: float,
        L: int,
        collection_angle: float,
        input_polarization: float | int | str | Quantity,
        output_polarization: float | int | Quantity,
        coefficients: Any,
        offset_z: float,
        z: float,
        working_distance: float,
        position_objective: tuple[float, float],
        return_fft: bool,
        coherence_length: float,
        output_region: tuple[int, int, int, int],
        illumination_angle: float,
        amp_factor: float,
        phase_shift_correction: bool,
        pupil: np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Calculates the scattered field using the geometric mode.

        In geometric mode, the scattered field is evaluated as a spherical wave
        on a virtual plane located at a distance `offset_z` from the particle.
        The field includes the geometric propagation factor `exp(i k R) / R`
        and is sampled on a finite spatial grid before being propagated through
        the optical system. The coordinates of the virtual plane are calculated
        based on the plane position, voxel size, and illumination angle, and
        are used to compute the spherical wave representation of the scattered
        field on the virtual plane, which is then propagated through the
        optical system.

        Parameters
        ----------
        inp: Any
            The input to the method, which can be used for additional
            processing if needed.
        position: np.ndarray
            The position of the particle in (x, y, z) coordinates.
        voxel_size: np.ndarray
            The size of each voxel in meters.
        padding: int
            The padding applied to the output region.
        wavelength: float
            The wavelength of the illumination in meters.
        refractive_index_medium: float
            The refractive index of the medium surrounding the particle.
        L: float
            The number of terms used to evaluate the Mie series.
        collection_angle: float
            The maximum collection angle in radians.
        input_polarization: np.ndarray
            The polarization state of the incident illumination.
        output_polarization: np.ndarray
            The angle of the polarization analyzer for detection.
        coefficients: np.ndarray
            The Mie coefficients used to calculate the scattering amplitudes
            S1 and S2.
        offset_z: float
            The distance from the particle in the z direction where the field
            is evaluated.
        z: float
            The axial position of the particle relative to the camera plane.
        working_distance: float
            The working distance of the objective lens in meters.
        position_objective: np.ndarray
            The position of the objective lens in (x, y, z) coordinates.
        return_fft: bool
            If True, the method returns the Fourier transform of the field
            rather than the spatial field itself.
        coherence_length: float
            The temporal coherence length of the illumination in meters. If
            None, illumination is assumed to be fully coherent.
        output_region: tuple[int, int]
            The coordinates defining the output region.
        illumination_angle: float
            The angle of illumination in radians.
        amp_factor: float
            The scaling factor applied to the scattered field amplitude.
        phase_shift_correction: bool
            If True, applies a phase correction to the field according to
            arr *= exp(1j * k * z + 1j * π / 2). This correction is used in
            ISCAT simulations.
        pupil: np.ndarray | None
            Optional pupil function applied to the scattered field. This can be
            used to simulate aberrations or other modifications of the optical
            system.

        Returns
        -------
        np.ndarray
            The calculated scattered field based on the geometric mode.

        """

        (
            arr,
            voxel_size,
            position,
            z,
            pupil_physical_size,
            k,
            relative_position,
        ) = self._common_setup(
            position,
            voxel_size,
            padding,
            output_region,
            wavelength,
            refractive_index_medium,
            collection_angle,
            z,
            working_distance,
            position_objective,
        )

        ratio = offset_z / (working_distance - z)

        R3_field, cos_theta_field, illumination_angle_field, phi_field = (
            self._plane_in_polar_coords_geometric(
                arr.shape,
                voxel_size,
                relative_position * ratio,
                illumination_angle,
            )
        )

        cos_phi_field = xp.cos(phi_field)
        sin_phi_field = xp.sin(phi_field)

        x_farfield = (
            position[0]
            + R3_field
            * xp.sqrt(1 - cos_theta_field**2)
            * cos_phi_field
            / ratio
        )
        y_farfield = (
            position[1]
            + R3_field
            * xp.sqrt(1 - cos_theta_field**2)
            * sin_phi_field
            / ratio
        )

        pupil_mask = (x_farfield - position_objective[0]) ** 2 + (
            y_farfield - position_objective[1]
        ) ** 2 < (pupil_physical_size / 2) ** 2
        cos_theta_field = cos_theta_field[pupil_mask]

        R3_field = R3_field[pupil_mask]
        phi_field = phi_field[pupil_mask]
        illumination_angle_field = illumination_angle_field[pupil_mask]

        S1_coef, S2_coef = self._polarization_coefficients(
            phi_field,
            illumination_angle_field,
            input_polarization,
            output_polarization,
        )
        S1, S2 = self._mie_scattering(
            L, illumination_angle_field, coefficients
        )

        scattered_values = (
            -1j
            / (k * R3_field)
            * xp.exp(1j * k * R3_field)
            * (S2 * S2_coef + S1 * S1_coef)
        ) / amp_factor

        if TORCH_AVAILABLE and torch.is_tensor(arr):
            flat_values = torch.zeros(
                arr.numel(), dtype=arr.dtype, device=arr.device
            )
            flat_values[pupil_mask.reshape(-1)] = scattered_values
            arr = arr + flat_values.reshape(arr.shape)
        else:
            arr[pupil_mask] = scattered_values

        # For phase shift correction (a multiplication of the field
        # by exp(1j * k * z)).
        if phase_shift_correction:
            arr = arr * xp.exp(1j * k * z + 1j * np.pi / 2)

        # For partially coherent illumination.
        if coherence_length:
            sigma = z * xp.sqrt((coherence_length / z + 1) ** 2 - 1)
            sigma = sigma * (offset_z / z)

            y = xp.arange(arr.shape[0], dtype=xp.float64)
            x = xp.arange(arr.shape[1], dtype=xp.float64)
            y = y - arr.shape[0] // 2
            x = x - arr.shape[1] // 2
            y, x = xp.meshgrid(y, x, indexing="ij")
            mask = xp.exp(-0.5 * (x**2 + y**2) / ((sigma) ** 2))
            arr = arr * mask

        fourier_field = xp.fft.fft2(arr)

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

        fourier_field = (
            fourier_field
            * propagation_matrix
            * xp.exp(-1j * k * offset_z)
        )

        if return_fft:
            return fourier_field[..., None]
        return xp.fft.ifft2(fourier_field)[..., None]

    def _solve_hybrid(
        self: MieScatterer,
        inp: Any,
        position: np.ndarray,
        voxel_size: np.ndarray,
        padding: tuple[int, int, int, int],
        wavelength: float,
        refractive_index_medium: float,
        L: int,
        collection_angle: float,
        input_polarization: float | int | str | Quantity,
        output_polarization: float | int | Quantity,
        coefficients: Any,
        offset_z: float,
        z: float,
        working_distance: float,
        position_objective: tuple[float, float],
        return_fft: bool,
        coherence_length: float,
        output_region: tuple[int, int, int, int],
        illumination_angle: float,
        amp_factor: float,
        phase_shift_correction: bool,
        pupil=None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Calculates the scattered field using the hybrid mode.

        In hybrid mode, the scattered field is constructed using the Mie
        scattering amplitudes S1 and S2 mapped to spatial frequencies
        corresponding to the objective pupil. The field is then propagated to
        the detector. This approach is less sensitive to the simulated
        field-of-view and generally more numerically stable compared to the
        geometric mode, which evaluates the scattered field as a spherical
        wave on a virtual plane.

        Parameters
        ----------
        inp: Any
            The input to the method, which can be used for additional
            processing if needed.
        position: np.ndarray
            The position of the particle in (x, y, z) coordinates.
        voxel_size: np.ndarray
            The size of each voxel in meters.
        padding: np.ndarray
            The padding applied to the output region.
        wavelength: float
            The wavelength of the illumination in meters.
        refractive_index_medium: float
            The refractive index of the medium surrounding the particle.
        L: float
            The number of terms used to evaluate the Mie series.
        collection_angle: float
            The maximum collection angle in radians.
        input_polarization: np.ndarray
            The polarization state of the incident illumination.
        output_polarization: np.ndarray
            The angle of the polarization analyzer for detection.
        coefficients: np.ndarray
            The Mie coefficients used to calculate the scattering amplitudes
            S1 and S2.
        offset_z: float
            The distance from the particle in the z direction where the field
            is evaluated.
        z: float
            The axial position of the particle relative to the camera plane.
        working_distance: float
            The working distance of the objective lens in meters.
        position_objective: np.ndarray
            The position of the objective lens in (x, y, z) coordinates.
        return_fft: bool
            If True, the method returns the Fourier transform of the field
            rather than the spatial field itself.
        coherence_length: float
            The temporal coherence length of the illumination in meters. If
            None, illumination is assumed to be fully coherent.
        output_region: tuple[int, int]
            The coordinates defining the output region.
        illumination_angle: float
            The angle of illumination in radians.
        amp_factor: float
            The scaling factor applied to the scattered field amplitude.
        phase_shift_correction: bool
            If True, applies a phase correction to the field according to
            arr *= exp(1j * k * z + 1j * π / 2). This correction is used in
            ISCAT simulations.
        pupil: np.ndarray | None
            Optional pupil function applied to the scattered field. This can be
            used to simulate aberrations or other modifications of the optical
            system.

        Returns
        -------
        np.ndarray
            The calculated scattered field based on the hybrid mode.

        """

        (
            arr,
            voxel_size,
            position,
            z,
            pupil_physical_size,
            k,
            relative_position,
        ) = self._common_setup(
            position,
            voxel_size,
            padding,
            output_region,
            wavelength,
            refractive_index_medium,
            collection_angle,
            z,
            working_distance,
            position_objective,
        )

        ratio = offset_z / (working_distance - z)

        (
            R3_field,
            cos_theta_field,
            illumination_angle_field,
            phi_field,
            pupil_mask,
        ) = self._plane_in_polar_coords_hybrid(
            arr.shape,
            voxel_size,
            relative_position * ratio,
            illumination_angle,
            k,
        )

        cos_phi_field = xp.cos(phi_field)
        sin_phi_field = xp.sin(phi_field)

        x_farfield = (
            position[0]
            + R3_field
            * xp.sqrt(1 - cos_theta_field**2)
            * cos_phi_field
            / ratio
        )
        y_farfield = (
            position[1]
            + R3_field
            * xp.sqrt(1 - cos_theta_field**2)
            * sin_phi_field
            / ratio
        )

        phi_valid = phi_field[pupil_mask]
        illum_valid = illumination_angle_field[pupil_mask]

        S1_coef, S2_coef = self._polarization_coefficients(
            phi_valid, illum_valid, input_polarization, output_polarization
        )
        S1, S2 = self._mie_scattering(L, illum_valid, coefficients)

        scattered_values = (S2 * S2_coef + S1 * S1_coef) / amp_factor

        if TORCH_AVAILABLE and torch.is_tensor(arr):
            flat_values = torch.zeros(
                arr.numel(), dtype=arr.dtype, device=arr.device
            )
            flat_values[pupil_mask.reshape(-1)] = scattered_values
            arr = arr + flat_values.reshape(arr.shape)
        else:
            arr[pupil_mask] = scattered_values

        # For phase shift correction (a multiplication of the field
        # by exp(1j * k * z)).
        if phase_shift_correction:
            arr = arr * xp.exp(1j * k * z + 1j * np.pi / 2)

        # For partially coherent illumination.
        if coherence_length:
            sigma = z * xp.sqrt((coherence_length / z + 1) ** 2 - 1)
            sigma = sigma * (offset_z / z)

            y = xp.arange(arr.shape[0], dtype=xp.float64)
            x = xp.arange(arr.shape[1], dtype=xp.float64)
            y = y - arr.shape[0] // 2
            x = x - arr.shape[1] // 2
            y, x = xp.meshgrid(y, x, indexing="ij")
            mask = xp.exp(-0.5 * (x**2 + y**2) / ((sigma) ** 2))
            arr = arr * mask

        if pupil is not None and len(pupil) > 0:
            c0 = arr.shape[0] // 2
            c1 = arr.shape[1] // 2
            h0 = pupil.shape[0] // 2
            h1 = pupil.shape[1] // 2
            pupil_mask = xp.ones_like(arr)
            pupil_mask[c0 - h0 : c0 + h0, c1 - h1 : c1 + h1] = _asarray(
                pupil,
                dtype=arr.dtype,
            )
            arr = arr * pupil_mask

        fourier_field = xp.fft.ifft2(
            xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(arr)))
        )

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

        fourier_field = fourier_field * propagation_matrix

        if return_fft:
            return fourier_field[..., None]
        return xp.fft.ifft2(fourier_field)[..., None]


class MieSphere(MieScatterer):
    """Scattered field produced by a homogeneous sphere.

    This class computes the coherent scattered field of a spherical particle in
    a homogeneous medium using Mie theory.

    In `"geometric"` mode, accurate results typically require a sufficiently
    large simulation grid (often at least 64 × 64) and adequate padding,
    because the scattered field is sampled on a finite virtual plane before
    propagation. In contrast, the `"hybrid"` mode is generally less sensitive
    to grid size and field-of-view.

    The induced phase shift is defined relative to the
    `refractive_index_medium` of the optical configuration.


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
    output_polarization: float | Quantity
        Defines the angle of the polarization filter after the sample. For
        off-axis, keep the same as input_polarization.

    """

    def __init__(
        self: MieSphere,
        radius: float = 1e-6,
        refractive_index: float = 1.45,
        **kwargs,
    ):
        """Initializes the MieSphere feature.

        Parameters
        ----------
        radius: float
            Radius of the mie particle in meter.
        refractive_index: float
            Refractive index of the particle.
        **kwargs: Any
            Additional keyword arguments passed to the parent class initializer.

        """

        def coeffs(
            radius: float,
            refractive_index: float,
            refractive_index_medium: float,
            wavelength: float,
        ) -> callable:
            """Calculates the Mie coefficients for a homogeneous sphere.

             This function computes the Mie coefficients an and bn for a
             homogeneous sphere based on the provided radius, refractive index,
             and wavelength. The coefficients are calculated using the
             `mie.coefficients` function, which implements the standard Mie
             theory formulas for a homogeneous sphere.

             Parameters
             ----------
             radius: float
                 The radius of the sphere in meters.
             refractive_index: float
                 The refractive index of the sphere.
             refractive_index_medium: float
                 The refractive index of the surrounding medium.
             wavelength: float
                 The wavelength of the illumination in meters.

            Returns
            -------
            callable
                A function that computes the Mie coefficients for a given
                number of terms.

            """

            if isinstance(radius, Quantity):
                radius = radius.to("m").magnitude
            if isinstance(wavelength, Quantity):
                wavelength = wavelength.to("m").magnitude

            def inner(L: int):
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


class MieStratifiedSphere(MieScatterer):
    """Scattered field produced by a stratified sphere.

    In `"geometric"` mode, accurate results typically require a sufficiently
    large simulation grid (often at least 64 × 64) and adequate padding,
    because the scattered field is sampled on a finite virtual plane before
    propagation. In contrast, the `"hybrid"` mode is generally less sensitive
    to grid size and field-of-view.

    The induced phase shift is defined relative to the
    `refractive_index_medium` of the optical configuration.

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
    output_polarization: float | Quantity
        Defines the angle of the polarization filter after the sample. For
        off-axis, keep the same as input_polarization.

    """

    def __init__(
        self: MieStratifiedSphere,
        radius: tuple[float, ...] = (1e-6,),
        refractive_index: tuple[float, ...] = (1.45,),
        **kwargs: Any,
    ) -> None:
        """Initializes the MieStratifiedSphere feature.

        Parameters
        ----------
        radius: tuple[float, ...]
            The radius of each cell in increasing order.
        refractive_index: tuple[float, ...]
            Refractive index of each cell in the same order as `radius`.
        **kwargs: Any
            Additional keyword arguments passed to the parent class
            initializer.

        """

        def coeffs(
            radius: tuple[float, ...] | np.ndarray,
            refractive_index: tuple[float | complex, ...] | np.ndarray,
            refractive_index_medium: float,
            wavelength: float | Quantity,
        ) -> callable:
            """Calculates the Mie coefficients for a stratified sphere.

            This function computes the Mie coefficients an and bn for a
            stratified sphere based on the provided radius, refractive index,
            and wavelength. The coefficients are calculated using the
            `mie.stratified_coefficients` function, which implements the Mie
            theory formulas for a sphere composed of multiple concentric layers
            with different refractive indices. The `radius` parameter specifies
            the radius of each layer, and the `refractive_index` parameter
            specifies the refractive index of each layer. The function returns
            a callable that computes the Mie coefficients for a given number of
            terms.

            Parameters
            ----------
            radius: tuple[float, ...] | np.ndarray
                The radius of each cell in increasing order.
            refractive_index: tuple[float | complex, ...] | np.ndarray
                Refractive index of each cell in the same order as `radius`.
            refractive_index_medium: float
                The refractive index of the surrounding medium.
            wavelength: float | Quantity
                The wavelength of the illumination in meters.

            Returns
            -------
            callable
                A function that computes the Mie coefficients for a given
                number of terms.

            """

            radius_for_check = radius
            if TORCH_AVAILABLE and torch.is_tensor(radius_for_check):
                radius_for_check = radius_for_check.detach().cpu().numpy()
            elif isinstance(radius_for_check, (list, tuple)):
                radius_for_check = [
                    item.detach().cpu().numpy()
                    if TORCH_AVAILABLE and torch.is_tensor(item)
                    else item
                    for item in radius_for_check
                ]

            if not np.all(
                np.asarray(radius_for_check)[1:]
                >= np.asarray(radius_for_check)[:-1]
            ):
                raise ValueError(
                    "Radius of the shells of a stratified sphere should be "
                    "monotonically increasing."
                )

            def inner(L: int):
                return mie.stratified_coefficients(
                    _asarray_vector(
                        refractive_index,
                    )
                    / refractive_index_medium,
                    _asarray_vector(
                        radius,
                    )
                    * 2
                    * np.pi
                    / wavelength
                    * refractive_index_medium,
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
    """Voxelized volume produced by a `VolumeScatterer`.

    Provides convenience accessors for the lateral position (`position`) and
    full 3D position (`pos3d`) stored in the feature properties.

    """

    @property
    def pos3d(self: ScatteredVolume) -> np.ndarray | None:
        if self.position is None:
            return None
        return np.array([*self.position, self.z], dtype=float)

    @property
    def position(self: ScatteredVolume) -> np.ndarray | None:
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
    configurations and returns the incoherent (intensity) average. If both
    `input_unpolarized` and `output_unpolarized` are False, the wrapper acts
    as a pass-through and returns the child feature unchanged.

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
            will be evaluated for two orthogonal input polarization states (0
            and π/2).
        output_unpolarized: bool, optional
            If True, the output light is treated as unpolarized, and the
            feature will be evaluated for two orthogonal output polarization
            states (0 and π/2).
        **kwargs: dict
            Additional keyword arguments passed to the parent
            StructuralFeature.

        """

        super().__init__(
            input_unpolarized=input_unpolarized,
            output_unpolarized=output_unpolarized,
            **kwargs,
        )
        self.feature = self.add_feature(feature)

    @staticmethod
    def _states(
        base: float | None,
        unpolarized: bool,
    ) -> tuple[float, ...]:
        """Return polarization states to sample.

        For unpolarized light, two orthogonal linear polarization states
        (0 and π/2) are used. Otherwise, the provided base state is returned,
        defaulting to 0 if `base` is None.

        """

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
        """Incoherently average the feature over polarization states.

        Evaluates the feature for different polarization states and returns
        the incoherent average. If both `input_unpolarized` and
        `output_unpolarized` are False, the feature is evaluated once with the
        provided polarization states (or defaults) and returned directly.

        Parameters
        ----------
        inputs: Any
            The input to the feature, passed through to the child feature.
        input_unpolarized: bool
            Whether the input light is unpolarized.
        output_unpolarized: bool
            Whether the output light is unpolarized.
        _ID: tuple, optional
            The identifier for the current feature evaluation, passed through
            to the child feature.
        **kwargs: dict
            Additional keyword arguments passed to the child feature.

        Returns
        -------
        Any
            The incoherent average of the feature evaluated over the specified
            polarization states.

        """

        # Fast path: no averaging needed
        if not input_unpolarized and not output_unpolarized:
            return self.feature(_ID=_ID, **kwargs)

        base_input = kwargs.get("input_polarization", 0.0)
        base_output = kwargs.get("output_polarization", 0.0)

        input_states = self._states(base_input, input_unpolarized)
        output_states = self._states(base_output, output_unpolarized)

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
                field = result.array if hasattr(result, "array") else result
                I = np.abs(field) ** 2

                if intensity_sum is None:
                    intensity_sum = np.array(I, copy=True)
                else:
                    intensity_sum += I

                count += 1

        return intensity_sum / count
