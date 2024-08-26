"""Implementations of Feature the model scattering objects.

Provides some basic implementations of scattering objects
that are frequently used.

Classes
--------
Scatterer
    Abstract base class for scatterers
PointParticle
    Generates point particles
Ellipse
    Generetes 2-d elliptical particles
Sphere
    Generates 3-d spheres
Ellipsoid
    Generates 3-d ellipsoids
"""


from pint import Quantity

from deeptrack.holography import get_propagation_matrix
from . import image
from deeptrack.backend.units import (
    ConversionTable,
    get_active_scale,
    get_active_voxel_size,
)
from typing import Callable, Tuple

import numpy as np

from . import backend as D
from .features import Feature, MERGE_STRATEGY_APPEND
from . import pad_image_to_fft, Image
from .types import PropertyLike, ArrayLike
from . import units as u
import warnings


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

    Parameters
    ----------
    position : array_like of length 2 or 3
        The position of the  particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    value : float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
    position_unit : "meter" or "pixel"
        The unit of the provided position property.

    Other Parameters
    ----------------
    upsample_axes : tuple of ints
        Sets the axes along which the calculation is upsampled (default is
        None, which implies all axes are upsampled).
    crop_zeros : bool
        Whether to remove slices in which all elements are zero.
    """

    __list_merge_strategy__ = MERGE_STRATEGY_APPEND
    __distributed__ = False
    __conversion_table__ = ConversionTable(
        position=(u.pixel, u.pixel),
        z=(u.zpixel, u.zpixel),
        voxel_size=(u.meter, u.meter),
    )

    def __init__(
        self,
        position: PropertyLike[ArrayLike[float]] = (32, 32),
        z: PropertyLike[float] = 0.0,
        value: PropertyLike[float] = 1.0,
        position_unit: PropertyLike[str] = "pixel",
        upsample: PropertyLike[int] = 1,
        voxel_size=None,
        pixel_size=None,
        **kwargs,
    ):
        # Ignore warning to help with comparison with arrays.
        if upsample is not 1:  # noqa: F632
            warnings.warn(
                f"Setting upsample != 1 is deprecated. Please, instead use dt.Upscale(f, factor={upsample})"
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

    def _process_properties(self, properties: dict) -> dict:
        # Rescales the position property
        properties = super()._process_properties(properties)
        self._processed_properties = True
        return properties

    def _process_and_get(
        self, *args, voxel_size, upsample, upsample_axes=None, crop_empty=True, **kwargs
    ):
        # Post processes the created object to handle upsampling,
        # as well as cropping empty slices.
        if not self._processed_properties:

            warnings.warn(
                "Overridden _process_properties method does not call super. "
                + "This is likely to result in errors if used with "
                + "Optics.upscale != 1."
            )

        voxel_size = get_active_voxel_size()

        # calls parent _process_and_get
        new_image = super()._process_and_get(
            *args,
            voxel_size=voxel_size,
            upsample=upsample,
            **kwargs,
        )
        new_image = new_image[0]

        if new_image.size == 0:
            warnings.warn(
                "Scatterer created that is smaller than a pixel. "
                + "This may yield inconsistent results."
                + " Consider using upsample on the scatterer,"
                + " or upscale on the optics.",
                Warning,
            )

        # Crops empty slices
        if crop_empty:
            new_image = new_image[~np.all(new_image == 0, axis=(1, 2))]
            new_image = new_image[:, ~np.all(new_image == 0, axis=(0, 2))]
            new_image = new_image[:, :, ~np.all(new_image == 0, axis=(0, 1))]

        return [Image(new_image)]

    def _no_wrap_format_input(self, *args, **kwargs) -> list:
        return self._image_wrapped_format_input(*args, **kwargs)
    
    def _no_wrap_process_and_get(self, *args, **feature_input) -> list:
        return self._image_wrapped_process_and_get(*args, **feature_input)
    
    def _no_wrap_process_output(self, *args, **feature_input):
        return self._image_wrapped_process_output(*args, **feature_input)

class PointParticle(Scatterer):
    """Generates a point particle

    A point particle is approximated by the size of a pixel. For subpixel
    positioning, the position is interpolated linearly.

    Parameters
    ----------
    position : array_like of length 2 or 3
        The position of the  particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    value : float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
    """

    def __init__(self, **kwargs):
        super().__init__(upsample=1, upsample_axes=(), **kwargs)

    def get(self, image, **kwargs):
        scale = get_active_scale()
        return np.ones((1, 1, 1)) * np.prod(scale)


class Ellipse(Scatterer):
    """Generates an elliptical disk scatterer

    Parameters
    ----------
    radius : float or array_like [float (, float)]
        Radius of the ellipse in meters. If only one value,
        assume circular.
    rotation : float
        Orientation angle of the ellipse in the camera plane in radians.
    position : array_like[float, float (, float)]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    value : float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
    upsample : int
        Upsamples the calculations of the pixel occupancy fraction.
    transpose : bool
        If True, the ellipse is transposed as to align the first axis of the radius with
        the first axis of the created volume. This is applied before rotation.

    """

    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        radius: PropertyLike[float] = 1e-6,
        rotation: PropertyLike[float] = 0,
        transpose: PropertyLike[bool] = False,
        **kwargs,
    ):
        super().__init__(
            radius=radius, rotation=rotation, transpose=transpose, **kwargs
        )

    def _process_properties(self, properties: dict) -> dict:
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

    def get(self, *ignore, radius, rotation, voxel_size, transpose, **kwargs):

        if not transpose:
            radius = radius[::-1]
            # rotation = rotation[::-1]
        # Create a grid to calculate on
        rad = radius[:2]
        ceil = int(np.ceil(np.max(rad) / np.min(voxel_size[:2])))
        Y, X = np.meshgrid(
            np.arange(-ceil, ceil) * voxel_size[1],
            np.arange(-ceil, ceil) * voxel_size[0],
        )

        # Rotate the grid
        if rotation != 0:
            Xt = X * np.cos(-rotation) + Y * np.sin(-rotation)
            Yt = -X * np.sin(-rotation) + Y * np.cos(-rotation)
            X = Xt
            Y = Yt

        # Evaluate ellipse
        mask = ((X * X) / (rad[0] * rad[0]) + (Y * Y) / (rad[1] * rad[1]) < 1) * 1.0
        mask = np.expand_dims(mask, axis=-1)
        return mask


class Sphere(Scatterer):
    """Generates a spherical scatterer

    Parameters
    ----------
    radius : float
        Radius of the sphere in meters.
    position : array_like[float, float (, float)]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    value : float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
    upsample : int
        Upsamples the calculations of the pixel occupancy fraction.
    """

    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
    )

    def __init__(self, radius: PropertyLike[float] = 1e-6, **kwargs):
        super().__init__(radius=radius, **kwargs)

    def get(self, image, radius, voxel_size, **kwargs):

        # Create a grid to calculate on
        rad = radius * np.ones(3) / voxel_size
        rad_ceil = np.ceil(rad)
        x = np.arange(-rad_ceil[0], rad_ceil[0])
        y = np.arange(-rad_ceil[1], rad_ceil[1])
        z = np.arange(-rad_ceil[2], rad_ceil[2])
        X, Y, Z = np.meshgrid((y / rad[1]) ** 2, (x / rad[0]) ** 2, (z / rad[2]) ** 2)

        mask = (X + Y + Z <= 1) * 1.0
        return mask


class Ellipsoid(Scatterer):
    """Generates an ellipsoidal scatterer

    Parameters
    ----------
    radius : float or array_like[float (, float, float)]
        Radius of the ellipsoid in meters. If only one value,
        assume spherical.
    rotation : float
        Rotation of the ellipsoid in about the x, y and z axis.
    position : array_like[float, float (, float)]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    value : float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
    upsample : int
        Upsamples the calculations of the pixel occupancy fraction.
    transpose : bool
        If True, the ellipse is transposed as to align the first axis of the radius with
        the first axis of the created volume. This is applied before rotation.
    """

    __conversion_table__ = ConversionTable(
        radius=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        radius: PropertyLike[float] = 1e-6,
        rotation: PropertyLike[float] = 0,
        transpose: PropertyLike[bool] = False,
        **kwargs,
    ):
        super().__init__(
            radius=radius, rotation=rotation, transpose=transpose, **kwargs
        )

    def _process_properties(self, propertydict):
        """Preprocess the input to the method .get()

        Ensures that the radius and the rotation properties both are arrays of
        length 3.

        If the radius is a single value, the particle is made a sphere
        If the radius are two values, the smallest value is appended as the
        third value

        The rotation vector is padded with zeros until it is of length 3
        """

        propertydict = super()._process_properties(propertydict)

        # Ensure radius has three values
        radius = np.array(propertydict["radius"])
        if radius.ndim == 0:
            radius = np.array([radius])
        if radius.size == 1:
            # If only one value, assume sphere
            radius = (*radius,) * 3
        elif radius.size == 2:
            # If two values, duplicate the minor axis
            radius = (*radius, np.min(radius[-1]))
        elif radius.size == 3:
            # If three values, convert to tuple for consistency
            radius = (*radius,)
        propertydict["radius"] = radius

        # Ensure rotation has three values
        rotation = np.array(propertydict["rotation"])
        if rotation.ndim == 0:
            rotation = np.array([rotation])
        if rotation.size == 1:
            # If only one value, pad with two zeros
            rotation = (*rotation, 0, 0)
        elif rotation.size == 2:
            # If two values, pad with one zero
            rotation = (*rotation, 0)
        elif rotation.size == 3:
            # If three values, convert to tuple for consistency
            rotation = (*rotation,)
        propertydict["rotation"] = rotation

        return propertydict

    def get(self, image, radius, rotation, voxel_size, transpose, **kwargs):
        if not transpose:
            # swap the first and second value of the radius vector
            radius = (radius[1], radius[0], radius[2])

        # radius_in_pixels = np.array(radius) / np.array(voxel_size)

        # max_rad = np.max(radius_in_pixels)
        rad_ceil = np.ceil(np.max(radius) / np.min(voxel_size))

        # Create grid to calculate on
        x = np.arange(-rad_ceil, rad_ceil) * voxel_size[0]
        y = np.arange(-rad_ceil, rad_ceil) * voxel_size[1]
        z = np.arange(-rad_ceil, rad_ceil) * voxel_size[2]
        Y, X, Z = np.meshgrid(y, x, z)

        # Rotate the grid
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
            (XR / radius[0]) ** 2 + (YR / radius[1]) ** 2 + (ZR / radius[2]) ** 2 < 1
        ) * 1.0
        return mask


class MieScatterer(Scatterer):
    """Base implementation of a Mie particle.

    New Mie-theory scatterers can be implemented by extending this class, and
    passing a function that calculates the coefficients of the harmonics up to
    order `L`. To beprecise, the feature expects a wrapper function that takes
    the current values of the properties, as well as a inner function that
    takes an integer as the only parameter, and calculates the coefficients up
    to that integer. The return format is expected to be a tuple with two
    values, corresponding to `an` and `bn`. See
    `deeptrack.backend.mie_coefficients` for an example.

    Parameters
    ----------
    coefficients : Callable[int] -> Tuple[ndarray, ndarray]
        Function that returns the harmonics coefficients.
    offset_z : "auto" or float
        Distance from the particle in the z direction the field is evaluated.
        If "auto", this is calculated from the pixel size and
        `collection_angle`
    collection_angle : "auto" or float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective is
        the limiting aperature).
    input_polarization: float or Quantity
        Defines the polarization angle of the input. For simulating circularly
        polarized light we recommend a coherent sum of two simulated fields. For
        unpolarized light we recommend a incoherent sum of two simulated fields. 
        If defined as "circular", the coefficients are set to 1/2.
    output_polarization: float or Quantity or None
        If None, the output light is not polarized. Otherwise defines the angle of the
        polarization filter after the sample. For off-axis, keep the same as input_polarization.
        If defined as "circular", the coefficients are multiplied by 1. I.e. no change.
    L : int or str
        The number of terms used to evaluate the mie theory. If `"auto"`,
        it determines the number of terms automatically.
    position : array_like[float, float (, float)]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    return_fft : bool
        If True, the feature returns the fft of the field, rather than the
        field itself.
    coherence_length : float
        The temporal coherence length of a partially coherent light given in meters. 
        If None, the illumination is assumed to be coherent.
    amp_factor : float
        A factor that scales the amplification of the field. 
        This is useful for scaling the field to the correct intensity. Default is 1.
    phase_shift_correction : bool
        If True, the feature applies a phase shift correction to the output field. 
        This is necessary for ISCAT simulations. 
        The correction depends on the k-vector and z according to the formula: 
        arr*=np.exp(1j * k * z + 1j * np.pi / 2)
    """

    __gpu_compatible__ = True

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
        coefficients: Callable[..., Callable[[int], Tuple[ArrayLike, ArrayLike]]],
        input_polarization=0,
        output_polarization=0,
        offset_z: PropertyLike[str] = "auto",
        collection_angle: PropertyLike[str] = "auto",
        L: PropertyLike[str] = "auto",
        refractive_index_medium=None,
        wavelength=None,
        NA=None,
        padding=(0,) * 4,
        output_region=None,
        polarization_angle=None,
        working_distance=1000000,  # large value to avoid numerical issues unless the user specifies a smaller value
        position_objective=(0, 0),
        return_fft=False,
        coherence_length=None,
        illumination_angle=0,
        amp_factor=1,
        phase_shift_correction=False,
        **kwargs,
    ):
        if polarization_angle is not None:
            warnings.warn(
                "polarization_angle is deprecated. Please use input_polarization instead"
            )
            input_polarization = polarization_angle
        kwargs.pop("is_field", None)
        kwargs.pop("crop_empty", None)

        super().__init__(
            is_field=True,
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
            **kwargs,
        )

    def _process_properties(self, properties):

        properties = super()._process_properties(properties)

        if properties["L"] == "auto":
            try:
                v = 2 * np.pi * np.max(properties["radius"]) / properties["wavelength"]
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

    def get_xy_size(self, output_region, padding):
        return (
            output_region[2] - output_region[0] + padding[0] + padding[2],
            output_region[3] - output_region[1] + padding[1] + padding[3],
        )

    def get_XY(self, shape, voxel_size):
        x = np.arange(shape[0]) - shape[0] / 2
        y = np.arange(shape[1]) - shape[1] / 2
        return np.meshgrid(x * voxel_size[0], y * voxel_size[1], indexing="ij")

    def get_detector_mask(self, X, Y, radius):
        return np.sqrt(X**2 + Y**2) < radius

    def get_plane_in_polar_coords(self, shape, voxel_size, plane_position, illumination_angle):

        X, Y = self.get_XY(shape, voxel_size)
        X = image.maybe_cupy(X)
        Y = image.maybe_cupy(Y)

        # the X, Y coordinates of the pupil relative to the particle
        X = X + plane_position[0]
        Y = Y + plane_position[1]
        Z = plane_position[2]  # might be +z or -z

        R2_squared = X**2 + Y**2
        R3 = np.sqrt(R2_squared + Z**2)  # might be +z instead of -z

        # get the angles
        cos_theta = Z / R3
        illumination_cos_theta=np.cos(np.arccos(cos_theta)+illumination_angle)
        phi = np.arctan2(Y, X)

        return R3, cos_theta, illumination_cos_theta, phi

    def get(
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
        **kwargs,
    ):
        # Get size of the output
        xSize, ySize = self.get_xy_size(output_region, padding)
        voxel_size = get_active_voxel_size()
        arr = pad_image_to_fft(np.zeros((xSize, ySize))).astype(complex)
        arr = image.maybe_cupy(arr)
        position = np.array(position) * voxel_size[: len(position)]

        pupil_physical_size = working_distance * np.tan(collection_angle) * 2

        z = z * voxel_size[2]

        ratio = offset_z / (working_distance - z)

        # position of pbjective relative particle
        relative_position = np.array(
            (
                position_objective[0] - position[0],
                position_objective[1] - position[1],
                working_distance - z,
            )
        )

        # get field evaluation plane at offset_z
        R3_field, cos_theta_field, illumination_angle_field, phi_field = self.get_plane_in_polar_coords(
            arr.shape, voxel_size, relative_position * ratio, illumination_angle
        )
        
        cos_phi_field, sin_phi_field = np.cos(phi_field), np.sin(phi_field)
        # x and y position of a beam passing through field evaluation plane on the objective
        x_farfield = (
            position[0]
            + R3_field * np.sqrt(1 - cos_theta_field**2) * cos_phi_field / ratio
        )
        y_farfield = (
            position[1]
            + R3_field * np.sqrt(1 - cos_theta_field**2) * sin_phi_field / ratio
        )

        # if the beam is within the pupil
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

            # If the input polarization is circular set the coefficients to 1/2.
            elif isinstance(input_polarization, (str)):
                if input_polarization == "circular":
                    S1_coef = 1/2
                    S2_coef = 1/2

        if isinstance(output_polarization, (float, int, Quantity)):
            if isinstance(input_polarization, Quantity):
                output_polarization = output_polarization.to("rad")
                output_polarization = output_polarization.magnitude

            S1_coef *= np.sin(phi_field + output_polarization)
            S2_coef *= np.cos(phi_field + output_polarization) * illumination_angle_field

        # Wave vector
        k = 2 * np.pi / wavelength * refractive_index_medium

        # Harmonics
        A, B = coefficients(L)
        PI, TAU = D.mie_harmonics(illumination_angle_field, L)

        # Normalization factor
        E = [(2 * i + 1) / (i * (i + 1)) for i in range(1, L + 1)]

        # Scattering terms
        S1 = sum([E[i] * A[i] * PI[i] + E[i] * B[i] * TAU[i] for i in range(0, L)])
        S2 = sum([E[i] * B[i] * PI[i] + E[i] * A[i] * TAU[i] for i in range(0, L)])
        
        arr[pupil_mask] = (
            -1j
            / (k * R3_field)
            * np.exp(1j * k * R3_field)
            * (S2 * S2_coef + S1 * S1_coef)
        ) / amp_factor
        
        # For phase shift correction (a multiplication of the field by exp(1j * k * z)).
        if phase_shift_correction:
            arr *= np.exp(1j * k * z + 1j * np.pi / 2)

        # For partially coherent illumination
        if coherence_length:
            sigma = z * np.sqrt((coherence_length / z + 1) ** 2 - 1)
            sigma = sigma * (offset_z / z)

            mask = np.zeros_like(arr)
            y, x = np.ogrid[
                -mask.shape[0] // 2 : mask.shape[0] // 2,
                -mask.shape[1] // 2 : mask.shape[1] // 2,
            ]
            mask = np.exp(-0.5 * (x**2 + y**2) / ((sigma) ** 2))

            mask = image.maybe_cupy(mask)
            arr = arr * mask

        fourier_field = np.fft.fft2(arr)

        propagation_matrix = get_propagation_matrix(
            fourier_field.shape,
            pixel_size=voxel_size[2],
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
                + (padding[1] - arr.shape[1] / 2) * voxel_size[1]
            ),
        )
        fourier_field = fourier_field * propagation_matrix * np.exp(-1j * k * offset_z)

        if return_fft:
            return fourier_field[..., np.newaxis]
        else:
            return np.fft.ifft2(fourier_field)[..., np.newaxis]


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
    radius : float
        Radius of the mie particle in meter.
    refractive_index : float
        Refractive index of the particle
    L : int or str
        The number of terms used to evaluate the mie theory. If `"auto"`,
        it determines the number of terms automatically.
    position : array_like[float, float (, float)]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    offset_z : "auto" or float
        Distance from the particle in the z direction the field is evaluated.
        If "auto", this is calculated from the pixel size and
        `collection_angle`
    collection_angle : "auto" or float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective
        is the limiting aperature).
    input_polarization: float or Quantity
        Defines the polarization angle of the input. For simulating circularly
        polarized light we recommend a coherent sum of two simulated fields. For
        unpolarized light we recommend a incoherent sum of two simulated fields.
    output_polarization: float or Quantity or None
        If None, the output light is not polarized. Otherwise defines the angle of the
        polarization filter after the sample. For off-axis, keep the same as input_polarization.
    """

    def __init__(
        self,
        radius: PropertyLike[float] = 1e-6,
        refractive_index: PropertyLike[float] = 1.45,
        **kwargs,
    ):
        def coeffs(radius, refractive_index, refractive_index_medium, wavelength):

            if isinstance(radius, Quantity):
                radius = radius.to("m").magnitude
            if isinstance(wavelength, Quantity):
                wavelength = wavelength.to("m").magnitude

            def inner(L):
                return D.mie_coefficients(
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
    """Scattered field by a stratified sphere

    A stratified sphere is a sphere with several concentric shells of uniform
    refractive index.

    Should be calculated on at least a 64 by 64 grid. Use padding in the
    optics if necessary

    Calculates the scattered field by in a homogenous medium, as predicted by
    Mie theory. Note that the induced phase shift is calculated in comparison
    to the `refractive_index_medium` property of the optical device.

    Parameters
    ----------
    radius : list of float
        The radius of each cell in increasing order.
    refractive_index : list of float
        Refractive index of each cell in the same order as `radius`
    L : int or str
        The number of terms used to evaluate the mie theory. If `"auto"`,
        it determines the number of terms automatically.
    position : array_like[float, float (, float)]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    offset_z : "auto" or float
        Distance from the particle in the z direction the field is evaluated.
        If "auto", this is calculated from the pixel size and
        `collection_angle`
    collection_angle : "auto" or float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective
        is the limiting aperature).
    input_polarization: float or Quantity
        Defines the polarization angle of the input. For simulating circularly
        polarized light we recommend a coherent sum of two simulated fields. For
        unpolarized light we recommend a incoherent sum of two simulated fields.
    output_polarization: float or Quantity or None
        If None, the output light is not polarized. Otherwise defines the angle of the
        polarization filter after the sample. For off-axis, keep the same as input_polarization.
    """

    def __init__(
        self,
        radius: PropertyLike[ArrayLike[float]] = [1e-6],
        refractive_index: PropertyLike[ArrayLike[float]] = [1.45],
        **kwargs,
    ):
        def coeffs(radius, refractive_index, refractive_index_medium, wavelength):
            assert np.all(
                radius[1:] >= radius[:-1]
            ), "Radius of the shells of a stratified sphere should be monotonically increasing"

            def inner(L):
                return D.stratified_mie_coefficients(
                    np.array(refractive_index) / refractive_index_medium,
                    np.array(radius) * 2 * np.pi / wavelength * refractive_index_medium,
                    L,
                )

            return inner

        super().__init__(
            coefficients=coeffs,
            radius=radius,
            refractive_index=refractive_index,
            **kwargs,
        )
