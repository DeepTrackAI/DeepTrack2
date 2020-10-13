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


from threading import Lock
import numpy as np

# from scipy.special import jv as jn, spherical_jn as jv, h1vp, eval_legendre as leg, jvp
import deeptrack.backend as D
from deeptrack.features import Feature, MERGE_STRATEGY_APPEND
from deeptrack.image import Image
import deeptrack.image


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
        Sets the axes along which the calculation is upsampled (default is None,
        which implies all axes are upsampled).
    crop_zeros : bool
        Whether to remove slices in which all elements are zero.
    """

    __list_merge_strategy__ = MERGE_STRATEGY_APPEND
    __distributed__ = False

    def __init__(
        self,
        position=(32, 32),
        z=0.0,
        value=1.0,
        position_unit="pixel",
        upsample=1,
        **kwargs
    ):
        self._processed_properties = False
        super().__init__(
            position=position,
            z=z,
            value=value,
            position_unit=position_unit,
            upsample=upsample,
            **kwargs
        )

    def _process_properties(self, properties: dict) -> dict:
        # Rescales the position property
        self._processed_properties = True
        if "position" in properties:
            if properties["position_unit"] == "meter":
                properties["position"] = (
                    np.array(properties["position"])
                    / np.array(properties["voxel_size"])[: len(properties["position"])]
                    / properties.get("upscale", 1)
                )
                properties["z"] = (
                    np.array(properties["z"])
                    / np.array(properties["voxel_size"])[: len(properties["position"])]
                    / properties.get("upscale", 1)
                )

        return properties

    def _process_and_get(
        self, *args, voxel_size, upsample, upsample_axes=None, crop_empty=True, **kwargs
    ):
        # Post processes the created object to handle upsampling,
        # as well as cropping empty slices.
        if not self._processed_properties:
            import warnings

            warnings.warn(
                "Overridden _process_properties method does not call super. This is likely to result in errors if used with Optics.upscale != 1."
            )

        # Calculates upsampled voxel_size
        if upsample_axes is None:
            upsample_axes = range(3)

        voxel_size = np.array(voxel_size)
        for axis in upsample_axes:
            voxel_size[axis] /= upsample

        # calls parent _process_and_get
        new_image = super()._process_and_get(
            *args, voxel_size=voxel_size, upsample=upsample, **kwargs
        )
        new_image = new_image[0]

        if new_image.size == 0:
            warnings.warn(
                "Scatterer created that is smaller than a pixel. This may yield inconsistent results. Consider using upsample on the scatterer, or upscale on the optics.",
                Warning,
            )

        # Downsamples the image along the axes it was upsampled
        if upsample != 1 and upsample_axes:

            # Pad image to ensure it is divisible by upsample
            increase = np.array(new_image.shape)
            for axis in upsample_axes:
                increase[axis] = upsample - (new_image.shape[axis] % upsample)
            pad_width = [(0, inc) for inc in increase]
            new_image = np.pad(new_image, pad_width, mode="constant")

            # Finds reshape size for downsampling
            new_shape = []
            for axis in range(new_image.ndim):
                if axis in upsample_axes:
                    new_shape += [new_image.shape[axis] // upsample, upsample]
                else:
                    new_shape += [new_image.shape[axis]]

            # Downsamples
            new_image = np.reshape(new_image, new_shape).mean(
                axis=tuple(np.array(upsample_axes, dtype=np.int32) * 2 + 1)
            )

        # Crops empty slices
        if crop_empty:
            new_image = new_image[~np.all(new_image == 0, axis=(1, 2))]
            new_image = new_image[:, ~np.all(new_image == 0, axis=(0, 2))]
            new_image = new_image[:, :, ~np.all(new_image == 0, axis=(0, 1))]

        return [Image(new_image)]


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
        kwargs.pop("upsample", False)
        kwargs.pop("upsample_axes", False)
        super().__init__(upsample=1, upsample_axes=(), **kwargs)

    def get(self, image, **kwargs):
        return np.ones((1, 1, 1))


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
    """

    def __init__(self, radius=1e-6, rotation=0, **kwargs):
        super().__init__(
            radius=radius, rotation=rotation, upsample_axes=(0, 1), **kwargs
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

    def get(self, *ignore, radius, rotation, voxel_size, **kwargs):

        # Create a grid to calculate on
        rad = radius[:2] / voxel_size[:2]
        ceil = int(np.max(np.ceil(rad)))
        X, Y = np.meshgrid(np.arange(-ceil, ceil), np.arange(-ceil, ceil))

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

    def __init__(self, radius=1e-6, **kwargs):
        super().__init__(radius=radius, **kwargs)

    def get(self, image, radius, voxel_size, **kwargs):

        # Create a grid to calculate on
        rad = radius / voxel_size
        rad_ceil = np.ceil(rad)
        x = np.arange(-rad_ceil[0], rad_ceil[0])
        y = np.arange(-rad_ceil[1], rad_ceil[1])
        z = np.arange(-rad_ceil[2], rad_ceil[2])
        X, Y, Z = np.meshgrid((x / rad[0]) ** 2, (y / rad[1]) ** 2, (z / rad[2]) ** 2)

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
    """

    def __init__(self, radius=1e-6, rotation=0, **kwargs):
        super().__init__(radius=radius, rotation=rotation, **kwargs)

    def _process_properties(self, propertydict):
        """Preprocess the input to the method .get()

        Ensures that the radius and the rotation properties both are arrays of
        length 3.

        If the radius is a single value, the particle is made a sphere
        If the radius are two values, the smallest value is appended as the third value

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

    def get(self, image, radius, rotation, voxel_size, **kwargs):

        radius_in_pixels = radius / voxel_size

        max_rad = np.max(radius) / voxel_size
        rad_ceil = np.ceil(max_rad)

        # Create grid to calculate on
        x = np.arange(-rad_ceil[0], rad_ceil[0])
        y = np.arange(-rad_ceil[1], rad_ceil[1])
        z = np.arange(-rad_ceil[2], rad_ceil[2])
        X, Y, Z = np.meshgrid(x, y, z)

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
            (XR / radius_in_pixels[0]) ** 2
            + (YR / radius_in_pixels[1]) ** 2
            + (ZR / radius_in_pixels[2]) ** 2
            < 1
        ) * 1.0

        return mask


class MieScatterer(Scatterer):
    """Base implementation of a Mie particle.

    New Mie-theory scatterers can be implemented by extending this class, and passing
    a function that calculates the coefficients of the harmonics up to order `L`. To be
    precise, the feature expects a wrapper function that takes the current values of the
    properties, as well as a inner function that takes an integer as the only parameter,
    and calculates the coefficients up to that integer. The return format is expected to
    be a tuple with two values, corresponding to `an` and `bn`. See
    `deeptrack.backend.mie_coefficients` for an example.

    Parameters
    ----------
    coefficients : Callable[int] -> Tuple[ndarray, ndarray]
        Function that returns the harmonics coefficients.
    offset_z : "auto" or float
        Distance from the particle in the z direction the field is evaluated. If "auto",
        this is calculated from the pixel size and `collection_angle`
    collection_angle : "auto" or float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective is the limiting
        aperature).
    polarization_angle : float
        Angle of the polarization of the incoming light relative to the x-axis.
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
    """

    def __init__(
        self,
        coefficients,
        offset_z="auto",
        polarization_angle=0,
        collection_angle="auto",
        L="auto",
        **kwargs
    ):
        kwargs.pop("is_field", None)
        kwargs.pop("crop_empty", None)

        super().__init__(
            is_field=True,
            crop_empty=False,
            L=L,
            offset_z=offset_z,
            polarization_angle=polarization_angle,
            collection_angle=collection_angle,
            coefficients=coefficients,
            **kwargs
        )

    def _process_properties(self, properties):

        properties = super()._process_properties(properties)

        if properties["L"] == "auto":
            try:
                v = 2 * np.pi * np.max(properties["radius"]) / properties["wavelength"]
                properties["L"] = int(np.ceil(v + 4 * (v ** (1 / 3)) + 2))
            except (ValueError, TypeError):
                pass
        if properties["collection_angle"] == "auto":
            properties["collection_angle"] = np.sqrt(
                1 - properties["NA"] ** 2 / properties["refractive_index_medium"] ** 2
            )
        if properties["offset_z"] == "auto":
            properties["offset_z"] = (
                32
                * min(properties["voxel_size"][:2])
                / np.sin(properties["collection_angle"])
                * properties["upscale"]
            )
        return properties

    def get(
        self,
        image,
        position,
        upscaled_output_region,
        voxel_size,
        padding,
        wavelength,
        refractive_index_medium,
        L,
        offset_z,
        collection_angle,
        polarization_angle,
        coefficients,
        upscale=1,
        **kwargs
    ):

        xSize = (
            padding[2]
            + upscaled_output_region[2]
            - upscaled_output_region[0]
            + padding[0]
        )
        ySize = (
            padding[3]
            + upscaled_output_region[3]
            - upscaled_output_region[1]
            + padding[1]
        )
        arr = deeptrack.image.pad_image_to_fft(np.zeros((xSize, ySize)))

        # Evluation grid
        x = np.arange(-padding[0], arr.shape[0] - padding[0]) - (position[1]) * upscale
        y = np.arange(-padding[1], arr.shape[1] - padding[1]) - (position[0]) * upscale
        X, Y = np.meshgrid(x * voxel_size[0], y * voxel_size[1], indexing="ij")

        R2 = np.sqrt(X ** 2 + Y ** 2)
        R3 = np.sqrt(R2 ** 2 + (offset_z) ** 2)
        ct = offset_z / R3

        ANGLE = np.arctan2(Y, X) + polarization_angle
        COS2 = np.square(np.cos(ANGLE))
        SIN2 = 1 - COS2

        ct_max = np.cos(collection_angle)

        # Wave vector
        k = 2 * np.pi / wavelength * refractive_index_medium

        # Harmonics

        A, B = coefficients(L)
        PI, TAU = D.mie_harmonics(ct, L)

        # Normalization factor
        E = [(2 * l + 1) / (l * (l + 1)) for l in range(1, L + 1)]

        # Scattering terms
        S1 = sum([E[l] * A[l] * TAU[l] + E[l] * B[l] * PI[l] for l in range(0, L)])
        S2 = sum([E[l] * B[l] * TAU[l] + E[l] * A[l] * PI[l] for l in range(0, L)])

        field = (
            (ct > ct_max)
            * 1j
            / (k * R3)
            * np.exp(1j * k * (R3 - offset_z))
            * (S1 * COS2 + S2 * SIN2)
        )

        return np.expand_dims(field, axis=-1)


class MieSphere(MieScatterer):
    """Scattered field by a sphere

    Should be calculated on at least a 64 by 64 grid. Use padding in the optics if necessary

    Calculates the scattered field by a spherical particle in a homogenous medium,
    as predicted by Mie theory. Note that the induced phase shift is calculated
    in comparison to the `refractive_index_medium` property of the optical device.

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
        Distance from the particle in the z direction the field is evaluated. If "auto",
        this is calculated from the pixel size and `collection_angle`
    collection_angle : "auto" or float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective is the limiting
        aperature).
    polarization_angle : float
        Angle of the polarization of the incoming light relative to the x-axis.
    """

    def __init__(
        self,
        radius=1e-6,
        refractive_index=1.45,
        offset_z="auto",
        polarization_angle=0,
        collection_angle="auto",
        L="auto",
        **kwargs
    ):
        def coeffs(radius, refractive_index, refractive_index_medium, wavelength):
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
            L=L,
            offset_z=offset_z,
            polarization_angle=polarization_angle,
            collection_angle=collection_angle,
            **kwargs
        )


class MieStratifiedSphere(MieScatterer):
    """Scattered field by a stratified sphere

    A stratified sphere is a sphere with several concentric shells of uniform refractive index.

    Should be calculated on at least a 64 by 64 grid. Use padding in the optics if necessary

    Calculates the scattered field by in a homogenous medium, as predicted by Mie theory.
    Note that the induced phase shift is calculated in comparison to the
    `refractive_index_medium` property of the optical device.

    Parameters
    ----------
    radius : float
        The radius of each cell in increasing order.
    refractive_index : float
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
        Distance from the particle in the z direction the field is evaluated. If "auto",
        this is calculated from the pixel size and `collection_angle`
    collection_angle : "auto" or float
        The maximum collection angle in radians. If "auto", this
        is calculated from the objective NA (which is true if the objective is the limiting
        aperature).
    polarization_angle : float
        Angle of the polarization of the incoming light relative to the x-axis.
    """

    def __init__(
        self,
        radius=1e-6,
        refractive_index=1.45,
        offset_z="auto",
        polarization_angle=0,
        collection_angle="auto",
        L="auto",
        **kwargs
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
            L=L,
            offset_z=offset_z,
            polarization_angle=polarization_angle,
            collection_angle=collection_angle,
            **kwargs
        )
