""" Features for optically imaging of samples

Contains features which performs physical simulations of optical devices to
create camera images of samples.

Classes
-------
Microscope
    Image a sample using an optical system.
Optics
    Abstract base optics class.
Fluorescence
    Optical device for fluorescenct imaging.
Brightfield
    Images coherently illuminated samples.
"""


from pint.quantity import Quantity
from deeptrack.backend.units import ConversionTable
from deeptrack.properties import propagate_data_to_dependencies
import numpy as np
from .features import DummyFeature, Feature, StructuralFeature
from .image import Image, pad_image_to_fft, maybe_cupy
from .types import ArrayLike, PropertyLike
from .backend._config import cupy
from scipy.ndimage import convolve

from . import units as u
from deeptrack import image


class Microscope(StructuralFeature):
    """Image a sample using an optical system.

    Wraps a feature-set defining a sample and a feature-set defining the optics.

    Parameters
    ----------
    sample : Feature
        A feature-set resolving a list of images describing the sample to be imaged
    objective : Feature
        A feature-set defining the optical device that images the sample
    """

    __distributed__ = False

    def __init__(self, sample: Feature, objective: Feature, **kwargs):
        super().__init__(**kwargs)
        self._sample = self.add_feature(sample)
        self._objective = self.add_feature(objective)

    def get(self, image, **kwargs):

        # Grab properties from the objective to pass to the sample
        additional_sample_kwargs = self._objective.properties()
        propagate_data_to_dependencies(self._sample, **additional_sample_kwargs)

        with u.context("dt", pixel_size=additional_sample_kwargs["voxel_size"][0]):
            list_of_scatterers = self._sample()

        if not isinstance(list_of_scatterers, list):
            list_of_scatterers = [list_of_scatterers]

        volume_samples = [
            scatterer
            for scatterer in list_of_scatterers
            if not scatterer.get_property("is_field", default=False)
        ]
        field_samples = [
            scatterer
            for scatterer in list_of_scatterers
            if scatterer.get_property("is_field", default=False)
        ]

        sample_volume, limits = _create_volume(
            volume_samples, **additional_sample_kwargs
        )
        sample_volume = Image(sample_volume)

        for scatterer in volume_samples + field_samples:
            sample_volume.merge_properties_from(scatterer)

        propagate_data_to_dependencies(
            self._objective,
            limits=limits,
            fields=field_samples,
        )

        imaged_sample = self._objective.resolve(sample_volume)

        # Merge with input
        if not image:
            return imaged_sample

        if not isinstance(image, list):
            image = [image]
        for i in range(len(image)):
            image[i].merge_properties_from(imaged_sample)
        return image


# OPTICAL SYSTEMS


class Optics(Feature):
    """Abstract base optics class.

    Provides structure and methods common for most optical devices.

    Parameters
    ----------
    NA : float
        The NA of the limiting aperature.
    wavelength : float
        The wavelength of the scattered light in meters.
    magnification : float
        The magnification of the optical system.
    resolution : array_like[float (, float, float)]
        The distance between pixels in the camera. A third value can be
        included to define the resolution in the z-direction.
    refractive_index_medium : float
        The refractive index of the medium.
    padding : array_like[int, int, int, int]
        Pads the sample volume with zeros to avoid edge effects.
    output_region : array_like[int, int, int, int]
        The region of the image to output (x,y,width,height). Default
        None returns entire image.
    pupil : Feature
        A feature-set resolving the pupil function at focus. The feature-set
        receive an unaberrated pupil as input.

    """

    __conversion_table__ = ConversionTable(
        wavelength=(u.meter, u.meter),
        resolution=(u.meter, u.meter),
        voxel_size=(u.meter, u.meter),
    )

    def __init__(
        self,
        NA: PropertyLike[float] = 0.7,
        wavelength: PropertyLike[float] = 0.66e-6,
        magnification: PropertyLike[float] = 10,
        resolution: PropertyLike[float or ArrayLike[float]] = 1e-6,
        refractive_index_medium: PropertyLike[float] = 1.33,
        padding: PropertyLike[ArrayLike[int]] = (10, 10, 10, 10),
        output_region: PropertyLike[ArrayLike[int]] = (0, 0, 128, 128),
        pupil: Feature = None,
        illumination: Feature = None,
        **kwargs
    ):
        def get_voxel_size(resolution, magnification):
            props = self._normalize(resolution=resolution, magnification=magnification)
            return np.ones((3,)) * props["resolution"] / props["magnification"]

        def get_pixel_size(resolution, magnification):
            props = self._normalize(resolution=resolution, magnification=magnification)
            pixel_size = props["resolution"] / props["magnification"]
            if isinstance(pixel_size, Quantity):
                return pixel_size.to(u.meter).magnitude
            else:
                return pixel_size

        super().__init__(
            NA=NA,
            wavelength=wavelength,
            refractive_index_medium=refractive_index_medium,
            magnification=magnification,
            resolution=resolution,
            padding=padding,
            output_region=output_region,
            voxel_size=get_voxel_size,
            pixel_size=get_pixel_size,
            limits=None,
            fields=None,
            **kwargs
        )

        self.pupil = self.add_feature(pupil) if pupil else DummyFeature()
        self.illumination = (
            self.add_feature(illumination) if illumination else DummyFeature()
        )

    def _pupil(
        self,
        shape,
        NA,
        wavelength,
        refractive_index_medium,
        voxel_size,
        include_aberration=True,
        defocus=0,
        **kwargs
    ):
        # Calculates the pupil at each z-position in defocus.
        shape = np.array(shape)

        # Pupil radius
        R = NA / wavelength * np.array(voxel_size)[:2]

        x_radius = R[0] * shape[0]
        y_radius = R[1] * shape[1]

        x = (np.linspace(-(shape[0] / 2), shape[0] / 2 - 1, shape[0])) / x_radius + 1e-8
        y = (np.linspace(-(shape[1] / 2), shape[1] / 2 - 1, shape[1])) / y_radius + 1e-8

        W, H = np.meshgrid(y, x)
        W = maybe_cupy(W)
        H = maybe_cupy(H)
        RHO = W ** 2 + H ** 2
        pupil_function = Image((RHO < 1) + 0.0j, copy=False)
        # Defocus
        z_shift = Image(
            2
            * np.pi
            * refractive_index_medium
            / wavelength
            * voxel_size[2]
            * np.sqrt(1 - (NA / refractive_index_medium) ** 2 * RHO),
            copy=False,
        )

        try:
            z_shift = np.nan_to_num(z_shift, False, 0, 0, 0)
        except TypeError:
            np.nan_to_num(z_shift, z_shift)

        defocus = np.reshape(defocus, (-1, 1, 1))
        z_shift = defocus * np.expand_dims(z_shift, axis=0)

        if include_aberration:
            pupil = self.pupil
            if isinstance(pupil, Feature):

                pupil_function = pupil(pupil_function)
            elif isinstance(pupil, np.ndarray):
                pupil_function *= pupil

        pupil_functions = pupil_function * np.exp(1j * z_shift)

        return pupil_functions

    def _pad_volume(
        self, volume, limits=None, padding=None, output_region=None, **kwargs
    ):
        if limits is None:
            limits = np.zeros((3, 2))

        new_limits = np.array(limits)
        output_region = np.array(output_region)

        # Replace None entries with current limit
        output_region[0] = (
            output_region[0] if not output_region[0] is None else new_limits[0, 0]
        )
        output_region[1] = (
            output_region[1] if not output_region[1] is None else new_limits[0, 1]
        )
        output_region[2] = (
            output_region[2] if not output_region[2] is None else new_limits[1, 0]
        )
        output_region[3] = (
            output_region[3] if not output_region[3] is None else new_limits[1, 1]
        )

        for i in range(2):
            new_limits[i, :] = (
                np.min([new_limits[i, 0], output_region[i] - padding[1]]),
                np.max(
                    [
                        new_limits[i, 1],
                        output_region[i + 2] + padding[i + 2],
                    ]
                ),
            )
        new_volume = np.zeros(
            np.diff(new_limits, axis=1)[:, 0].astype(np.int32),
            dtype=np.complex,
        )

        old_region = (limits - new_limits).astype(np.int32)
        limits = limits.astype(np.int32)
        new_volume[
            old_region[0, 0] : old_region[0, 0] + limits[0, 1] - limits[0, 0],
            old_region[1, 0] : old_region[1, 0] + limits[1, 1] - limits[1, 0],
            old_region[2, 0] : old_region[2, 0] + limits[2, 1] - limits[2, 0],
        ] = volume

        return new_volume, new_limits

    def __call__(self, sample, **kwargs):
        return Microscope(sample, self, **kwargs)


class Fluorescence(Optics):

    """Optical device for fluorescenct imaging

    Images samples by creating a discretized volume, where each pixel
    represents the intensity of the light emitted by fluorophores in
    the the voxel.

    Parameters
    ----------
    NA : float
        The NA of the limiting aperature.
    wavelength : float
        The wavelength of the scattered light in meters.
    magnification : float
        The magnification of the optical system.
    resolution : array_like[float (, float, float)]
        The distance between pixels in the camera. A third value can be
        included to define the resolution in the z-direction.
    refractive_index_medium : float
        The refractive index of the medium.
    padding : array_like[int, int, int, int]
        Pads the sample volume with zeros to avoid edge effects.
    output_region : array_like[int, int, int, int]
        The region of the image to output (x,y,width,height). Default
        None returns entire image.
    pupil : Feature
        A feature-set resolving the pupil function at focus. The feature-set
        receive an unaberrated pupil as input.

    """

    __gpu_compatible__ = True

    def get(self, illuminated_volume, limits, **kwargs):
        """Convolves the image with a pupil function"""

        # Pad volume
        padded_volume, limits = self._pad_volume(
            illuminated_volume, limits=limits, **kwargs
        )

        # Extract indexes of the output region
        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))
        output_region[0] = (
            None
            if output_region[0] is None
            else int(output_region[0] - limits[0, 0] - pad[0])
        )
        output_region[1] = (
            None
            if output_region[1] is None
            else int(output_region[1] - limits[1, 0] - pad[1])
        )
        output_region[2] = (
            None
            if output_region[2] is None
            else int(output_region[2] - limits[0, 0] + pad[2])
        )
        output_region[3] = (
            None
            if output_region[3] is None
            else int(output_region[3] - limits[1, 0] + pad[3])
        )

        padded_volume = padded_volume[
            output_region[0] : output_region[2],
            output_region[1] : output_region[3],
            :,
        ]
        z_limits = limits[2, :]

        output_image = Image(
            maybe_cupy(np.zeros((*padded_volume.shape[0:2], 1))), copy=False
        )

        index_iterator = range(padded_volume.shape[2])

        # Get planes in volume where not all values are 0.
        z_iterator = np.linspace(
            z_limits[0],
            z_limits[1],
            num=padded_volume.shape[2],
            endpoint=False,
        )
        zero_plane = np.all(padded_volume == 0, axis=(0, 1), keepdims=False)
        z_values = z_iterator[~zero_plane]

        # Further pad image to speed up fft
        volume = maybe_cupy(pad_image_to_fft(padded_volume, axes=(0, 1)))

        pupils = self._pupil(volume.shape[:2], defocus=z_values, **kwargs)

        z_index = 0

        # Loop through voluma and convole sample with pupil function

        for i, z in zip(index_iterator, z_iterator):

            if zero_plane[i]:
                continue

            pupil = pupils[z_index]
            z_index += 1

            psf = np.square(np.abs(np.fft.ifft2(np.fft.fftshift(pupil))))

            optical_transfer_function = np.fft.fft2(psf)

            fourier_field = np.fft.fft2(volume[:, :, i])
            convolved_fourier_field = fourier_field * optical_transfer_function

            field = np.fft.ifft2(convolved_fourier_field)

            # # Discard remaining imaginary part (should be 0 up to rounding error)
            field = np.real(field)

            output_image._value[:, :, 0] += field[
                : padded_volume.shape[0], : padded_volume.shape[1]
            ]

        output_image = output_image[pad[0] : -pad[2], pad[1] : -pad[3]]

        output_image.properties = illuminated_volume.properties + pupils.properties

        return output_image


class Brightfield(Optics):
    """Images coherently illuminated samples.

    Images samples by creating a discretized volume, where each pixel
    represents the effective refractive index of that pixel. Light is
    propagated through the sample iteratively by first propagating the
    light in the fourier space, followed by a refractive index correction
    in the real space.

    Parameters
    ----------
    illumination : Feature
        Feature-set resolving the complex field entering the sample. Default
        is a field with all values 1.
    NA : float
        The NA of the limiting aperature.
    wavelength : float
        The wavelength of the scattered light in meters.
    magnification : float
        The magnification of the optical system.
    resolution : array_like[float (, float, float)]
        The distance between pixels in the camera. A third value can be
        included to define the resolution in the z-direction.
    refractive_index_medium : float
        The refractive index of the medium.
    padding : array_like[int, int, int, int]
        Pads the sample volume with zeros to avoid edge effects.
    output_region : array_like[int, int, int, int]
        The region of the image to output (x,y,width,height). Default
        None returns entire image.
    pupil : Feature
        A feature-set resolving the pupil function at focus. The feature-set
        receive an unaberrated pupil as input.

    """

    __gpu_compatible__ = True

    def get(self, illuminated_volume, limits, fields, **kwargs):
        """Convolves the image with a pupil function"""
        # Pad volume
        padded_volume, limits = self._pad_volume(
            illuminated_volume, limits=limits, **kwargs
        )

        # Extract indexes of the output region
        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))
        output_region[0] = (
            None
            if output_region[0] is None
            else int(output_region[0] - limits[0, 0] - pad[0])
        )
        output_region[1] = (
            None
            if output_region[1] is None
            else int(output_region[1] - limits[1, 0] - pad[1])
        )
        output_region[2] = (
            None
            if output_region[2] is None
            else int(output_region[2] - limits[0, 0] + pad[2])
        )
        output_region[3] = (
            None
            if output_region[3] is None
            else int(output_region[3] - limits[1, 0] + pad[3])
        )

        padded_volume = padded_volume[
            output_region[0] : output_region[2],
            output_region[1] : output_region[3],
            :,
        ]
        z_limits = limits[2, :]

        output_image = Image(image.maybe_cupy(np.zeros((*padded_volume.shape[0:2], 1))))

        index_iterator = range(padded_volume.shape[2])
        z_iterator = np.linspace(
            z_limits[0],
            z_limits[1],
            num=padded_volume.shape[2],
            endpoint=False,
        )

        zero_plane = np.all(padded_volume == 0, axis=(0, 1), keepdims=False)
        # z_values = z_iterator[~zero_plane]

        volume = pad_image_to_fft(padded_volume, axes=(0, 1))

        voxel_size = kwargs["voxel_size"]

        pupils = [
            self._pupil(
                volume.shape[:2], defocus=[1], include_aberration=False, **kwargs
            )[0],
            self._pupil(
                volume.shape[:2],
                defocus=[-z_limits[1]],
                include_aberration=True,
                **kwargs
            )[0],
        ]

        pupil_step = np.fft.fftshift(pupils[0])

        light_in = image.maybe_cupy(np.ones(volume.shape[:2], dtype=np.complex))
        light_in = self.illumination.resolve(light_in)
        light_in = np.fft.fft2(light_in)

        K = 2 * np.pi / kwargs["wavelength"]

        field_z = [field.get_property("z") for field in fields]
        field_offsets = [field.get_property("offset_z", default=0) for field in fields]

        z = z_limits[1]
        for i, z in zip(index_iterator, z_iterator):
            light_in = light_in * pupil_step

            to_remove = []
            for idx, fz in enumerate(field_z):
                if fz < z:
                    propagation_matrix = self._pupil(
                        fields[idx].shape,
                        defocus=[z - fz - field_offsets[idx] / voxel_size[-1]],
                        include_aberration=False,
                        **kwargs
                    )[0]

                    propagation_matrix = propagation_matrix * np.exp(
                        1j
                        * voxel_size[-1]
                        * 2
                        * np.pi
                        / kwargs["wavelength"]
                        * kwargs["refractive_index_medium"]
                        * (z - fz)
                    )
                    light_in += np.fft.fft2(fields[idx][:, :, 0]) * np.fft.fftshift(
                        propagation_matrix
                    )
                    to_remove.append(idx)

            for idx in reversed(to_remove):
                fields.pop(idx)
                field_z.pop(idx)
                field_offsets.pop(idx)

            if zero_plane[i]:
                continue

            ri_slice = volume[:, :, i]
            light = np.fft.ifft2(light_in)
            light_out = light * np.exp(1j * ri_slice * voxel_size[-1] * K)
            light_in = np.fft.fft2(light_out)

        # Add remaining fields
        for idx, fz in enumerate(field_z):
            prop_dist = z - fz - field_offsets[idx] / voxel_size[-1]
            propagation_matrix = self._pupil(
                fields[idx].shape,
                defocus=[prop_dist],
                include_aberration=False,
                **kwargs
            )[0]

            propagation_matrix = propagation_matrix * np.exp(
                -1j
                * voxel_size[-1]
                * 2
                * np.pi
                / kwargs["wavelength"]
                * kwargs["refractive_index_medium"]
                * prop_dist
            )
            light_in += np.fft.fft2(fields[idx][:, :, 0]) * np.fft.fftshift(
                propagation_matrix
            )

        light_in_focus = light_in * np.fft.fftshift(pupils[-1])

        output_image = np.fft.ifft2(light_in_focus)[
            : padded_volume.shape[0], : padded_volume.shape[1]
        ]
        output_image = np.expand_dims(output_image, axis=-1)
        output_image = Image(output_image[pad[0] : -pad[2], pad[1] : -pad[3]])

        if not kwargs.get("return_field", False):
            output_image = np.square(np.abs(output_image))

        output_image.properties = illuminated_volume.properties

        return output_image


class IlluminationGradient(Feature):
    """Adds a gradient in the illumination

    Parameters
    ----------
    gradient : array_like[float, float]
        Gradient of the plane to add to the amplitude of the field in pixels.
    constant : float
        Constant value to add to the amplitude of the field.
    vmin : float
        clips the amplitude of the field to be at least this value
    vmax : float
        clips the amplitude of the field to be at most this value

    """

    def __init__(
        self,
        gradient: PropertyLike[ArrayLike[float]] = (0, 0),
        constant: PropertyLike[float] = 0,
        vmin: PropertyLike[float] = 0,
        vmax: PropertyLike[float] = np.inf,
        **kwargs
    ):
        super().__init__(
            gradient=gradient, constant=constant, vmin=vmin, vmax=vmax, **kwargs
        )

    def get(self, image, gradient, constant, vmin, vmax, **kwargs):

        x = np.arange(image.shape[0])
        y = np.arange(image.shape[1])

        X, Y = np.meshgrid(y, x)

        amplitude = X * gradient[0] + Y * gradient[1]

        if image.ndim == 3:
            amplitude = np.expand_dims(amplitude, axis=-1)
        amplitude = np.clip(np.abs(image) + amplitude + constant, vmin, vmax)

        image = amplitude * image / np.abs(image)
        image[np.isnan(image)] = 0

        return image


def _get_position(image, mode="corner", return_z=False):
    # Extracts the position of the upper left corner of a scatterer
    num_outputs = 2 + return_z

    if mode == "corner" and image.size > 0:
        import scipy.ndimage

        shift = scipy.ndimage.measurements.center_of_mass(np.abs(image))

        if np.isnan(shift).any():
            shift = np.array(image.shape) / 2

    else:
        shift = np.zeros((num_outputs))

    position = np.array(image.get_property("position", default=None))

    # position[:2] = position[:2]

    if position is None:
        return position

    if len(position) == 3:
        if return_z:
            return position - shift
        else:
            return position[0:2] - shift[0:2]

    elif len(position) == 2:
        if return_z:
            outp = (
                np.array([position[0], position[1], image.get_property("z", default=0)])
                - shift
            )
            return outp
        else:
            return position - shift[0:2]

    return position


def _create_volume(
    list_of_scatterers,
    pad=(0, 0, 0, 0),
    output_region=(None, None, None, None),
    refractive_index_medium=1.33,
    **kwargs
):
    # Converts a list of scatterers into a volume.

    if not isinstance(list_of_scatterers, list):
        list_of_scatterers = [list_of_scatterers]

    volume = np.zeros((1, 1, 1), dtype=np.complex)
    limits = None
    OR = np.zeros((4,))
    OR[0] = np.inf if output_region[0] is None else int(output_region[0] - pad[0])
    OR[1] = -np.inf if output_region[1] is None else int(output_region[1] - pad[1])
    OR[2] = np.inf if output_region[2] is None else int(output_region[2] + pad[2])
    OR[3] = -np.inf if output_region[3] is None else int(output_region[3] + pad[3])

    for scatterer in list_of_scatterers:

        position = _get_position(scatterer, mode="corner", return_z=True)

        if scatterer.get_property("intensity", None) is not None:
            scatterer_value = scatterer.get_property("intensity")
        elif scatterer.get_property("refractive_index", None) is not None:
            scatterer_value = (
                scatterer.get_property("refractive_index") - refractive_index_medium
            )
        else:
            scatterer_value = scatterer.get_property("value")

        scatterer = scatterer * scatterer_value

        if limits is None:
            limits = np.zeros((3, 2), dtype=np.int32)
            limits[:, 0] = np.floor(position).astype(np.int32)
            limits[:, 1] = np.floor(position).astype(np.int32) + 1

        if (
            position[0] + scatterer.shape[0] < OR[0]
            or position[0] > OR[2]
            or position[1] + scatterer.shape[1] < OR[1]
            or position[1] > OR[3]
        ):
            continue

        padded_scatterer = Image(
            np.pad(
                scatterer,
                [(2, 2), (2, 2), (2, 2)],
                "constant",
                constant_values=0,
            )
        )
        padded_scatterer.merge_properties_from(scatterer)

        scatterer = padded_scatterer
        position = _get_position(scatterer, mode="corner", return_z=True)
        shape = np.array(scatterer.shape)

        if position is None:
            RuntimeWarning(
                "Optical device received an image without a position property. It will be ignored."
            )
            continue

        splined_scatterer = np.zeros_like(scatterer)

        x_off = position[0] - np.floor(position[0])
        y_off = position[1] - np.floor(position[1])

        kernel = np.array(
            [
                [0, 0, 0],
                [0, (1 - x_off) * (1 - y_off), (1 - x_off) * y_off],
                [0, x_off * (1 - y_off), x_off * y_off],
            ]
        )

        for z in range(scatterer.shape[2]):
            if splined_scatterer.dtype == np.complex:
                splined_scatterer[:, :, z] = (
                    convolve(np.real(scatterer[:, :, z]), kernel, mode="constant")
                    + convolve(np.imag(scatterer[:, :, z]), kernel, mode="constant")
                    * 1j
                )
            else:
                splined_scatterer[:, :, z] = convolve(
                    scatterer[:, :, z], kernel, mode="constant"
                )

        scatterer = splined_scatterer
        position = np.floor(position)
        new_limits = np.zeros(limits.shape, dtype=np.int32)
        for i in range(3):
            new_limits[i, :] = (
                np.min([limits[i, 0], position[i]]),
                np.max([limits[i, 1], position[i] + shape[i]]),
            )

        if not (np.array(new_limits) == np.array(limits)).all():
            new_volume = np.zeros(
                np.diff(new_limits, axis=1)[:, 0].astype(np.int32),
                dtype=np.complex,
            )
            old_region = (limits - new_limits).astype(np.int32)
            limits = limits.astype(np.int32)
            new_volume[
                old_region[0, 0] : old_region[0, 0] + limits[0, 1] - limits[0, 0],
                old_region[1, 0] : old_region[1, 0] + limits[1, 1] - limits[1, 0],
                old_region[2, 0] : old_region[2, 0] + limits[2, 1] - limits[2, 0],
            ] = volume
            volume = new_volume
            limits = new_limits

        within_volume_position = position - limits[:, 0]

        # NOTE: Maybe shouldn't be additive.
        volume[
            int(within_volume_position[0]) : int(within_volume_position[0] + shape[0]),
            int(within_volume_position[1]) : int(within_volume_position[1] + shape[1]),
            int(within_volume_position[2]) : int(within_volume_position[2] + shape[2]),
        ] += scatterer
    return volume, limits
