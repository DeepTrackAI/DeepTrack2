"""Features for optical imaging of samples.

This module provides classes and functionalities for simulating optical
imaging systems, enabling the generation of realistic camera images of
biological and physical samples. The primary goal is to offer tools for
modeling and computing optical phenomena such as brightfield, fluorescence,
holography, and other imaging modalities.

Key Features
------------
- **Microscope Simulation**

  The `Microscope` class acts as a high-level interface for imaging samples
  using defined optical systems. It coordinates the interaction between the
  sample and the optical system, ensuring seamless simulation of imaging
  processes.

- **Optical Systems**

  The `Optics` class and its derived classes represent various optical
  devices, defining core imaging properties such as resolution, magnification,
  numerical aperture (NA), and wavelength. Subclasses like `Brightfield`,
  `Fluorescence`, `Holography`, `Darkfield`, and `ISCAT` offer specialized
  configurations tailored to different imaging techniques.

- **Sample Illumination and Volume Simulation**

  Features like `IlluminationGradient` enable realistic simulation of
  non-uniform sample illumination, critical for generating realistic images.
  The `_create_volume` function facilitates combining multiple scatterers
  into a single unified volume, supporting 3D imaging.

- **Integration with DeepTrack**

  Full compatibility with DeepTrack's feature pipeline allows for dynamic
  and complex simulations, incorporating physics-based models and real-time
  adjustments to sample and imaging properties.

Module Structure
----------------
Classes:

- `Microscope`: Represents a simulated optical microscope that integrates the 
sample and optical systems. It provides an interface to simulate imaging by 
combining the sample properties with the configured optical system.

- `Optics`: An abstract base class representing a generic optical device. 
Subclasses implement specific optical systems by defining imaging properties 
and behaviors.

- `Brightfield`:  Simulates brightfield microscopy, commonly used for observing
unstained or stained samples under transmitted light. This class serves as the 
base for additional imaging techniques.

- `Holography`: Simulates holographic imaging, capturing phase information from
the sample. Suitable for reconstructing 3D images and measuring refractive 
index variations.  

- `Darkfield`: Simulates darkfield microscopy, which enhances contrast by 
imaging scattered light against a dark background. Often used to highlight fine
structures in samples.  

- `ISCAT`: Simulates interferometric scattering microscopy (ISCAT), an advanced 
technique for detecting small particles or molecules based on scattering and 
interference.  

- `Fluorescence`: Simulates fluorescence microscopy, modeling emission 
processes for fluorescent samples. Includes essential optical system 
configurations and fluorophore behavior.

- `IlluminationGradient`: Adds a gradient to the illumination of the sample, 
enabling simulations of non-uniform lighting conditions often seen in 
real-world experiments.

Utility Functions:

- `_get_position(image, mode, return_z)`

    def _get_position(
        image: np.ndarray, mode: str = "corner", return_z: bool = False
    ) -> Tuple[int, int, Optional[int]]

    Extracts the position of the upper-left corner of a scatterer in the image.

- `_create_volume(list_of_scatterers:, pad, output_region, refractive_index_medium, **kwargs)`

    def _create_volume(
        list_of_scatterers: List[np.ndarray],
        pad: int,
        output_region: Tuple[int, int, int, int],
        refractive_index_medium: float,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray

    Combines multiple scatterer objects into a single 3D volume for imaging.

- `_pad_volume(volume, limits, padding, output_region, **kwargs)`

    def _pad_volume(
        volume: np.ndarray,
        limits: np.ndarray,
        padding: Tuple[int, int, int, int],
        output_region: Tuple[int, int, int, int],
        **kwargs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]

    Pads a volume with zeros to avoid edge effects during imaging.

Examples
--------
Simulating an image with the `Brightfield` class:

>>> import deeptrack as dt

>>> scatterer = dt.PointParticle()
>>> optics = dt.Brightfield()
>>> image = optics(scatterer)
>>> print(image().shape)
(128, 128, 1)
>>> image.plot(cmap="gray")

Simulating an image with the `Fluorescence` class:

>>> import deeptrack as dt

>>> scatterer = dt.PointParticle()
>>> optics = dt.Fluorescence()
>>> image = optics(scatterer)
>>> print(image().shape)
(128, 128, 1)
>>> image.plot(cmap="gray")

"""

from pint import Quantity
from typing import Any, Dict, List, Tuple, Union
from deeptrack.backend.units import (
    ConversionTable,
    create_context,
    get_active_scale,
    get_active_voxel_size,
)
from deeptrack.math import AveragePooling
from deeptrack.features import propagate_data_to_dependencies
import numpy as np
from .features import DummyFeature, Feature, StructuralFeature
from .image import Image, pad_image_to_fft, maybe_cupy
from .types import ArrayLike, PropertyLike
from .backend._config import cupy
from scipy.ndimage import convolve
import warnings

from . import units as u
from .backend import config
from deeptrack import image


class Microscope(StructuralFeature):
    """Simulates imaging of a sample using an optical system.

    This class combines a feature-set that defines the sample to be imaged with
    a feature-set defining the optical system, enabling the simulation of 
    optical imaging processes.

    Parameters
    ----------
    sample: Feature
        A feature-set resolving a list of images describing the sample to be
        imaged.
    objective: Feature
        A feature-set defining the optical device that images the sample.

    Attributes
    -----------
    __distributed__: bool
        If True, the feature is distributed across multiple workers.
    _sample: Feature
        The feature-set defining the sample to be imaged.
    _objective: Feature
        The feature-set defining the optical system imaging the sample.

    Methods
    -------
    `get(image: Image or None, **kwargs: Dict[str, Any]) -> Image`
        Simulates the imaging process using the defined optical system and 
        returns the resulting image.

    Examples
    --------
    Simulating an image using a brightfield optical system:

    >>> import deeptrack as dt

    >>> scatterer = dt.PointParticle()
    >>> optics = dt.Brightfield()
    >>> microscope = dt.Microscope(sample=scatterer, objective=optics)
    >>> image = microscope.get(None)
    >>> print(image.shape)
    (128, 128, 1)

    """

    __distributed__ = False

    def __init__(
        self:  'Microscope',
        sample: Feature,
        objective: Feature,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the `Microscope` instance.

        Parameters
        ----------
        sample: Feature
            A feature-set resolving a list of images describing the sample to be
            imaged.
        objective: Feature
            A feature-set defining the optical device that images the sample.
        **kwargs: Dict[str, Any]
            Additional parameters passed to the base `StructuralFeature` class.

        Attributes
        ----------
        _sample: Feature
            The feature-set defining the sample to be imaged.
        _objective: Feature
            The feature-set defining the optical system imaging the sample.

        """

        super().__init__(**kwargs)
        self._sample = self.add_feature(sample)
        self._objective = self.add_feature(objective)
        self._sample.store_properties()

    def get(
        self: 'Microscope',
        image: Union[Image, None],
        **kwargs:  Dict[str, Any],
    ) -> Image:
        """Generate an image of the sample using the defined optical system.

        This method processes the sample through the optical system to
        produce a simulated image.

        Parameters
        ----------
        image: Union[Image, None]
            The input image to be processed. If None, a new image is created.
        **kwargs: Dict[str, Any]
            Additional parameters for the imaging process.

        Returns
        -------
        Image: Image
            The processed image after applying the optical system.

        Examples
        --------
        Simulating an image with specific parameters:

        >>> import deeptrack as dt
        
        >>> scatterer = dt.PointParticle()
        >>> optics = dt.Brightfield()
        >>> microscope = dt.Microscope(sample=scatterer, objective=optics)
        >>> image = microscope.get(None, upscale=(2, 2, 2))
        >>> print(image.shape)
        (256, 256, 1)

        """

        # Grab properties from the objective to pass to the sample
        additional_sample_kwargs = self._objective.properties()

        # Calculate required output image for the given upscale
        # This way of providing the upscale will be deprecated in the future
        # in favor of dt.Upscale().
        _upscale_given_by_optics = additional_sample_kwargs["upscale"]
        if np.array(_upscale_given_by_optics).size == 1:
            _upscale_given_by_optics = (_upscale_given_by_optics,) * 3

        with u.context(
            create_context(
                *additional_sample_kwargs["voxel_size"], *_upscale_given_by_optics
            )
        ):

            upscale = np.round(get_active_scale())

            output_region = additional_sample_kwargs.pop("output_region")
            additional_sample_kwargs["output_region"] = [
                int(o * upsc)
                for o, upsc in zip(
                    output_region, (upscale[0], upscale[1], upscale[0], upscale[1])
                )
            ]

            padding = additional_sample_kwargs.pop("padding")
            additional_sample_kwargs["padding"] = [
                int(p * upsc)
                for p, upsc in zip(
                    padding, (upscale[0], upscale[1], upscale[0], upscale[1])
                )
            ]

            self._objective.output_region.set_value(
                additional_sample_kwargs["output_region"]
            )
            self._objective.padding.set_value(additional_sample_kwargs["padding"])

            propagate_data_to_dependencies(
                self._sample, **{"return_fft": True, **additional_sample_kwargs}
            )

            list_of_scatterers = self._sample()

            if not isinstance(list_of_scatterers, list):
                list_of_scatterers = [list_of_scatterers]

            # All scatterers that are defined as volumes.
            volume_samples = [
                scatterer
                for scatterer in list_of_scatterers
                if not scatterer.get_property("is_field", default=False)
            ]

            # All scatterers that are defined as fields.
            field_samples = [
                scatterer
                for scatterer in list_of_scatterers
                if scatterer.get_property("is_field", default=False)
            ]

            # Merge all volumes into a single volume.
            sample_volume, limits = _create_volume(
                volume_samples,
                **additional_sample_kwargs,
            )
            sample_volume = Image(sample_volume)

            # Merge all properties into the volume.
            for scatterer in volume_samples + field_samples:
                sample_volume.merge_properties_from(scatterer)

            # Let the objective know about the limits of the volume and all the fields.
            propagate_data_to_dependencies(
                self._objective,
                limits=limits,
                fields=field_samples,
            )

            imaged_sample = self._objective.resolve(sample_volume)

        # Upscale given by the optics needs to be handled separately.
        if _upscale_given_by_optics != (1, 1, 1):
            imaged_sample = AveragePooling((*_upscale_given_by_optics[:2], 1))(
                imaged_sample
            )

        # Merge with input
        if not image:
            if not self._wrap_array_with_image and isinstance(imaged_sample, Image):
                return imaged_sample._value
            else:
                return imaged_sample

        if not isinstance(image, list):
            image = [image]
        for i in range(len(image)):
            image[i].merge_properties_from(imaged_sample)
        return image

    # def _no_wrap_format_input(self, *args, **kwargs) -> list:
    #     return self._image_wrapped_format_input(*args, **kwargs)
    
    # def _no_wrap_process_and_get(self, *args, **feature_input) -> list:
    #     return self._image_wrapped_process_and_get(*args, **feature_input)
    
    # def _no_wrap_process_output(self, *args, **feature_input):
    #     return self._image_wrapped_process_output(*args, **feature_input)


class Optics(Feature):
    """Abstract base optics class.

    Provides structure and methods common for most optical devices. Subclasses
    implement specific optical systems by defining imaging properties and
    behaviors. The `Optics` class is used to define the core imaging properties
    of an optical system, such as resolution, magnification, numerical aperture
    (NA), and wavelength.

    Parameters
    ----------
    NA: float, optional
        Numerical aperture (NA) of the limiting aperture, by default 0.7.
    wavelength: float, optional
        Wavelength of the scattered light in meters, by default 0.66e-6.
    magnification: float, optional
        Magnification of the optical system, by default 10.
    resolution: float or array_like[float], optional
        Distance between pixels in the camera (meters). A third value can 
        define the resolution in the z-direction, by default 1e-6.
    refractive_index_medium: float, optional
        Refractive index of the medium, by default 1.33.
    padding: array_like[int, int, int, int], optional
        Padding applied to the sample volume to avoid edge effects, 
        by default (10, 10, 10, 10).
    output_region: array_like[int, int, int, int], optional
        Region of the image to output (x, y, width, height). If None, the 
        entire image is returned, by default (0, 0, 128, 128).
    pupil: Feature, optional
        Feature-set resolving the pupil function at focus. By default, no pupil
        is applied.
    illumination: Feature, optional
        Feature-set resolving the illumination source. By default, no specific 
        illumination is applied.
    upscale: int, optional
        Scaling factor for the resolution of the optical system, by default 1.
    **kwargs: Dict[str, Any]
        Additional parameters passed to the base `Feature` class.

    Attributes
    ----------
    __conversion_table__: ConversionTable
        Table used to convert properties of the feature to desired units.
    NA: float
        Numerical aperture of the optical system.
    wavelength: float
        Wavelength of the scattered light in meters.
    refractive_index_medium: float
        Refractive index of the medium.
    magnification: float
        Magnification of the optical system.
    resolution: float or array_like[float]
        Pixel spacing in the camera. Optionally includes the z-direction.
    padding: array_like[int]
        Padding applied to the sample volume to reduce edge effects.
    output_region: array_like[int]
        Region of the output image to extract (x, y, width, height).
    voxel_size: function
        Function returning the voxel size of the optical system.
    pixel_size: function
        Function returning the pixel size of the optical system.
    upscale: int
        Scaling factor for the resolution of the optical system.
    limits: array_like[int, int]
        Limits of the volume to be imaged.
    fields: list[Feature]
        List of fields to be imaged.

    Methods
    -------
    `_process_properties(propertydict: Dict[str, Any]) -> Dict[str, Any]`
        Processes and validates the input properties.
    `_pupil(shape:  array_like[int, int], NA: float, wavelength: float, refractive_index_medium: float, include_aberration: bool, defocus: float, **kwargs: Dict[str, Any]) -> array_like[complex]`
        Calculates the pupil function at different focal points.
    `_pad_volume(volume: array_like[complex], limits: array_like[int, int], padding: array_like[int], output_region: array_like[int], **kwargs: Dict[str, Any]) -> tuple`
        Pads the volume with zeros to avoid edge effects.
    `__call__(sample: Feature, **kwargs: Dict[str, Any]) -> Microscope`
        Creates a Microscope instance with the given sample and optics.

    Examples
    --------
    Creating an `Optics` instance:

    >>> import deeptrack as dt

    >>> optics = dt.Optics(NA=0.8, wavelength=0.55e-6, magnification=20)
    >>> print(optics.NA())
    0.8

    """

    __conversion_table__ = ConversionTable(
        wavelength=(u.meter, u.meter),
        resolution=(u.meter, u.meter),
        voxel_size=(u.meter, u.meter),
    )

    def __init__(
        self:  'Optics',
        NA: PropertyLike[float] = 0.7,
        wavelength: PropertyLike[float] = 0.66e-6,
        magnification: PropertyLike[float] = 10,
        resolution: PropertyLike[Union[float, ArrayLike[float]]] = 1e-6,
        refractive_index_medium: PropertyLike[float] = 1.33,
        padding: PropertyLike[ArrayLike[int]] = (10, 10, 10, 10),
        output_region: PropertyLike[ArrayLike[int]] = (0, 0, 128, 128),
        pupil: Feature = None,
        illumination: Feature = None,
        upscale: int = 1,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the `Optics` instance.

        Parameters
        ----------
        NA: float, optional
            Numerical aperture (NA) of the limiting aperture, by default 0.7.
        wavelength: float, optional
            Wavelength of the scattered light in meters, by default 0.66e-6.
        magnification: float, optional
            Magnification of the optical system, by default 10.
        resolution: float or array_like[float], optional
            Distance between pixels in the camera (meters). A third value can
            define the resolution in the z-direction, by default 1e-6.
        refractive_index_medium: float, optional
            Refractive index of the medium, by default 1.33.
        padding: array_like[int, int, int, int], optional
            Padding applied to the sample volume to avoid edge effects,
            by default (10, 10, 10, 10).
        output_region: array_like[int, int, int, int], optional
            Region of the image to output (x, y, width, height). If None, the
            entire image is returned, by default (0, 0, 128, 128).
        pupil: Feature, optional
            Feature-set resolving the pupil function at focus. By default, no pupil
            is applied.
        illumination: Feature, optional
            Feature-set resolving the illumination source. By default, no specific
            illumination is applied.
        upscale: int, optional
            Scaling factor for the resolution of the optical system, by default 1.
        **kwargs: Dict[str, Any]
            Additional parameters passed to the base `Feature` class.

        Attributes
        ----------
        NA: float
            Numerical aperture of the optical system.
        wavelength: float
            Wavelength of the scattered light in meters.
        refractive_index_medium: float
            Refractive index of the medium.
        magnification: float
            Magnification of the optical system.
        resolution: float or array_like[float]
            Pixel spacing of the camera in meters. Optionally includes the 
            z-direction.
        padding: array_like[int]
            Padding applied to the sample volume to reduce edge effects.
        output_region: array_like[int]
            Region of the output image to extract (x, y, width, height).
        voxel_size: function
            Function returning the voxel size of the optical system.
        pixel_size: function
            Function returning the pixel size of the optical system.
        upscale: int
            Scaling factor for the resolution of the optical system.
        limits: array_like[int, int]
            Limits of the volume to be imaged.
        fields: list[Feature]
            List of fields to be imaged.

        Helper Functions
        ----------------
        `get_voxel_size(resolution: float or array_like[float], magnification: float) -> array_like[float]`
            Calculate the voxel size.
        `get_pixel_size(resolution: float or array_like[float], magnification: float) -> float`
            Calculate the pixel size.

        """

        def get_voxel_size(
            resolution: Union[float, ArrayLike[float]], 
            magnification: float,
        ) -> ArrayLike[float]:
            """ Calculate the voxel size.
            
            Parameters
            ----------
            resolution: float or array_like[float]
                The distance between pixels of the camera in meters. A third 
                value can define the resolution in the z-direction.
            magnification: float
                The magnification of the optical system.

            Returns
            -------
            array_like[float]
                The voxel size of the optical system.

            """

            props = self._normalize(resolution=resolution, magnification=magnification)
            return np.ones((3,)) * props["resolution"] / props["magnification"]

        def get_pixel_size(
            resolution: Union[float, ArrayLike[float]],
            magnification: float,
        ) -> float:
            """ Calculate the pixel size.

            It differs from the voxel size by only being a single value.

            Parameters
            ----------
            resolution: float or array_like[float]
                The distance between pixels in the camera. A third value can
                define the resolution in the z-direction.
            magnification: float
                The magnification of the optical system.

            Returns
            -------
            float
                The pixel size of the optical system.
    
            """
            
            props = self._normalize(
                resolution=resolution, 
                magnification=magnification,
            )
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
            upscale=upscale,
            limits=None,
            fields=None,
            **kwargs,
        )

        self.pupil = self.add_feature(pupil) if pupil else DummyFeature()
        self.illumination = (
            self.add_feature(illumination) if illumination else DummyFeature()
        )

    def _process_properties(
        self:   'Optics',
        propertydict:   Dict[str, Any],
    ) -> Dict[str, Any]:
        """Processes and validates the input properties.

        Ensures that the provided optical parameters are reasonable.

        Parameters
        ----------
        propertydict:  Dict[str, Any]
            The input properties.

        Returns
        -------
        dict: Dict[str, Any]
            The processed properties.
        
        """
        
        propertydict = super()._process_properties(propertydict)

        NA = propertydict["NA"]
        wavelength = propertydict["wavelength"]
        voxel_size = get_active_voxel_size()
        radius = NA / wavelength * np.array(voxel_size)

        if np.any(radius[:2] > 0.5):
            required_upscale = np.max(np.ceil(radius[:2] * 2))
            warnings.warn(
                f"""Likely bad optical parameters. NA / wavelength * 
                resolution / magnification = {radius} should be at most 0.5. 
                To fix, set magnification to {required_upscale}, and downsample
                the resulting image with 
                dt.AveragePooling(({required_upscale}, {required_upscale}, 1))
                """
            )

        return propertydict

    def _pupil(
        self:  'Optics',
        shape: ArrayLike[int],
        NA: float,
        wavelength: float,
        refractive_index_medium: float,
        include_aberration: bool = True,   
        defocus: Union[float, ArrayLike[float]] = 0,
        **kwargs: Dict[str, Any],
    ):
        """Calculates the pupil function at different focal points.

        Parameters
        ----------
        shape: array_like[int, int]
            The shape of the pupil function.
        NA: float
            The NA of the limiting aperture.
        wavelength: float
            The wavelength of the scattered light in meters.
        refractive_index_medium: float
            The refractive index of the medium.
        voxel_size: array_like[float (, float, float)]
            The distance between pixels in the camera. A third value can be
            included to define the resolution in the z-direction.
        include_aberration: bool
            If True, the aberration is included in the pupil function.
        defocus: float or list[float]
            The defocus of the system. If a list is given, the pupil is
            calculated for each focal point. Defocus is given in meters.

        Returns
        -------
        pupil: array_like[complex]
            The pupil function. Shape is (z, y, x).

        Examples
        --------
        Calculating the pupil function:

        >>> import deeptrack as dt

        >>> optics = dt.Optics()
        >>> pupil = optics._pupil(
        ...     shape=(128, 128),
        ...     NA=0.8,
        ...     wavelength=0.55e-6,
        ...     refractive_index_medium=1.33,
        ... )
        >>> print(pupil.shape)
        (1, 128, 128)
        
        """

        # Calculates the pupil at each z-position in defocus.
        voxel_size = get_active_voxel_size()
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
        RHO = (W ** 2 + H ** 2).astype(complex)
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

        z_shift._value[z_shift._value.imag != 0] = 0

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
        self:   'Optics',
        volume: ArrayLike[complex],
        limits: ArrayLike[int] = None,
        padding: ArrayLike[int] = None,
        output_region: ArrayLike[int] = None,
        **kwargs: Dict[str, Any],
    ) -> tuple:
        """Pads the volume with zeros to avoid edge effects.

        Parameters
        ----------
        volume: array_like[complex]
            The volume to pad.
        limits: array_like[int, int]
            The limits of the volume.
        padding: array_like[int]
            The padding to apply. Format is (left, right, top, bottom).
        output_region: array_like[int, int]
            The region of the volume to return. Used to remove regions of the
            volume that are far outside the view. If None, the full volume is
            returned.

        Returns
        -------
        new_volume: array_like[complex]
            The padded volume.
        new_limits: array_like[int, int]
            The new limits of the volume.

        Examples
        --------
        Padding a volume:

        >>> import deeptrack as dt
        >>> import numpy as np

        >>> volume = np.ones((10, 10, 10), dtype=complex)
        >>> limits = np.array([[0, 10], [0, 10], [0, 10]])
        >>> optics = dt.Optics()
        >>> padded_volume, new_limits = optics._pad_volume(
        ...     volume, limits=limits, padding=[5, 5, 5, 5],
        ...     output_region=[0, 0, 10, 10],
        ... )
        >>> print(padded_volume.shape)
        (20, 20, 10)
        >>> print(new_limits)
        [[-5 15]
         [-5 15]
         [ 0 10]]
        
        """
        
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
                np.min([new_limits[i, 0], output_region[i] - padding[i]]),
                np.max(
                    [
                        new_limits[i, 1],
                        output_region[i + 2] + padding[i + 2],
                    ]
                ),
            )
        new_volume = np.zeros(
            np.diff(new_limits, axis=1)[:, 0].astype(np.int32),
            dtype=complex,
        )

        old_region = (limits - new_limits).astype(np.int32)
        limits = limits.astype(np.int32)
        new_volume[
            old_region[0, 0] : old_region[0, 0] + limits[0, 1] - limits[0, 0],
            old_region[1, 0] : old_region[1, 0] + limits[1, 1] - limits[1, 0],
            old_region[2, 0] : old_region[2, 0] + limits[2, 1] - limits[2, 0],
        ] = volume
        return new_volume, new_limits

    def __call__(
        self:  'Optics',
        sample: Feature,
        **kwargs: Dict[str, Any],
    ) -> Microscope:
        """Creates a Microscope instance with the given sample and optics.

        Parameters
        ----------
        sample: Feature
            The sample to be imaged.
        **kwargs: Dict[str, Any]
            Additional parameters for the Microscope.

        Returns
        -------
        Microscope: Microscope
            A Microscope instance configured with the sample and optics.

        Examples
        --------
        Creating a Microscope instance:

        >>> import deeptrack as dt

        >>> scatterer = dt.PointParticle()
        >>> optics = dt.Optics()
        >>> microscope = optics(scatterer)
        >>> print(isinstance(microscope, dt.Microscope))
        True

        """
        
        return Microscope(sample, self, **kwargs)

    # def _no_wrap_format_input(self, *args, **kwargs) -> list:
    #     return self._image_wrapped_format_input(*args, **kwargs)
    
    # def _no_wrap_process_and_get(self, *args, **feature_input) -> list:
    #     return self._image_wrapped_process_and_get(*args, **feature_input)
    
    # def _no_wrap_process_output(self, *args, **feature_input):
    #     return self._image_wrapped_process_output(*args, **feature_input)


class Fluorescence(Optics):
    """Optical device for fluorescent imaging.

    The `Fluorescence` class simulates the imaging process in fluorescence
    microscopy by creating a discretized volume where each pixel represents 
    the intensity of light emitted by fluorophores in the sample. It extends 
    the `Optics` class to include fluorescence-specific functionalities.

    Parameters
    ----------
    NA: float
        Numerical aperture of the optical system.
    wavelength: float
        Emission wavelength of the fluorescent light (in meters).
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. Optionally includes the z-direction.
    refractive_index_medium: float
        Refractive index of the imaging medium.
    padding: array_like[int, int, int, int]
        Padding applied to the sample volume to reduce edge effects.
    output_region: array_like[int, int, int, int], optional
        Region of the output image to extract (x, y, width, height). If None, 
        returns the full image.
    pupil: Feature, optional
        A feature set defining the pupil function at focus. The input is 
        the unaberrated pupil.
    illumination: Feature, optional
        A feature set defining the illumination source.
    upscale: int, optional
        Scaling factor for the resolution of the optical system.
    **kwargs: Dict[str, Any]

    Attributes
    ----------
    __gpu_compatible__: bool
        Indicates whether the class supports GPU acceleration.
    NA: float
        Numerical aperture of the optical system.
    wavelength: float
        Emission wavelength of the fluorescent light (in meters).
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. Optionally includes the z-direction.
    refractive_index_medium: float
        Refractive index of the imaging medium.
    padding: array_like[int, int, int, int]
        Padding applied to the sample volume to reduce edge effects.
    output_region: array_like[int, int, int, int]
        Region of the output image to extract (x, y, width, height).
    voxel_size: function
        Function returning the voxel size of the optical system.
    pixel_size: function
        Function returning the pixel size of the optical system.
    upscale: int
        Scaling factor for the resolution of the optical system.
    limits: array_like[int, int]
        Limits of the volume to be imaged.
    fields: list[Feature]
        List of fields to be imaged

    Methods
    -------
    `get(illuminated_volume: array_like[complex], limits: array_like[int, int], **kwargs: Dict[str, Any]) -> Image`
        Simulates the imaging process using a fluorescence microscope.

    Examples
    --------
    Create a `Fluorescence` instance:

    >>> import deeptrack as dt

    >>> optics = dt.Fluorescence(
    ...     NA=1.4, wavelength=0.52e-6, magnification=60,
    ... )
    >>> print(optics.NA())
    1.4

    """

    __gpu_compatible__ = True

    def get(
        self:  'Fluorescence', 
        illuminated_volume: ArrayLike[complex], 
        limits: ArrayLike[int], 
        **kwargs: Dict[str, Any]
    ) -> Image:
        """Simulates the imaging process using a fluorescence microscope.

        This method convolves the 3D illuminated volume with a pupil function 
        to generate a 2D image projection.

        Parameters
        ----------
        illuminated_volume: array_like[complex]
            The illuminated 3D volume to be imaged.
        limits: array_like[int, int]
            Boundaries of the illuminated volume in each dimension.
        **kwargs: Dict[str, Any]
            Additional properties for the imaging process, such as:
            - 'padding': Padding to apply to the sample.
            - 'output_region': Specific region to extract from the image.

        Returns
        -------
        Image: Image
            A 2D image object representing the fluorescence projection.

        Notes
        -----
        - Empty slices in the volume are skipped for performance optimization.
        - The pupil function incorporates defocus effects based on z-slice.

        Examples
        --------
        Simulate imaging a volume:

        >>> import deeptrack as dt
        >>> import numpy as np

        >>> optics = dt.Fluorescence(
        ...     NA=1.4, wavelength=0.52e-6, magnification=60,
        ... )
        >>> volume = dt.Image(np.ones((128, 128, 10), dtype=complex))
        >>> limits = np.array([[0, 128], [0, 128], [0, 10]])
        >>> properties = optics.properties()
        >>> filtered_properties = {
        ...     k: v for k, v in properties.items() 
        ...     if k in {"padding", "output_region", "NA", 
        ...              "wavelength", "refractive_index_medium"}
        ... }
        >>> image = optics.get(volume, limits, **filtered_properties)
        >>> print(image.shape)
        (128, 128, 1)
        
        """

        # Pad volume
        padded_volume, limits = self._pad_volume(
            illuminated_volume, limits=limits, **kwargs
        )

        # Extract indexes of the output region
        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))

        # Calculate the how much to crop from the volume
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

        # Find planes that are not empty for optimization
        z_iterator = np.linspace(
            z_limits[0],
            z_limits[1],
            num=padded_volume.shape[2],
            endpoint=False,
        )
        zero_plane = np.all(padded_volume == 0, axis=(0, 1), keepdims=False)
        z_values = z_iterator[~zero_plane]

        # Further pad image to speed up fft (multiples of 2 and 3)
        volume = maybe_cupy(pad_image_to_fft(padded_volume, axes=(0, 1)))
        pupils = self._pupil(volume.shape[:2], defocus=z_values, **kwargs)

        z_index = 0

        # Loop through volume and convolve sample with pupil function
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
    """Simulates imaging of coherently illuminated samples.

    The `Brightfield` class models a brightfield microscopy setup, imaging 
    samples by iteratively propagating light through a discretized volume.
    Each voxel in the volume represents the effective refractive index 
    of the sample at that point. Light is propagated iteratively through 
    Fourier space and corrected in real space.

    Parameters
    ----------
    illumination: Feature, optional
        Feature-set representing the complex field entering the sample. 
        Default is a uniform field with all values set to 1.
    NA: float
        Numerical aperture of the limiting aperture.
    wavelength: float
        Wavelength of the incident light in meters.
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. A third value can define the 
        resolution in the z-direction.
    refractive_index_medium: float
        Refractive index of the medium.
    padding: array_like[int, int, int, int]
        Padding added to the sample volume to minimize edge effects.
    output_region: array_like[int, int, int, int], optional
        Specifies the region of the image to output (x, y, width, height).
        Default is None, which outputs the entire image.
    pupil: Feature, optional
        Feature-set defining the pupil function. The input is the 
        unaberrated pupil.

    Attributes
    ----------
    __gpu_compatible__: bool
        Indicates whether the class supports GPU acceleration.
    __conversion_table__: ConversionTable
        Table used to convert properties of the feature to desired units.
    NA: float
        Numerical aperture of the optical system.
    wavelength: float
        Wavelength of the scattered light in meters.
    magnification: float
        Magnification of the optical system.
    resolution: array_like[float (, float, float)]
        Pixel spacing in the camera. Optionally includes the z-direction.
    refractive_index_medium: float
        Refractive index of the medium.
    padding: array_like[int, int, int, int]
        Padding applied to the sample volume to reduce edge effects.
    output_region: array_like[int, int, int, int]
        Region of the output image to extract (x, y, width, height).
    voxel_size: function
        Function returning the voxel size of the optical system.
    pixel_size: function
        Function returning the pixel size of the optical system.
    upscale: int
        Scaling factor for the resolution of the optical system.
    limits: array_like[int, int]
        Limits of the volume to be imaged.
    fields: list[Feature]
        List of fields to be imaged.

    Methods
    -------
    `get(illuminated_volume: array_like[complex], 
        limits: array_like[int, int], fields: array_like[complex], 
        **kwargs: Dict[str, Any]) -> Image`
        Simulates imaging with brightfield microscopy.


    Examples
    --------
    Create a `Brightfield` instance:

    >>> import deeptrack as dt

    >>> optics = dt.Brightfield(NA=1.4, wavelength=0.52e-6, magnification=60)
    >>> print(optics.NA())
    1.4
    
    """

    __gpu_compatible__ = True

    __conversion_table__ = ConversionTable(
        working_distance=(u.meter, u.meter),
    )

    def get(
        self:  'Brightfield',
        illuminated_volume: ArrayLike[complex],
        limits: ArrayLike[int],
        fields: ArrayLike[complex],
        **kwargs: Dict[str, Any],
    ) -> Image:
        """Simulates imaging with brightfield microscopy.

        This method propagates light through the given volume, applying 
        pupil functions at various defocus levels and incorporating 
        refraction corrections in real space to produce the final 
        brightfield image.

        Parameters
        ----------
        illuminated_volume: array_like[complex]
            Discretized volume representing the sample to be imaged.
        limits: array_like[int, int]
            Boundaries of the sample volume in each dimension.
        fields: array_like[complex]
            Input fields to be used in the imaging process.
        **kwargs: Dict[str, Any]
            Additional parameters for the imaging process, including:
            - 'padding': Padding to apply to the sample volume.
            - 'output_region': Specific region to extract from the image.
            - 'wavelength': Wavelength of the light.
            - 'refractive_index_medium': Refractive index of the medium.

        Returns
        -------
        Image: Image
            Processed image after simulating the brightfield imaging process.

        Examples
        --------
        Simulate imaging a volume:

        >>> import deeptrack as dt
        >>> import numpy as np

        >>> optics = dt.Brightfield(
        ...     NA=1.4, 
        ...     wavelength=0.52e-6, 
        ...     magnification=60,
        ... )
        >>> volume = dt.Image(np.ones((128, 128, 10), dtype=complex))
        >>> limits = np.array([[0, 128], [0, 128], [0, 10]])
        >>> fields = np.array([np.ones((162, 162), dtype=complex)])
        >>> properties = optics.properties()
        >>> filtered_properties = {
        ...     k: v for k, v in properties.items()
        ...     if k in {'padding', 'output_region', 'NA', 
        ...              'wavelength', 'refractive_index_medium'}
        ... }
        >>> image = optics.get(volume, limits, fields, **filtered_properties)
        >>> print(image.shape)
        (128, 128, 1)
        
        """

        # Pad volume
        padded_volume, limits = self._pad_volume(
            illuminated_volume, limits=limits, **kwargs
        )

        # Extract indexes of the output region
        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(
            kwargs.get("output_region", (None, None, None, None))
        )
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

        output_image = Image(image.maybe_cupy(
            np.zeros((*padded_volume.shape[0:2], 1))
            ))

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

        voxel_size = get_active_voxel_size()

        pupils = [
            self._pupil(
                volume.shape[:2], defocus=[1], include_aberration=False, **kwargs
            )[0],
            self._pupil(
                volume.shape[:2],
                defocus=[-z_limits[1]],
                include_aberration=True,
                **kwargs,
            )[0],
            self._pupil(
                volume.shape[:2],
                defocus=[0],
                include_aberration=True,
                **kwargs,
            )[0]
        ]

        pupil_step = np.fft.fftshift(pupils[0])

        light_in = image.maybe_cupy(np.ones(volume.shape[:2], dtype=complex))
        light_in = self.illumination.resolve(light_in)
        light_in = np.fft.fft2(light_in)

        K = 2 * np.pi / kwargs["wavelength"]*kwargs["refractive_index_medium"]

        z = z_limits[1]
        for i, z in zip(index_iterator, z_iterator):
            light_in = light_in * pupil_step

            if zero_plane[i]:
                continue

            ri_slice = volume[:, :, i]
            light = np.fft.ifft2(light_in)
            light_out = light * np.exp(1j * ri_slice * voxel_size[-1] * K)
            light_in = np.fft.fft2(light_out)
  
        shifted_pupil = np.fft.fftshift(pupils[1])
        light_in_focus = light_in * shifted_pupil

        if len(fields) > 0:
            field = np.sum(fields, axis=0)
            light_in_focus += field[..., 0]
        shifted_pupil = np.fft.fftshift(pupils[-1])
        light_in_focus = light_in_focus * shifted_pupil
        # Mask to remove light outside the pupil.
        mask = np.abs(shifted_pupil) > 0
        light_in_focus = light_in_focus * mask

        output_image = np.fft.ifft2(light_in_focus)[
            : padded_volume.shape[0], : padded_volume.shape[1]
        ]
        output_image = np.expand_dims(output_image, axis=-1)
        output_image = Image(output_image[pad[0] : -pad[2], pad[1] : -pad[3]])

        if not kwargs.get("return_field", False):
            output_image = np.square(np.abs(output_image))
        # else:
        # Fudge factor. Not sure why this is needed.
        # output_image = output_image - 1
        # output_image = output_image * np.exp(1j * -np.pi / 4)
        # output_image = output_image + 1

        output_image.properties = illuminated_volume.properties

        return output_image


class Holography(Brightfield):
    """An alias for the Brightfield class, representing holographic 
    imaging setups.

    Holography shares the same implementation as Brightfield, as both use 
    coherent illumination and similar propagation techniques.

    """
    pass


class ISCAT(Brightfield):
    """Images coherently illuminated samples using Interferometric Scattering 
    (ISCAT) microscopy.

    This class models ISCAT by creating a discretized volume where each pixel
    represents the effective refractive index of the sample. Light is 
    propagated through the sample iteratively, first in the Fourier space 
    and then corrected in the real space for refractive index.

    Parameters
    ----------
    illumination: Feature
        Feature-set defining the complex field entering the sample. Default 
        is a field with all values set to 1.
    NA: float
        Numerical aperture (NA) of the limiting aperture.
    wavelength: float
        Wavelength of the scattered light, in meters.
    magnification: float
        Magnification factor of the optical system.
    resolution: array_like of float
        Pixel spacing in the camera. Optionally includes a third value for 
        z-direction resolution.
    refractive_index_medium: float
        Refractive index of the medium surrounding the sample.
    padding: array_like of int
        Padding for the sample volume to minimize edge effects. Format: 
        (left, right, top, bottom).
    output_region: array_like of int
        Region of the image to output as (x, y, width, height). If None 
        (default), the entire image is returned.
    pupil: Feature
        Feature-set defining the pupil function at focus. The feature-set 
        takes an unaberrated pupil as input.
    illumination_angle: float, optional
        Angle of illumination relative to the optical axis, in radians. 
        Default is π radians.
    amp_factor: float, optional
        Amplitude factor of the illuminating field relative to the reference 
        field. Default is 1.

    Attributes
    ----------
    illumination_angle: float
        The angle of illumination, stored for reference.
    amp_factor: float
        Amplitude factor of the illuminating field.

    Examples
    --------
    Creating an ISCAT instance:
    
    >>> import deeptrack as dt

    >>> iscat = dt.ISCAT(NA=1.4, wavelength=0.532e-6, magnification=60)
    >>> print(iscat.illumination_angle())
    3.141592653589793
    
    """

    def __init__(
        self:  'ISCAT',
        illumination_angle: float = np.pi,
        amp_factor: float = 1, 
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the ISCAT class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        amp_factor: float
            Amplitude factor of the illuminating field relative to the reference 
            field.
        **kwargs: Dict[str, Any]
            Additional parameters for the Brightfield class.

        """

        super().__init__(
            illumination_angle=illumination_angle,
            amp_factor=amp_factor,
            input_polarization="circular",
            output_polarization="circular",
            phase_shift_correction=True,
            **kwargs
            )
        
class Darkfield(Brightfield):
    """Images coherently illuminated samples using Darkfield microscopy.

    This class models Darkfield microscopy by creating a discretized volume 
    where each pixel represents the effective refractive index of the sample. 
    Light is propagated through the sample iteratively, first in the Fourier 
    space and then corrected in the real space for refractive index.

    Parameters
    ----------
    illumination: Feature
        Feature-set defining the complex field entering the sample. Default 
        is a field with all values set to 1.
    NA: float
        Numerical aperture (NA) of the limiting aperture.
    wavelength: float
        Wavelength of the scattered light, in meters.
    magnification: float
        Magnification factor of the optical system.
    resolution: array_like of float
        Pixel spacing in the camera. Optionally includes a third value for 
        z-direction resolution.
    refractive_index_medium: float
        Refractive index of the medium surrounding the sample.
    padding: array_like of int
        Padding for the sample volume to minimize edge effects. Format: 
        (left, right, top, bottom).
    output_region: array_like of int
        Region of the image to output as (x, y, width, height). If None 
        (default), the entire image is returned.
    pupil: Feature
        Feature-set defining the pupil function at focus. The feature-set 
        takes an unaberrated pupil as input.
    illumination_angle: float, optional
        Angle of illumination relative to the optical axis, in radians. 
        Default is π/2 radians.

    Attributes
    ----------
    illumination_angle: float
        The angle of illumination, stored for reference.

    Methods
    -------
    get(illuminated_volume, limits, fields, **kwargs)
        Retrieves the darkfield image of the illuminated volume.

    Examples
    --------
    Creating a Darkfield instance:

    >>> import deeptrack as dt

    >>> darkfield = dt.Darkfield(NA=0.9, wavelength=0.532e-6)
    >>> print(darkfield.illumination_angle())
    1.5707963267948966

    """

    def __init__(
        self: 'Darkfield', 
        illumination_angle: float = np.pi/2, 
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the Darkfield class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        **kwargs: Dict[str, Any]
            Additional parameters for the Brightfield class.

        """

        super().__init__(
            illumination_angle=illumination_angle,
            **kwargs)

    #Retrieve get as super
    def get(
        self: 'Darkfield',
        illuminated_volume: ArrayLike[complex],
        limits: ArrayLike[int],
        fields: ArrayLike[complex],
        **kwargs: Dict[str, Any],
    ) -> Image:
        """Retrieve the darkfield image of the illuminated volume.

        Parameters
        ----------
        illuminated_volume: array_like
            The volume of the sample being illuminated.
        limits: array_like
            The spatial limits of the volume.
        fields: array_like
            The fields interacting with the sample.
        **kwargs: Dict[str, Any]
            Additional parameters passed to the super class's get method.

        Returns
        -------
        numpy.ndarray
            The darkfield image obtained by calculating the squared absolute
            difference from 1.
        
        """

        field = super().get(illuminated_volume, limits, fields, return_field=True, **kwargs)
        return np.square(np.abs(field-1))


class IlluminationGradient(Feature):
    """
    Adds a gradient to the illumination of the sample.

    This class modifies the amplitude of the field by adding a planar gradient
    and a constant offset. The amplitude is clipped within the specified 
    bounds.

    Parameters
    ----------
    gradient: array_like of float, optional
        Gradient of the plane to add to the field amplitude, specified in 
        pixels. Default is (0, 0).
    constant: float, optional
        Constant value to add to the field amplitude. Default is 0.
    vmin: float, optional
        Minimum allowed value for the amplitude. Values below this are clipped. 
        Default is 0.
    vmax: float, optional
        Maximum allowed value for the amplitude. Values above this are clipped. 
        Default is infinity.

    Attributes
    ----------
    gradient: array_like of float
        Gradient of the plane to add to the field amplitude.
    constant: float
        Constant value to add to the field amplitude.
    vmin: float
        Minimum allowed value for the amplitude.
    vmax: float
        Maximum allowed value for the amplitude.

    Methods
    -------
    get(image, gradient, constant, vmin, vmax, **kwargs)
        Applies the gradient and constant offset to the amplitude of the field.

    Examples
    --------
    Adding a gradient to the illumination:

    >>> gradient_feature = dt.IlluminationGradient(gradient=(0.1, 0.2))
    >>> print(gradient_feature.properties['gradient']())
    (0.1, 0.2)

    """

    def __init__(
        self: 'IlluminationGradient',
        gradient: PropertyLike[ArrayLike[float]] = (0, 0),
        constant: PropertyLike[float] = 0,
        vmin: PropertyLike[float] = 0,
        vmax: PropertyLike[float] = np.inf,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initializes the IlluminationGradient class.

        Parameters
        ----------
        gradient: array_like of float, optional
            Gradient of the plane to add to the field amplitude, specified in 
            pixels. Default is (0, 0).
        constant: float, optional
            Constant value to add to the field amplitude. Default is 0.
        vmin: float, optional
            Minimum allowed value for the amplitude. Values below this are 
            clipped. Default is 0.
        vmax: float, optional
            Maximum allowed value for the amplitude. Values above this are 
            clipped. Default is infinity.
        **kwargs: Dict[str, Any]
            Additional parameters for customization.

        """

        super().__init__(
            gradient=gradient, constant=constant, vmin=vmin, vmax=vmax, **kwargs
        )

    def get(
        self: 'IlluminationGradient',
        image: ArrayLike[complex],
        gradient: ArrayLike[float],
        constant: float,
        vmin: float,
        vmax: float,
        **kwargs: Dict[str, Any],
    ) -> ArrayLike[complex]:
        """Applies the gradient and constant offset to the amplitude of the 
        field.

        Parameters
        ----------
        image: numpy.ndarray
            The input field to which the gradient and constant are applied.
        gradient: array_like of float
            Gradient of the plane to add to the field amplitude.
        constant: float
            Constant value to add to the field amplitude.
        vmin: float
            Minimum value for clipping the amplitude.
        vmax: float
            Maximum value for clipping the amplitude.
        **kwargs: Dict[str, Any]
            Additional parameters for customization.

        Returns
        -------
        numpy.ndarray
            The modified field with the gradient and constant applied.

        Examples
        --------
        >>> import deeptrack as dt

        >>> image=np.ones((100, 100))
        >>> gradient_feature = dt.IlluminationGradient(gradient=(0.3, 0.1))
        >>> properties_dict = gradient_feature.properties()
        >>> modified_image = gradient_feature.get(image, **properties_dict)
        >>> print(modified_image.shape)
        (100, 100)
        
        """
        
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


def _get_position(
    image:  Image,
    mode: str = "corner",
    return_z: bool = False,
) -> np.ndarray:
    """Extracts the position of the upper-left corner of a scatterer.

    Parameters
    ----------
    image: numpy.ndarray
        Input image or volume containing the scatterer.
    mode: str, optional
        Mode for position extraction. Default is "corner".
    return_z: bool, optional
        Whether to include the z-coordinate in the output. Default is False.

    Returns
    -------
    numpy.ndarray
        Array containing the position of the scatterer.
    
    """

    num_outputs = 2 + return_z

    if mode == "corner" and image.size > 0:
        import scipy.ndimage

        image = image.to_numpy()

        shift = scipy.ndimage.center_of_mass(np.abs(image))

        if np.isnan(shift).any():
            shift = np.array(image.shape) / 2

    else:
        shift = np.zeros((num_outputs))

    position = np.array(image.get_property("position", default=None))

    if position is None:
        return position

    scale = np.array(get_active_scale())

    if len(position) == 3:
        position = position * scale + 0.5 * (scale - 1)
        if return_z:
            return position * scale - shift
        else:
            return position[0:2] - shift[0:2]

    elif len(position) == 2:
        if return_z:
            outp = (
                np.array([position[0], position[1], image.get_property("z", default=0)])
                * scale
                - shift
                + 0.5 * (scale - 1)
            )
            return outp
        else:
            return position * scale[:2] - shift[0:2] + 0.5 * (scale[:2] - 1)

    return position


def _create_volume(
    list_of_scatterers: list,
    pad: tuple = (0, 0, 0, 0),
    output_region: tuple = (None, None, None, None),
    refractive_index_medium: float = 1.33,
    **kwargs: Dict[str, Any],
) -> tuple:
    """Converts a list of scatterers into a volumetric representation.

    Parameters
    ----------
    list_of_scatterers: list or single scatterer
        List of scatterers to include in the volume.
    pad: tuple of int, optional
        Padding for the volume in the format (left, right, top, bottom).
        Default is (0, 0, 0, 0).
    output_region: tuple of int, optional
        Region to output, defined as (x_min, y_min, x_max, y_max). Default is 
        None.
    refractive_index_medium: float, optional
        Refractive index of the medium surrounding the scatterers. Default is 
        1.33.
    **kwargs: Dict[str, Any]
        Additional arguments for customization.

    Returns
    -------
    tuple
        - volume: numpy.ndarray
            The generated volume containing the scatterers.
        - limits: numpy.ndarray
            Spatial limits of the volume.

    """

    if not isinstance(list_of_scatterers, list):
        list_of_scatterers = [list_of_scatterers]

    volume = np.zeros((1, 1, 1), dtype=complex)
    limits = None
    OR = np.zeros((4,))
    OR[0] = np.inf if output_region[0] is None else int(
        output_region[0] - pad[0]
    )
    OR[1] = -np.inf if output_region[1] is None else int(
        output_region[1] - pad[1]
    )
    OR[2] = np.inf if output_region[2] is None else int(
        output_region[2] + pad[2]
    )
    OR[3] = -np.inf if output_region[3] is None else int(
        output_region[3] + pad[3]
    )

    scale = np.array(get_active_scale())

    # This accounts for upscale doing AveragePool instead of SumPool. This is
    # a bit of a hack, but it works for now.
    fudge_factor = scale[0] * scale[1] / scale[2]

    for scatterer in list_of_scatterers:

        position = _get_position(scatterer, mode="corner", return_z=True)

        if scatterer.get_property("intensity", None) is not None:
            intensity = scatterer.get_property("intensity")
            scatterer_value = intensity * fudge_factor
        elif scatterer.get_property("refractive_index", None) is not None:
            refractive_index = scatterer.get_property("refractive_index")
            scatterer_value = (
                refractive_index - refractive_index_medium
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
                "Optical device received an image without a position property."
                " It will be ignored."
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
            if splined_scatterer.dtype == complex:
                splined_scatterer[:, :, z] = (
                    convolve(
                        np.real(scatterer[:, :, z]), kernel, mode="constant"
                    )
                    + convolve(
                        np.imag(scatterer[:, :, z]), kernel, mode="constant"
                    )
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
                dtype=complex,
            )
            old_region = (limits - new_limits).astype(np.int32)
            limits = limits.astype(np.int32)
            new_volume[
                old_region[0, 0] : 
                old_region[0, 0] + limits[0, 1] - limits[0, 0],
                old_region[1, 0] : 
                old_region[1, 0] + limits[1, 1] - limits[1, 0],
                old_region[2, 0] : 
                old_region[2, 0] + limits[2, 1] - limits[2, 0],
            ] = volume
            volume = new_volume
            limits = new_limits

        within_volume_position = position - limits[:, 0]

        # NOTE: Maybe shouldn't be additive.
        volume[
            int(within_volume_position[0]) : 
            int(within_volume_position[0] + shape[0]),
            
            int(within_volume_position[1]) : 
            int(within_volume_position[1] + shape[1]),

            int(within_volume_position[2]) : 
            int(within_volume_position[2] + shape[2]),
        ] += scatterer
    return volume, limits