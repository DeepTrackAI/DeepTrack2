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

- **Integration with feature pipelines**

  Full compatibility with feature pipelines, allows for dynamic and complex
  simulations, incorporating physics-based models and real-time adjustments to
  sample and imaging properties.

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
    ) -> tuple[int, int, Optional[int]]

    Extracts the position of the upper-left corner of a scatterer in the image.

- `_create_volume(list_of_scatterers:, pad, output_region, refractive_index_medium, **kwargs)`

    def _create_volume(
        list_of_scatterers: list[np.ndarray],
        pad: int,
        output_region: tuple[int, int, int, int],
        refractive_index_medium: float,
        **kwargs: Any,
    ) -> np.ndarray

    Combines multiple scatterer objects into a single 3D volume for imaging.

- `_pad_volume(volume, limits, padding, output_region, **kwargs)`

    def _pad_volume(
        volume: np.ndarray,
        limits: np.ndarray,
        padding: tuple[int, int, int, int],
        output_region: tuple[int, int, int, int],
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]

    Pads a volume with zeros to avoid edge effects during imaging.

Examples
--------
>>> import deeptrack as dt

Simulating an image with the `Brightfield` class:
>>> scatterer = dt.PointParticle()
>>> optics = dt.Brightfield()
>>> image = optics(scatterer)
>>> image().shape
(128, 128, 1)

>>> image.plot(cmap="gray")

Simulating an image with the `Fluorescence` class:
>>> scatterer = dt.PointParticle()
>>> optics = dt.Fluorescence()
>>> image = optics(scatterer)
>>> image().shape
(128, 128, 1)

>>> image.plot(cmap="gray")

"""

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT323
#TODO ***??*** polish imports

from __future__ import annotations

from pint import Quantity
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
from scipy.ndimage import convolve # might be removed later
import torch
import torch.nn.functional as F

from deeptrack.backend.units import (
    ConversionTable,
    create_context,
    get_active_scale,
    get_active_voxel_size,
)
from deeptrack.math import AveragePooling, SumPooling
from deeptrack.features import propagate_data_to_dependencies
from deeptrack.features import DummyFeature, Feature, StructuralFeature
from deeptrack.image import pad_image_to_fft
from deeptrack.types import ArrayLike, PropertyLike

from deeptrack import image
from deeptrack import units_registry as u

from deeptrack import TORCH_AVAILABLE, image
from deeptrack.backend import xp
from deeptrack.scatterers import ScatteredVolume, ScatteredField

if TORCH_AVAILABLE:
    import torch

if TYPE_CHECKING:
    import torch


#TODO ***??*** revise Microscope - torch, typing, docstring, unit test
class Microscope(StructuralFeature):
    """Simulates imaging of a sample using an optical system.

    This class combines the sample to be imaged with the optical system, 
    enabling the simulation of optical imaging processes.
    A Microscope:
    - validates the semantic compatibility between scatterers and optics
    - interprets volume-based scatterers into scalar fields when needed
    - delegates numerical propagation to the objective (Optics)
    - performs detector downscaling according to its physical semantics

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
    `get(image: np.ndarray or None, **kwargs: Any) -> np.ndarray`
        Simulates the imaging process using the defined optical system and 
        returns the resulting image.

    Notes
    -----
    All volume scatterers imaged by a Microscope instance are assumed to
    share the same contrast mechanism (e.g. refractive index or fluorescence).
    Mixing contrast types is not supported.

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
        self:  Microscope,
        sample: Feature,
        objective: Feature,
        **kwargs: Any,
    ):
        """Initialize the `Microscope` instance.

        Parameters
        ----------
        sample: Feature
            A feature-set resolving a list of images describing the sample to be
            imaged.
        objective: Feature
            A feature-set defining the optical device that images the sample.
        **kwargs: Any
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

    def _validate_input(self, scattered):
        if hasattr(self._objective, "validate_input"):
            self._objective.validate_input(scattered)

    def _extract_contrast_volume(self, scattered):
        if hasattr(self._objective, "extract_contrast_volume"):
            return self._objective.extract_contrast_volume(
                scattered,
                **self._objective.properties(),
            )
        return scattered.array

    def _downscale_image(self, image, upscale):
        if hasattr(self._objective, "downscale_image"):
            return self._objective.downscale_image(image, upscale)

        if not np.any(np.array(upscale) != 1):
            return image

        ux, uy = upscale[:2]
        if ux != uy:
            raise ValueError(
                f"Energy-conserving detector integration requires ux == uy, "
                f"got ux={ux}, uy={uy}."
            )
        if isinstance(ux, float) and ux.is_integer():
            ux = int(ux)
        return AveragePooling(ux)(image)

    def get(
        self: Microscope,
        image: np.ndarray | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Generate an image of the sample using the defined optical system.

        This method processes the sample through the optical system to
        produce a simulated image.

        Parameters
        ----------
        image: np.ndarray | torch.Tensor | None
            The input image to be processed. If None, a new image is created.
        **kwargs: Any
            Additional parameters for the imaging process.

        Returns
        -------
        image: np.ndarray | torch.Tensor
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

            # Semantic validation (per scatterer)
            for scattered in list_of_scatterers:
                self._validate_input(scattered)

            # All scatterers that are defined as volumes.
            volume_samples = [
                scatterer
                for scatterer in list_of_scatterers
                if isinstance(scatterer, ScatteredVolume)
            ]

            # All scatterers that are defined as fields.
            field_samples = [
                scatterer
                for scatterer in list_of_scatterers
                if isinstance(scatterer, ScatteredField)
            ]
                
            # Merge all volumes into a single volume.
            sample_volume, limits = _create_volume(
                volume_samples,
                **additional_sample_kwargs,
            )

            print('prop', volume_samples[0].properties)

            # Interpret the merged volume semantically
            sample_volume = self._extract_contrast_volume(
                ScatteredVolume(
                    array=sample_volume,
                    properties=volume_samples[0].properties,
                ),
            )

            # Let the objective know about the limits of the volume and all the fields.
            propagate_data_to_dependencies(
                self._objective,
                limits=limits,
                fields=field_samples, # should We add upscale?
            )

            imaged_sample = self._objective.resolve(sample_volume)

        imaged_sample = self._downscale_image(imaged_sample, upscale)
        # # Handling upscale from dt.Upscale() here to eliminate Image
        # # wrapping issues.
        # if np.any(np.array(upscale) != 1):
        #     ux, uy = upscale[:2]
        #     if contrast_type == "intensity":
        #         print("Using sum pooling for intensity downscaling.")   
        #         imaged_sample = SumPoolingCM((ux, uy, 1))(imaged_sample)
        #     else:
        #         imaged_sample = AveragePoolingCM((ux, uy, 1))(imaged_sample)

        return imaged_sample


#TODO ***??*** revise Optics - torch, typing, docstring, unit test
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
    **kwargs: Any
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
    `_process_properties(propertydict: dict[str, Any]) -> dict[str, Any]`
        Processes and validates the input properties.
    `_pupil(shape:  array_like[int, int], NA: float, wavelength: float, refractive_index_medium: float, include_aberration: bool, defocus: float, **kwargs: Any) -> array_like[complex]`
        Calculates the pupil function at different focal points.
    `_pad_volume(volume: array_like[complex], limits: array_like[int, int], padding: array_like[int], output_region: array_like[int], **kwargs: Any) -> tuple`
        Pads the volume with zeros to avoid edge effects.
    `__call__(sample: Feature, **kwargs: Any) -> Microscope`
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
        self: Optics,
        NA: PropertyLike[float] = 0.7,
        wavelength: PropertyLike[float] = 0.66e-6,
        magnification: PropertyLike[float] = 10,
        resolution: PropertyLike[float | ArrayLike[float]] = 1e-6,
        refractive_index_medium: PropertyLike[float] = 1.33,
        padding: PropertyLike[ArrayLike[int]] = (10, 10, 10, 10),
        output_region: PropertyLike[ArrayLike[int]] = (0, 0, 128, 128),
        pupil: Feature = None,
        illumination: Feature = None,
        upscale: int = 1,
        **kwargs: Any,
    ):
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
        **kwargs: Any
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

        def validate_scattered(self, scattered):
            pass

        def extract_contrast_volume(self, scattered):
            pass

        def downscale_image(self, image, upscale):
            pass

        def get_voxel_size(
            resolution: float | ArrayLike[float], 
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
            resolution: float | ArrayLike[float],
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
        self: Optics,
        propertydict: dict[str, Any],
    ) -> dict[str, Any]:
        """Processes and validates the input properties.

        Ensures that the provided optical parameters are reasonable.

        Parameters
        ----------
        propertydict: dict[str, Any]
            The input properties.

        Returns
        -------
        dict[str, Any]
            The processed properties.
        
        """
        
        propertydict = super()._process_properties(propertydict)

        NA = propertydict["NA"]
        wavelength = propertydict["wavelength"]
        voxel_size = get_active_voxel_size()
        radius = NA / wavelength * np.array(voxel_size)
        print('Pupil radius (in pixels):', radius)

        if np.any(radius[:2] > 0.5):
            required_upscale = np.max(np.ceil(radius[:2] * 2))
            warnings.warn(
                f"""Likely bad optical parameters. NA / wavelength * 
                resolution / magnification = {radius} should be at most 0.5. 
                To fix, set magnification to {required_upscale}, and downsample
                the resulting image with 
                dt.AveragePooling(({required_upscale}, {required_upscale}, 1))
                """,
                UserWarning,
            )

        return propertydict

    def _pupil(
        self: Optics,
        shape: ArrayLike[int],
        NA: float,
        wavelength: float,
        refractive_index_medium: float,
        include_aberration: bool = True,   
        defocus: float | ArrayLike[float] = 0,
        **kwargs: Any,
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
        RHO = (W ** 2 + H ** 2).astype(complex)
        pupil_function = (RHO < 1) + 0.0j
        # Defocus
        z_shift = (
            2
            * np.pi
            * refractive_index_medium
            / wavelength
            * voxel_size[2]
            * np.sqrt(1 - (NA / refractive_index_medium) ** 2 * RHO)
        )

        z_shift[z_shift.imag != 0] = 0

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
        self: Optics,
        volume: ArrayLike[complex],
        limits: ArrayLike[int] = None,
        padding: ArrayLike[int] = None,
        output_region: ArrayLike[int] = None,
        **kwargs: Any,
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
        self: Optics,
        sample: Feature,
        **kwargs: Any,
    ) -> Microscope:
        """Creates a Microscope instance with the given sample and optics.

        Parameters
        ----------
        sample: Feature
            The sample to be imaged.
        **kwargs: Any
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
        from deeptrack.scatterers import MieScatterer # Temporary place for this import.

        if isinstance(self, (Darkfield, ISCAT, Holography)) and not isinstance(sample, MieScatterer):
            warnings.warn(
                f"{type(self).__name__} optics must be used with Mie scatterers "
                f"to produce a {type(self).__name__} image. "
                f"Got sample of type {type(sample).__name__}.",
                UserWarning,
            )

        return Microscope(sample, self, **kwargs)

    # def _no_wrap_format_input(self, *args, **kwargs) -> list:
    #     return self._image_wrapped_format_input(*args, **kwargs)
    
    # def _no_wrap_process_and_get(self, *args, **feature_input) -> list:
    #     return self._image_wrapped_process_and_get(*args, **feature_input)
    
    # def _no_wrap_process_output(self, *args, **feature_input):
    #     return self._image_wrapped_process_output(*args, **feature_input)


#TODO ***??*** revise Fluorescence - torch, typing, docstring, unit test
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
    **kwargs: Any

    Attributes
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
    `get(illuminated_volume: array_like[complex], limits: array_like[int, int], **kwargs: Any) -> np.ndarray`
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


    def validate_input(self, scattered):
        """Semantic validation for fluorescence microscopy."""
        
        # Fluorescence cannot operate on coherent fields
        if isinstance(scattered, ScatteredField):
            raise TypeError(
                "Fluorescence microscope cannot operate on ScatteredField."
            )


    def extract_contrast_volume(self, scattered: ScatteredVolume) -> np.ndarray:
        voxel_size = np.asarray(get_active_voxel_size(), float)
        voxel_volume = np.prod(voxel_size)

        intensity = scattered.get_property("intensity", None)
        value = scattered.get_property("value", None)
        ri = scattered.get_property("refractive_index", None)

        # Refractive index is always ignored in fluorescence
        if ri is not None:
            warnings.warn(
                "Scatterer defines 'refractive_index', which is ignored in "
                "fluorescence microscopy.",
                UserWarning,
            )

        # Preferred, physically meaningful case
        if intensity is not None:
            return intensity * voxel_volume * scattered.array

        # Fallback: legacy / dimensionless brightness
        warnings.warn(
            "Fluorescence scatterer has no 'intensity'. Interpreting 'value' as a "
            "non-physical brightness factor. Quantitative interpretation is invalid. "
            "Define 'intensity' to model physical fluorescence emission.",
            UserWarning,
        )

        return value * scattered.array

    def downscale_image(self, image: np.ndarray, upscale):
        """Detector downscaling (energy conserving)"""
        if not np.any(np.array(upscale) != 1):
            return image

        ux, uy = upscale[:2]
        if ux != uy:
            raise ValueError(
                f"Energy-conserving detector integration requires ux == uy, "
                f"got ux={ux}, uy={uy}."
            )
        if isinstance(ux, float) and ux.is_integer():
            ux = int(ux)

        # Energy-conserving detector integration
        return SumPooling(ux)(image)


    def get(
        self:  Fluorescence,
        illuminated_volume: ArrayLike[complex],
        limits: ArrayLike[int],
        **kwargs: Any,
    ) -> ArrayLike[complex]:
        """Simulates the imaging process using a fluorescence microscope.

        This method convolves the 3D illuminated volume with a pupil function 
        to generate a 2D image projection.

        Parameters
        ----------
        illuminated_volume: array_like[complex]
            The illuminated 3D volume to be imaged.
        limits: array_like[int, int]
            Boundaries of the illuminated volume in each dimension.
        **kwargs: Any
            Additional properties for the imaging process, such as:
            - 'padding': Padding to apply to the sample.
            - 'output_region': Specific region to extract from the image.

        Returns
        -------
        image: np.ndarray
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
        >>> volume = np.ones((128, 128, 10), dtype=complex)
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

        output_image = np.zeros((*padded_volume.shape[0:2], 1))

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
        volume = pad_image_to_fft(padded_volume, axes=(0, 1))
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
            output_image[:, :, 0] += field[
                : padded_volume.shape[0], : padded_volume.shape[1]
            ]

        output_image = output_image[pad[0] : -pad[2], pad[1] : -pad[3]]
        # output_image.properties = illuminated_volume.properties + pupils.properties

        return output_image


#TODO ***??*** revise Brightfield - torch, typing, docstring, unit test
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
        **kwargs: Any) -> np.ndarray`
        Simulates imaging with brightfield microscopy.


    Examples
    --------
    Create a `Brightfield` instance:

    >>> import deeptrack as dt

    >>> optics = dt.Brightfield(NA=1.4, wavelength=0.52e-6, magnification=60)
    >>> print(optics.NA())
    1.4
    
    """


    __conversion_table__ = ConversionTable(
    working_distance=(u.meter, u.meter),
)

    def validate_input(self, scattered):
        """Semantic validation for brightfield microscopy."""

        if isinstance(scattered, ScatteredVolume):
            warnings.warn(
                "Brightfield imaging from ScatteredVolume assumes a "
                "weak-phase / projection approximation. "
                "Use ScatteredField for physically accurate brightfield simulations.",
                UserWarning,
            )

    def extract_contrast_volume(
        self,
        scattered: ScatteredVolume,
        refractive_index_medium: float,
        **kwargs: Any,
    ) -> np.ndarray:
        print('ri_medium', refractive_index_medium)

        ri = scattered.get_property("refractive_index", None)
        value = scattered.get_property("value", None)
        intensity = scattered.get_property("intensity", None)

        if intensity is not None:
            warnings.warn(
                "Scatterer defines 'intensity', which is ignored in "
                "brightfield microscopy.",
                UserWarning,
            )

        if ri is not None:
            return (ri - refractive_index_medium) * scattered.array

        warnings.warn(
            "No 'refractive_index' specified; using 'value' as a non-physical "
            "brightfield contrast. Results are not physically calibrated. "
            "Define 'refractive_index' for physically meaningful contrast.",
            UserWarning,
        )

        return value * scattered.array

    def get(
        self: Brightfield,
        illuminated_volume: ArrayLike[complex],
        limits: ArrayLike[int],
        fields: ArrayLike[complex],
        **kwargs: Any,
    ) -> np.ndarray:
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
        **kwargs: Any
            Additional parameters for the imaging process, including:
            - 'padding': Padding to apply to the sample volume.
            - 'output_region': Specific region to extract from the image.
            - 'wavelength': Wavelength of the light.
            - 'refractive_index_medium': Refractive index of the medium.

        Returns
        -------
        image: np.ndarray
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
        >>> volume = np.ones((128, 128, 10), dtype=complex)
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

        output_image = np.zeros((*padded_volume.shape[0:2], 1))

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

        light_in = np.ones(volume.shape[:2], dtype=complex)
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
            # field = np.sum(fields, axis=0)
            field_arrays = []

            for fs in fields:
                # fs is a ScatteredField
                arr = fs.array

                # Enforce (H, W, 1) shape
                if arr.ndim == 2:
                    arr = arr[..., None]

                if arr.ndim != 3 or arr.shape[-1] != 1:
                    raise ValueError(
                        f"Expected field of shape (H, W, 1), got {arr.shape}"
                    )

                field_arrays.append(arr)

            field = np.sum(field_arrays, axis=0)
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
        output_image = output_image[pad[0] : -pad[2], pad[1] : -pad[3]]

        if not kwargs.get("return_field", False):
            output_image = np.square(np.abs(output_image))
        # else:
        # Fudge factor. Not sure why this is needed.
        # output_image = output_image - 1
        # output_image = output_image * np.exp(1j * -np.pi / 4)
        # output_image = output_image + 1

        # output_image.properties = illuminated_volume.properties

        return output_image


#TODO ***??*** revise Holography - torch, typing, docstring, unit test
class Holography(Brightfield):
    """An alias for the Brightfield class, representing holographic 
    imaging setups.

    Holography shares the same implementation as Brightfield, as both use 
    coherent illumination and similar propagation techniques.

    """
    pass


#TODO ***??*** revise ISCAT - torch, typing, docstring, unit test
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
        self:  ISCAT,
        illumination_angle: float = np.pi,
        amp_factor: float = 1, 
        **kwargs: Any,
    ) -> None:
        """Initializes the ISCAT class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        amp_factor: float
            Amplitude factor of the illuminating field relative to the reference 
            field.
        **kwargs: Any
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
  

#TODO ***??*** revise Darkfield - torch, typing, docstring, unit test      
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
        self: Darkfield,
        illumination_angle: float = np.pi/2,
        **kwargs: Any,
    ) -> None:
        """Initializes the Darkfield class.

        Parameters
        ----------
        illumination_angle: float
            The angle of illumination, in radians.
        **kwargs: Any
            Additional parameters for the Brightfield class.

        """

        super().__init__(
            illumination_angle=illumination_angle,
            **kwargs)

    def validate_input(self, scattered):
        if isinstance(scattered, ScatteredVolume):
            warnings.warn(
                "Darkfield imaging from ScatteredVolume is a very rough "
                "approximation. Use ScatteredField for physically meaningful "
                "darkfield simulations.",
                UserWarning,
            )

    def extract_contrast_volume(
        self,
        scattered: ScatteredVolume,
        refractive_index_medium: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Approximate darkfield contrast from a volume (toy model).

        This is a non-physical approximation intended for qualitative simulations.
        """

        ri = scattered.get_property("refractive_index", None)
        value = scattered.get_property("value", None)
        intensity = scattered.get_property("intensity", None)

        # Intensity has no meaning here
        if intensity is not None:
            warnings.warn(
                "Scatterer defines 'intensity', which is ignored in "
                "darkfield microscopy.",
                UserWarning,
            )

        if ri is not None:
            delta_n = ri - refractive_index_medium
            warnings.warn(
                "Approximating darkfield contrast from refractive index. "
                "Result is non-physical and qualitative only.",
                UserWarning,
            )
            return (delta_n ** 2) * scattered.array

        warnings.warn(
            "No 'refractive_index' specified; using 'value' as a non-physical "
            "darkfield scattering strength. Results are qualitative only.",
            UserWarning,
        )

        return (value ** 2) * scattered.array

    def downscale_image(self, image: np.ndarray, upscale):
        """Detector downscaling (energy conserving)"""
        if not np.any(np.array(upscale) != 1):
            return image

        ux, uy = upscale[:2]
        if ux != uy:
            raise ValueError(
                f"Energy-conserving detector integration requires ux == uy, "
                f"got ux={ux}, uy={uy}."
            )
        if isinstance(ux, float) and ux.is_integer():
            ux = int(ux)

        # Energy-conserving detector integration
        return SumPooling(ux)(image)

    #Retrieve get as super
    def get(
        self: Darkfield,
        illuminated_volume: ArrayLike[complex],
        limits: ArrayLike[int],
        fields: ArrayLike[complex],
        **kwargs: Any,
    ) -> np.ndarray:
        """Retrieve the darkfield image of the illuminated volume.

        Parameters
        ----------
        illuminated_volume: array_like
            The volume of the sample being illuminated.
        limits: array_like
            The spatial limits of the volume.
        fields: array_like
            The fields interacting with the sample.
        **kwargs: Any
            Additional parameters passed to the super class's get method.

        Returns
        -------
        numpy.ndarray
            The darkfield image obtained by calculating the squared absolute
            difference from 1.
        
        """

        field = super().get(illuminated_volume, limits, fields, return_field=True, **kwargs)
        return np.square(np.abs(field-1))


#TODO ***??*** revise IlluminationGradient - torch, typing, docstring, unit test
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
        self: IlluminationGradient,
        gradient: PropertyLike[ArrayLike[float]] = (0, 0),
        constant: PropertyLike[float] = 0,
        vmin: PropertyLike[float] = 0,
        vmax: PropertyLike[float] = np.inf,
        **kwargs: Any,
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
        **kwargs: Any
            Additional parameters for customization.

        """

        super().__init__(
            gradient=gradient, constant=constant, vmin=vmin, vmax=vmax, **kwargs
        )

    def get(
        self: IlluminationGradient,
        image: ArrayLike[complex],
        gradient: ArrayLike[float],
        constant: float,
        vmin: float,
        vmax: float,
        **kwargs: Any,
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
        **kwargs: Any
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


class NonOverlapping(Feature):
    """Ensure volumes are placed non-overlapping in a 3D space.

    This feature ensures that a list of 3D volumes are positioned such that 
    their non-zero voxels do not overlap. If volumes overlap, their positions 
    are resampled until they are non-overlapping. If the maximum number of 
    attempts is exceeded, the feature regenerates the list of volumes and 
    raises a warning if non-overlapping placement cannot be achieved.
    
    Note: `min_distance` refers to the distance between the edges of volumes, 
    not their centers. Due to the way volumes are calculated, slight rounding 
    errors may affect the final distance.
    
    This feature is incompatible with non-volumetric scatterers such as 
    `MieScatterers`.
    
    Parameters
    ----------
    feature: Feature
        The feature that generates the list of volumes to place 
        non-overlapping.
    min_distance: float, optional
        The minimum distance between volumes in pixels. It can be negative to
        allow for partial overlap. Defaults to 1. 
    max_attempts: int, optional
        The maximum number of attempts to place volumes without overlap.
        Defaults to 5. 
    max_iters: int, optional
        The maximum number of resamplings. If this number is exceeded, a new
        list of volumes is generated. Defaults to 100.

    Attributes
    ----------
    __distributed__: bool
        Always `False` for `NonOverlapping`, indicating that this feature’s
        `.get()` method processes the entire input at once even if it is a
        list, rather than distributing calls for each item of the list.N

    Methods
    -------
    `get(*_, min_distance, max_attempts, **kwargs) -> array`
        Generate a list of non-overlapping 3D volumes.
    `_check_non_overlapping(list_of_volumes) -> bool`
        Check if all volumes in the list are non-overlapping.
    `_check_bounding_cubes_non_overlapping(...) -> bool`
        Check if two bounding cubes are non-overlapping.
    `_get_overlapping_cube(...) -> list[int]`
        Get the overlapping cube between two bounding cubes.
    `_get_overlapping_volume(...) -> array`
        Get the overlapping volume between a volume and a bounding cube.
    `_check_volumes_non_overlapping(...) -> bool`
        Check if two volumes are non-overlapping.
    `_resample_volume_position(volume) -> Image`
        Resample the position of a volume to avoid overlap.
    
    Notes
    -----
    - This feature performs bounding cube checks first to quickly reject
      obvious overlaps before voxel-level checks.
    - If the bounding cubes overlap, precise voxel-based checks are performed.

    Examples
    ---------
    >>> import deeptrack as dt

    Define an ellipse scatterer with randomly positioned objects:

    >>> import numpy as np
    >>>
    >>> scatterer = dt.Ellipse(
    >>>    radius= 13 * dt.units.pixels,
    >>>    position=lambda: np.random.uniform(5, 115, size=2)* dt.units.pixels,
    >>> )

    Create multiple scatterers:

    >>> scatterers = (scatterer ^ 8)  

    Define the optics and create the image with possible overlap:

    >>> optics = dt.Fluorescence()
    >>> im_with_overlap = optics(scatterers)
    >>> im_with_overlap.store_properties()
    >>> im_with_overlap_resolved = image_with_overlap()

    Gather position from image:

    >>> pos_with_overlap = np.array(
    >>>     im_with_overlap_resolved.get_property(
    >>>         "position", 
    >>>         get_one=False
    >>>     )
    >>> )

    Enforce non-overlapping and create the image without overlap:
    
    >>> non_overlapping_scatterers = dt.NonOverlapping(
    ...     scatterers,
    ...     min_distance=4,
    ... )
    >>> im_without_overlap =  optics(non_overlapping_scatterers)
    >>> im_without_overlap.store_properties()
    >>> im_without_overlap_resolved = im_without_overlap()

    Gather position from image:

    >>> pos_without_overlap = np.array(
    >>>     im_without_overlap_resolved.get_property(
    >>>         "position",
    >>>        get_one=False
    >>>     )
    >>> )

    Create a figure with two subplots to visualize the difference:

    >>> import matplotlib.pyplot as plt
    >>>
    >>> fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    >>>
    >>> axes[0].imshow(im_with_overlap_resolved, cmap="gray")
    >>> axes[0].scatter(pos_with_overlap[:,1],pos_with_overlap[:,0])
    >>> axes[0].set_title("Overlapping Objects")
    >>> axes[0].axis("off")
    >>>
    >>> axes[1].imshow(im_without_overlap_resolved, cmap="gray")
    >>> axes[1].scatter(pos_without_overlap[:,1],pos_without_overlap[:,0])
    >>> axes[1].set_title("Non-Overlapping Objects")
    >>> axes[1].axis("off")
    >>> plt.tight_layout()
    >>>
    >>> plt.show()

    Define function to calculate minimum distance:

    >>> def calculate_min_distance(positions):
    >>> distances = [
    >>>     np.linalg.norm(positions[i] - positions[j])
    >>>     for i in range(len(positions))
    >>>         for j in range(i + 1, len(positions))
    >>> ]
    >>> return min(distances)

    Print minimum distances with and without overlap:

    >>> print(calculate_min_distance(pos_with_overlap))
    10.768742383382174

    >>> print(calculate_min_distance(pos_without_overlap))
    30.82531120942446

    """

    __distributed__: bool = False

    def __init__(
        self: NonOverlapping,
        feature: Feature,
        min_distance: float = 1,
        max_attempts: int = 5,
        max_iters: int = 100,
        **kwargs: Any,
    ):
        """Initializes the NonOverlapping feature.

        Ensures that volumes are placed **non-overlapping** by iteratively 
        resampling their positions. If the maximum number of attempts is 
        exceeded, the feature regenerates the list of volumes.

        Parameters
        ----------
        feature: Feature
            The feature that generates the list of volumes.
        min_distance: float, optional
            The minimum separation distance **between volume edges**, in 
            pixels. It defaults to `1`. Negative values allow for partial
            overlap.
        max_attempts: int, optional
            The maximum number of attempts to place the volumes without 
            overlap. It defaults to `5`.
        max_iters: int, optional
            The maximum number of resampling iterations per attempt. If 
            exceeded, a new list of volumes is generated. It defaults to `100`.

        """

        super().__init__(
            min_distance=min_distance, 
            max_attempts=max_attempts, 
            max_iters=max_iters,
            **kwargs,
        )
        self.feature = self.add_feature(feature, **kwargs)

    def get(
        self: NonOverlapping,
        *_: Any,
        min_distance: float,
        max_attempts: int,
        max_iters: int,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Generates a list of non-overlapping 3D volumes within a defined 
        field of view (FOV).

        This method **iteratively** attempts to place volumes while ensuring 
        they maintain at least `min_distance` separation. If non-overlapping 
        placement is not achieved within `max_attempts`, a warning is issued, 
        and the best available configuration is returned.

        Parameters
        ----------
        _: Any
            Placeholder parameter, typically for an input image.
        min_distance: float
            The minimum required separation distance between volumes, in 
            pixels.
        max_attempts: int
            The maximum number of attempts to generate a valid non-overlapping 
            configuration.
        max_iters: int
            The maximum number of resampling iterations per attempt.
        **kwargs: Any
            Additional parameters that may be used by subclasses.

        Returns
        -------
        list[np.ndarray]
            A list of 3D volumes represented as NumPy arrays. If 
            non-overlapping placement is unsuccessful, the best available 
            configuration is returned.

        Warns
        -----
        UserWarning
            If non-overlapping placement is **not** achieved within 
            `max_attempts`, suggesting parameter adjustments such as increasing
            the FOV or reducing `min_distance`.

        Notes
        -----
        - The placement process prioritizes bounding cube checks for
          efficiency.
        - If bounding cubes overlap, voxel-based overlap checks are performed.
        
        """

        for _ in range(max_attempts):
            list_of_volumes = self.feature()

            if not isinstance(list_of_volumes, list):
                list_of_volumes = [list_of_volumes]

            for _ in range(max_iters):
                
                list_of_volumes = [
                    self._resample_volume_position(volume) 
                    for volume in list_of_volumes
                ]

                if self._check_non_overlapping(list_of_volumes):
                    return list_of_volumes

            # Generate a new list of volumes if max_attempts is exceeded.
            self.feature.update()

        warnings.warn(
            "Non-overlapping placement could not be achieved. Consider "
            "adjusting parameters: reduce object radius, increase FOV, "
            "or decrease min_distance.",
            UserWarning,
        )
        return list_of_volumes

    def _check_non_overlapping(
        self: NonOverlapping, 
        list_of_volumes: list[np.ndarray],
    ) -> bool:
        """Determines whether all volumes in the provided list are 
        non-overlapping.

        This method verifies that the non-zero voxels of each 3D volume in 
        `list_of_volumes` are at least `min_distance` apart. It first checks 
        bounding boxes for early rejection and then examines actual voxel 
        overlap when necessary. Volumes are assumed to have a `position` 
        attribute indicating their placement in 3D space.

        Parameters
        ----------
        list_of_volumes: list[np.ndarray]
            A list of 3D arrays representing the volumes to be checked for 
            overlap. Each volume is expected to have a position attribute.

        Returns
        -------
        bool
            `True` if all volumes are non-overlapping, otherwise `False`.

        Notes
        -----
        - If `min_distance` is negative, volumes are shrunk using isotropic 
          erosion before checking overlap.
        - If `min_distance` is positive, volumes are padded and expanded using 
          isotropic dilation.
        - Overlapping checks are first performed on bounding cubes for 
            efficiency.
        - If bounding cubes overlap, voxel-level checks are performed.

        """
        from deeptrack.scatterers import ScatteredVolume

        from deeptrack.augmentations import CropTight, Pad # these are not compatibles with torch backend
        from deeptrack.optics import _get_position
        from deeptrack.math import isotropic_erosion, isotropic_dilation

        min_distance = self.min_distance()
        crop = CropTight()

        new_volumes = []
        
        for volume in list_of_volumes:
            arr = volume.array
            mask = arr != 0

            if min_distance < 0:
                new_arr = isotropic_erosion(mask, -min_distance / 2, backend=self.get_backend())
            else:
                pad = Pad(px=[int(np.ceil(min_distance / 2))] * 6, keep_size=True)
                new_arr = isotropic_dilation(pad(mask) != 0 , min_distance / 2, backend=self.get_backend())
                new_arr = crop(new_arr)

            if self.get_backend() == "torch":
                new_arr = new_arr.to(dtype=arr.dtype)
            else:
                new_arr = new_arr.astype(arr.dtype)

            new_volume = ScatteredVolume(
                array=new_arr,
                properties=volume.properties.copy(),
            )

            new_volumes.append(new_volume)

        list_of_volumes = new_volumes       
        min_distance = 1

        # The position of the top left corner of each volume (index (0, 0, 0)).
        volume_positions_1 = [
            _get_position(volume, mode="corner", return_z=True).astype(int)
            for volume in list_of_volumes
        ]

        # The position of the bottom right corner of each volume 
        # (index (-1, -1, -1)).
        volume_positions_2 = [
            p0 + np.array(v.shape) 
            for v, p0 in zip(list_of_volumes, volume_positions_1)
        ]

        # (x1, y1, z1, x2, y2, z2) for each volume.
        volume_bounding_cube = [
            [*p0, *p1] 
            for p0, p1 in zip(volume_positions_1, volume_positions_2)
        ]

        for i, j in itertools.combinations(range(len(list_of_volumes)), 2):

            # If the bounding cubes do not overlap, the volumes do not overlap.
            if self._check_bounding_cubes_non_overlapping(
                volume_bounding_cube[i], volume_bounding_cube[j], min_distance
            ):
                continue

            # If the bounding cubes overlap, get the overlapping region of each 
            # volume.
            overlapping_cube = self._get_overlapping_cube(
                volume_bounding_cube[i], volume_bounding_cube[j]
            )
            overlapping_volume_1 = self._get_overlapping_volume(
                list_of_volumes[i].array, volume_bounding_cube[i], overlapping_cube
            )
            overlapping_volume_2 = self._get_overlapping_volume(
                list_of_volumes[j].array, volume_bounding_cube[j], overlapping_cube
            )

            # If either the overlapping regions are empty, the volumes do not 
            # overlap (done for speed).
            if (np.all(overlapping_volume_1 == 0)
                or np.all(overlapping_volume_2 == 0)):
                continue

            # If products of overlapping regions are non-zero, return False.
            # if np.any(overlapping_volume_1 * overlapping_volume_2):
            #     return False

            # Finally, check that the non-zero voxels of the volumes are at 
            # least min_distance apart.
            if not self._check_volumes_non_overlapping(
                overlapping_volume_1, overlapping_volume_2, min_distance
            ):
                return False

        return True

    def _check_bounding_cubes_non_overlapping(
        self: NonOverlapping,
        bounding_cube_1: list[int],
        bounding_cube_2: list[int], 
        min_distance: float,
    ) -> bool:
        """Determines whether two 3D bounding cubes are non-overlapping.

        This method checks whether the bounding cubes of two volumes are 
        **separated by at least** `min_distance` along **any** spatial axis.

        Parameters
        ----------
        bounding_cube_1: list[int]
            A list of six integers `[x1, y1, z1, x2, y2, z2]` representing 
            the first bounding cube.
        bounding_cube_2: list[int]
            A list of six integers `[x1, y1, z1, x2, y2, z2]` representing 
            the second bounding cube.
        min_distance: float
            The required **minimum separation distance** between the two 
            bounding cubes.

        Returns
        -------
        bool
            `True` if the bounding cubes are non-overlapping (separated by at 
            least `min_distance` along **at least one axis**), otherwise 
            `False`.

        Notes
        -----
        - This function **only checks bounding cubes**, **not actual voxel 
          data**.
        - If the bounding cubes are non-overlapping, the corresponding 
          **volumes are also non-overlapping**.
        - This check is much **faster** than full voxel-based comparisons.
        
        """

        # bounding_cube_1 and bounding_cube_2 are (x1, y1, z1, x2, y2, z2).
        # Check that the bounding cubes are non-overlapping.
        return (
        (bounding_cube_1[0] >= bounding_cube_2[3] + min_distance) or
        (bounding_cube_2[0] >= bounding_cube_1[3] + min_distance) or
        (bounding_cube_1[1] >= bounding_cube_2[4] + min_distance) or
        (bounding_cube_2[1] >= bounding_cube_1[4] + min_distance) or
        (bounding_cube_1[2] >= bounding_cube_2[5] + min_distance) or
        (bounding_cube_2[2] >= bounding_cube_1[5] + min_distance)
        )

    def _get_overlapping_cube(
        self: NonOverlapping,
        bounding_cube_1: list[int],
        bounding_cube_2: list[int],
    ) -> list[int]:
        """Computes the overlapping region between two 3D bounding cubes.

        This method calculates the coordinates of the intersection of two 
        axis-aligned bounding cubes, each represented as a list of six 
        integers:

        - `[x1, y1, z1]`: Coordinates of the **top-left-front** corner.
        - `[x2, y2, z2]`: Coordinates of the **bottom-right-back** corner.

        The resulting overlapping region is determined by:
        - Taking the **maximum** of the starting coordinates (`x1, y1, z1`).
        - Taking the **minimum** of the ending coordinates (`x2, y2, z2`).

        If the cubes **do not** overlap, the resulting coordinates will not 
        form a valid cube (i.e., `x1 > x2`, `y1 > y2`, or `z1 > z2`).

        Parameters
        ----------
        bounding_cube_1: list[int]
            The first bounding cube, formatted as `[x1, y1, z1, x2, y2, z2]`.
        bounding_cube_2: list[int]
            The second bounding cube, formatted as `[x1, y1, z1, x2, y2, z2]`.

        Returns
        -------
        list[int]
            A list of six integers `[x1, y1, z1, x2, y2, z2]` representing the 
            overlapping bounding cube. If no overlap exists, the coordinates 
            will **not** define a valid cube.

        Notes
        -----
        - This function does **not** check for valid input or ensure the 
          resulting cube is well-formed.
        - If no overlap exists, downstream functions must handle the invalid 
          result.
        
        """

        return [
            max(bounding_cube_1[0], bounding_cube_2[0]),
            max(bounding_cube_1[1], bounding_cube_2[1]),
            max(bounding_cube_1[2], bounding_cube_2[2]),
            min(bounding_cube_1[3], bounding_cube_2[3]),
            min(bounding_cube_1[4], bounding_cube_2[4]),
            min(bounding_cube_1[5], bounding_cube_2[5]),
        ]

    def _get_overlapping_volume(
        self: NonOverlapping,
        volume: np.ndarray,  # 3D array.
        bounding_cube: tuple[float, float, float, float, float, float],
        overlapping_cube: tuple[float, float, float, float, float, float],
    ) -> np.ndarray:
        """Extracts the overlapping region of a 3D volume within the specified 
        overlapping cube.

        This method identifies and returns the subregion of `volume` that 
        lies within the `overlapping_cube`. The bounding information of the 
        volume is provided via `bounding_cube`.

        Parameters
        ----------
        volume: np.ndarray
            A 3D NumPy array representing the volume from which the 
            overlapping region is extracted.
        bounding_cube: tuple[float, float, float, float, float, float]
            The bounding cube of the volume, given as a tuple of six floats: 
            `(x1, y1, z1, x2, y2, z2)`. The first three values define the 
            **top-left-front** corner, while the last three values define the 
            **bottom-right-back** corner.
        overlapping_cube: tuple[float, float, float, float, float, float]
            The overlapping region between the volume and another volume, 
            represented in the same format as `bounding_cube`.

        Returns
        -------
        np.ndarray
            A 3D NumPy array representing the portion of `volume` that 
            lies within `overlapping_cube`. If the overlap does not exist, 
            an empty array may be returned.

        Notes
        -----
        - The method computes the relative indices of `overlapping_cube` 
          within `volume` by subtracting the bounding cube's starting 
          position.
        - The extracted region is determined by integer indices, meaning 
          coordinates are implicitly **floored to integers**.
        - If `overlapping_cube` extends beyond `volume` boundaries, the 
          returned subregion is **cropped** to fit within `volume`.
        
        """

        # The position of the top left corner of the overlapping cube in the volume
        overlapping_cube_position = np.array(overlapping_cube[:3]) - np.array(
            bounding_cube[:3]
        )

        # The position of the bottom right corner of the overlapping cube in the volume
        overlapping_cube_end_position = np.array(
            overlapping_cube[3:]
            ) - np.array(bounding_cube[:3])

        # cast to int
        overlapping_cube_position = overlapping_cube_position.astype(int)
        overlapping_cube_end_position = overlapping_cube_end_position.astype(int)

        return volume[
            overlapping_cube_position[0] : overlapping_cube_end_position[0],
            overlapping_cube_position[1] : overlapping_cube_end_position[1],
            overlapping_cube_position[2] : overlapping_cube_end_position[2],
        ]

    def _check_volumes_non_overlapping(
        self: NonOverlapping,
        volume_1: np.ndarray,
        volume_2: np.ndarray,
        min_distance: float,
    ) -> bool:
        """Determines whether the non-zero voxels in two 3D volumes are at 
        least `min_distance` apart.

        This method checks whether the active regions (non-zero voxels) in 
        `volume_1` and `volume_2` maintain a minimum separation of 
        `min_distance`. If the volumes differ in size, the positions of their 
        non-zero voxels are adjusted accordingly to ensure a fair comparison.

        Parameters
        ----------
        volume_1: np.ndarray
            A 3D NumPy array representing the first volume.
        volume_2: np.ndarray
            A 3D NumPy array representing the second volume.
        min_distance: float
            The minimum Euclidean distance required between any two non-zero 
            voxels in the two volumes.

        Returns
        -------
        bool
            `True` if all non-zero voxels in `volume_1` and `volume_2` are at 
            least `min_distance` apart, otherwise `False`.

        Notes
        -----
        - This function assumes both volumes are correctly aligned within a 
          shared coordinate space.
        - If the volumes are of different sizes, voxel positions are scaled 
          or adjusted for accurate distance measurement.
        - Uses **Euclidean distance** for separation checking.
        - If either volume is empty (i.e., no non-zero voxels), they are 
          considered non-overlapping.
        
        """

        # Get the positions of the non-zero voxels of each volume.
        if self.get_backend() == "torch":
            positions_1 = torch.nonzero(volume_1, as_tuple=False)
            positions_2 = torch.nonzero(volume_2, as_tuple=False)
        else:
            positions_1 = np.argwhere(volume_1)
            positions_2 = np.argwhere(volume_2)

        # if positions_1.size == 0 or positions_2.size == 0:
        #     return True  # If either volume is empty, they are "non-overlapping"

        # # If the volumes are not the same size, the positions of the non-zero 
        # # voxels of each volume need to be scaled.
        # if positions_1.size == 0 or positions_2.size == 0:
        #     return True  # If either volume is empty, they are "non-overlapping"

        # If the volumes are not the same size, the positions of the non-zero 
        # voxels of each volume need to be scaled.
        if volume_1.shape != volume_2.shape:
            positions_1 = (
                positions_1 * np.array(volume_2.shape) 
                / np.array(volume_1.shape)
            )
            positions_1 = positions_1.astype(int)

        # Check that the non-zero voxels of the volumes are at least 
        # min_distance apart.
        if self.get_backend() == "torch":
            dist = torch.cdist(
                positions_1.float(),
                positions_2.float(),
            )
            return bool((dist > min_distance).all())
        else:
            return np.all(cdist(positions_1, positions_2) > min_distance)

    def _resample_volume_position(
        self: NonOverlapping,
        volume: np.ndarray | Image,
    ) -> Image:
        """Resamples the position of a 3D volume using its internal position 
        sampler.

        This method updates the `position` property of the given `volume` by 
        drawing a new position from the `_position_sampler` stored in the 
        volume's `properties`. If the sampled position is a `Quantity`, it is 
        converted to pixel units.

        Parameters
        ----------
        volume: np.ndarray
            The 3D volume whose position is to be resampled. The volume must 
            have a `properties` attribute containing dictionaries with 
            `position` and `_position_sampler` keys.

        Returns
        -------
        Image
            The same input volume with its `position` property updated to the 
            newly sampled value.

        Notes
        -----
        - The `_position_sampler` function is expected to return a **tuple of 
        three floats** (e.g., `(x, y, z)`).
        - If the sampled position is a `Quantity`, it is converted to pixels.
        - **Only** dictionaries in `volume.properties` that contain both 
        `position` and `_position_sampler` keys are modified.
        
        """

        pdict = volume.properties
        if "position" in pdict and "_position_sampler" in pdict:
            new_position = pdict["_position_sampler"]()
            if isinstance(new_position, Quantity):
                new_position = new_position.to("pixel").magnitude
            pdict["position"] = new_position

        return volume


class SampleToMasks(Feature):
    """Create a mask from a list of images.

    This feature applies a transformation function to each input image and 
    merges the resulting masks into a single multi-layer image. Each input 
    image must have a `position` property that determines its placement within 
    the final mask. When used with scatterers, the `voxel_size` property must 
    be provided for correct object sizing.

    Parameters
    ----------
    transformation_function: Callable[[Image], Image]
        A function that transforms each input image into a mask with 
        `number_of_masks` layers.
    number_of_masks: PropertyLike[int], optional
        The number of mask layers to generate. Default is 1.
    output_region: PropertyLike[tuple[int, int, int, int]], optional
        The size and position of the output mask, typically aligned with 
        `optics.output_region`.
    merge_method: PropertyLike[str | Callable | list[str | Callable]], optional
        Method for merging individual masks into the final image. Can be:
        - "add" (default): Sum the masks.
        - "overwrite": Later masks overwrite earlier masks.
        - "or": Combine masks using a logical OR operation.
        - "mul": Multiply masks.
        - Function: Custom function taking two images and merging them.

    **kwargs: dict[str, Any]
        Additional keyword arguments passed to the parent `Feature` class.

    Methods
    -------
    `get(image, transformation_function, **kwargs) -> Image`
        Applies the transformation function to the input image.
    `_process_and_get(images, **kwargs) -> Image | np.ndarray`
        Processes a list of images and generates a multi-layer mask.

    Returns
    -------
    np.ndarray
        The final mask image with the specified number of layers.

    Raises
    ------
    ValueError
        If `merge_method` is invalid.

    Examples
    -------
    >>> import deeptrack as dt

    Define number of particles:

    >>> n_particles = 12

    Define optics and particles:

    >>> import numpy as np
    >>>    
    >>> optics = dt.Fluorescence(output_region=(0, 0, 64, 64))
    >>> particle = dt.PointParticle(
    >>>     position=lambda: np.random.uniform(5, 55, size=2),
    >>> )
    >>> particles = particle ^ n_particles

    Define pipelines:

    >>> sim_im_pip = optics(particles)
    >>> sim_mask_pip = particles >> dt.SampleToMasks(
    ...     lambda: lambda particles: particles > 0,
    ...     output_region=optics.output_region,
    ...     merge_method="or",
    ... )
    >>> pipeline = sim_im_pip & sim_mask_pip
    >>> pipeline.store_properties()

    Generate image and mask:

    >>> image, mask = pipeline.update()()

    Get particle positions:

    >>> positions = np.array(image.get_property("position", get_one=False))

    Visualize results:

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(image, cmap="gray")
    >>> plt.title("Original Image")
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(mask, cmap="gray")
    >>> plt.scatter(positions[:,1], positions[:,0], c="y", marker="x", s = 50)
    >>> plt.title("Mask")
    >>> plt.show()

    """

    def __init__(
        self: Feature,
        transformation_function: Callable[[np.ndarray], np.ndarray, torch.Tensor],
        number_of_masks: PropertyLike[int] = 1,
        output_region: PropertyLike[tuple[int, int, int, int]] = None,
        merge_method: PropertyLike[str | Callable | list[str | Callable]] = "add",
        **kwargs: Any,
    ):
        """Initialize the SampleToMasks feature.

        Parameters
        ----------
        transformation_function: Callable[[Image], Image]
            Function to transform input images into masks.
        number_of_masks: PropertyLike[int], optional
            Number of mask layers. Default is 1.
        output_region: PropertyLike[tuple[int, int, int, int]], optional
            Output region of the mask. Default is None.
        merge_method: PropertyLike[str | Callable | list[str | Callable]], optional
            Method to merge masks. Defaults to "add".
        **kwargs: dict[str, Any]
            Additional keyword arguments passed to the parent class.
        
        """

        super().__init__(
            transformation_function=transformation_function,
            number_of_masks=number_of_masks,
            output_region=output_region,
            merge_method=merge_method,
            **kwargs,
        )

    def get(
        self: Feature,
        image: np.ndarray,
        transformation_function: Callable[list[np.ndarray] | np.ndarray | torch.Tensor],
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply the transformation function to a single image.

        Parameters
        ----------
        image: np.ndarray
            The input image.
        transformation_function: Callable[[np.ndarray], np.ndarray]
            Function to transform the image.
        **kwargs: dict[str, Any]
            Additional parameters.

        Returns
        -------
        Image
            The transformed image.

        """

        return transformation_function(image.array)

    def _process_and_get(
        self: Feature,
        images: list[np.ndarray] | np.ndarray | list[torch.Tensor] | torch.Tensor,
        **kwargs: Any,
    ) -> np.ndarray:
        """Process a list of images and generate a multi-layer mask.

        Parameters
        ----------
        images: np.ndarray or list[np.ndarrray] or  Image or list[Image]
            List of input images or a single image.
        **kwargs: dict[str, Any]
            Additional parameters including `output_region`, `number_of_masks`, 
            and `merge_method`.

        Returns
        -------
        Image or np.ndarray
            The final mask image.
            
        """

        # Handle list of images.
        # if isinstance(images, list) and len(images) != 1:
        list_of_labels = super()._process_and_get(images, **kwargs)

        from deeptrack.scatterers import ScatteredVolume
        
        for idx, (label, image) in enumerate(zip(list_of_labels, images)):
            list_of_labels[idx] = \
                ScatteredVolume(array=label, properties=image.properties.copy())        

        # Create an empty output image.
        output_region = kwargs["output_region"]
        output = xp.zeros(
            (
                output_region[2] - output_region[0],
                output_region[3] - output_region[1],
                kwargs["number_of_masks"],
            ),
            dtype=list_of_labels[0].array.dtype,
        )

        from deeptrack.optics import _get_position

        # Merge masks into the output.
        for volume in list_of_labels:
            label = volume.array
            position = _get_position(volume)

            p0 = xp.round(position - xp.asarray(output_region[0:2]))
            p0 = p0.astype(xp.int64)


            if xp.any(p0 > xp.asarray(output.shape[:2])) or \
                xp.any(p0 + xp.asarray(label.shape[:2]) < 0):
                continue

            crop_x = (-xp.minimum(p0[0], 0)).item()
            crop_y = (-xp.minimum(p0[1], 0)).item()

            crop_x_end = int(
                label.shape[0]
                - np.max([p0[0] + label.shape[0] - output.shape[0], 0])
            )
            crop_y_end = int(
                label.shape[1]
                - np.max([p0[1] + label.shape[1] - output.shape[1], 0])
            )

            labelarg = label[crop_x:crop_x_end, crop_y:crop_y_end, :]

            p0[0] = np.max([p0[0], 0])
            p0[1] = np.max([p0[1], 0])

            p0 = p0.astype(int)

            output_slice = output[
                p0[0] : p0[0] + labelarg.shape[0],
                p0[1] : p0[1] + labelarg.shape[1],
            ]

            for label_index in range(kwargs["number_of_masks"]):

                if isinstance(kwargs["merge_method"], list):
                    merge = kwargs["merge_method"][label_index]
                else:
                    merge = kwargs["merge_method"]

                if merge == "add":
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] += labelarg[..., label_index]

                elif merge == "overwrite":
                    output_slice[
                        labelarg[..., label_index] != 0, label_index
                    ] = labelarg[labelarg[..., label_index] != 0, \
                        label_index]
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] = output_slice[..., label_index]

                elif merge == "or":
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] = xp.logical_or(
                        output_slice[..., label_index] != 0, 
                        labelarg[..., label_index] != 0
                        )

                elif merge == "mul":
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] *= labelarg[..., label_index]

                else:
                    # No match, assume function
                    output[
                        p0[0] : p0[0] + labelarg.shape[0],
                        p0[1] : p0[1] + labelarg.shape[1],
                        label_index,
                    ] = merge(
                        output_slice[..., label_index],
                        labelarg[..., label_index],
                    )

        return output


#TODO ***??*** revise _get_position - torch, typing, docstring, unit test
def _get_position(
    scatterer: ScatteredObject,
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

    if mode == "corner" and scatterer.array.size > 0:
        import scipy.ndimage

        shift = scipy.ndimage.center_of_mass(np.abs(scatterer.array))

        if np.isnan(shift).any():
            shift = np.array(scatterer.array.shape) / 2

    else:
        shift = np.zeros((num_outputs))

    position = np.array(scatterer.get_property("position", default=None))

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
                np.array([position[0], position[1], scatterer.get_property("z", default=0)])
                * scale
                - shift
                + 0.5 * (scale - 1)
            )
            return outp
        else:
            return position * scale[:2] - shift[0:2] + 0.5 * (scale[:2] - 1)

    return position


def _bilinear_interpolate_numpy(
    scatterer: np.ndarray, x_off: float, y_off: float
) -> np.ndarray:
    """Apply bilinear subpixel interpolation in the x–y plane (NumPy)."""
    kernel = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, (1 - x_off) * (1 - y_off), (1 - x_off) * y_off],
            [0.0, x_off * (1 - y_off), x_off * y_off],
        ]
    )
    out = np.zeros_like(scatterer)
    for z in range(scatterer.shape[2]):
        if np.iscomplexobj(scatterer):
            out[:, :, z] = (
                convolve(np.real(scatterer[:, :, z]), kernel, mode="constant")
                + 1j
                * convolve(np.imag(scatterer[:, :, z]), kernel, mode="constant")
            )
        else:
            out[:, :, z] = convolve(scatterer[:, :, z], kernel, mode="constant")
    return out


def _bilinear_interpolate_torch(
    scatterer: torch.Tensor, x_off: float, y_off: float
) -> torch.Tensor:
    """Apply bilinear subpixel interpolation in the x–y plane (Torch).

    Uses grid_sample for autograd-friendly interpolation.
    """
    H, W, D = scatterer.shape

    # Normalized shifts in [-1,1]
    x_shift = 2 * x_off / (W - 1)
    y_shift = 2 * y_off / (H - 1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=scatterer.device, dtype=scatterer.dtype),
        torch.linspace(-1, 1, W, device=scatterer.device, dtype=scatterer.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx + x_shift, yy + y_shift), dim=-1)  # (H,W,2)
    grid = grid.unsqueeze(0).repeat(D, 1, 1, 1)               # (D,H,W,2)

    inp = scatterer.permute(2, 0, 1).unsqueeze(1)             # (D,1,H,W)

    out = F.grid_sample(inp, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.squeeze(1).permute(1, 2, 0)                    # (H,W,D)


#TODO ***??*** revise _create_volume - torch, typing, docstring, unit test
def _create_volume(
    list_of_scatterers: list,
    pad: tuple = (0, 0, 0, 0),
    output_region: tuple = (None, None, None, None),
    refractive_index_medium: float = 1.33,
    **kwargs: Any,
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
    **kwargs: Any
        Additional arguments for customization.

    Returns
    -------
    tuple
        - volume: numpy.ndarray
            The generated volume containing the scatterers.
        - limits: numpy.ndarray
            Spatial limits of the volume.

    """
    # contrast_type = kwargs.get("contrast_type", None)
    # if contrast_type is None:
    #     raise RuntimeError(
    #         "_create_volume requires a contrast_type "
    #         "(e.g. 'intensity' or 'refractive_index')"
    #     )

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
    # fudge_factor = scale[0] * scale[1] / scale[2]

    for scatterer in list_of_scatterers:
        position = _get_position(scatterer, mode="corner", return_z=True)

        # if contrast_type == "intensity":
        #     value = scatterer.get_property("intensity", None)
        #     if value is None:
        #         raise ValueError("Scatterer has no intensity.")
        #     scatterer_value = value

        # elif contrast_type == "refractive_index":
        #     ri = scatterer.get_property("refractive_index", None)
        #     if ri is None:
        #         raise ValueError("Scatterer has no refractive_index.")
        #     scatterer_value = ri - refractive_index_medium

        # else:
        #     raise RuntimeError(f"Unknown contrast_type: {contrast_type}")

        # # Scale the array accordingly
        # scatterer.array = scatterer.array * scatterer_value

        if limits is None:
            limits = np.zeros((3, 2), dtype=np.int32)
            limits[:, 0] = np.floor(position).astype(np.int32)
            limits[:, 1] = np.floor(position).astype(np.int32) + 1

        if (
            position[0] + scatterer.array.shape[0] < OR[0]
            or position[0] > OR[2]
            or position[1] + scatterer.array.shape[1] < OR[1]
            or position[1] > OR[3]
        ):
            continue

        # Pad scatterer to avoid edge effects during interpolation
        padded_scatterer_arr = np.pad(  #Use Pad instead and make it torch-compatible?
                scatterer.array,
                [(2, 2), (2, 2), (2, 2)],
                "constant",
                constant_values=0,
            )
        padded_scatterer = ScatteredVolume(
            array=padded_scatterer_arr, properties=scatterer.properties.copy(),
            )
        position = _get_position(padded_scatterer, mode="corner", return_z=True)
        shape = np.array(padded_scatterer.array.shape)

        if position is None:
            RuntimeWarning(
                "Optical device received an image without a position property."
                " It will be ignored."
            )
            continue

        x_off = position[0] - np.floor(position[0])
        y_off = position[1] - np.floor(position[1])

        
        if isinstance(padded_scatterer.array, np.ndarray): # get_backend is a method of Features and not exposed 
            splined_scatterer = _bilinear_interpolate_numpy(padded_scatterer.array, x_off, y_off)
        elif isinstance(padded_scatterer.array, torch.Tensor):
            splined_scatterer = _bilinear_interpolate_torch(padded_scatterer.array, x_off, y_off)
        else:
            raise TypeError(
                f"Unsupported array type {type(padded_scatterer.array)}. "
                "Expected np.ndarray or torch.Tensor."
            )

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

        # NOTE: Maybe shouldn't be ONLY additive.
        # give options: sum default, but also mean, max, min, or
        volume[
            int(within_volume_position[0]) : 
            int(within_volume_position[0] + shape[0]),
            
            int(within_volume_position[1]) : 
            int(within_volume_position[1] + shape[1]),

            int(within_volume_position[2]) : 
            int(within_volume_position[2] + shape[2]),
        ] += splined_scatterer
    return volume, limits