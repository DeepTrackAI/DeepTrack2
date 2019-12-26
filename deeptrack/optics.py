''' Imaging features through optical systems

Any feature can be viewed through an optical system. The current image
will be regarded as a map of the complex field.

Contains
--------
Optics
    Base abstract class for optical devices.

OpticsBranch
    Special feature for imaging features.

OpticalDevice
    Implementation of Optics. Handles incoherent fields.

'''

import numpy as np
from deeptrack.features import Feature
from deeptrack.image import Image
from deeptrack.utils import as_list

from scipy.interpolate import RectBivariateSpline


class Microscope(Feature):

    __distributed__ = False

    def __init__(self, sample, objective, pupil, *args, **kwargs):
        super().__init__(*args, sample=sample, objective=objective, pupil=pupil, **kwargs)

    def get(self, image, sample=None, objective=None, pupil=None):

        list_of_scatterers = sample.resolve(**objective.properties.current_value_dict())
        if not isinstance(list_of_scatterers, list):
            list_of_scatterers = [list_of_scatterers]
        
        sample_volume, limits = create_volume(list_of_scatterers)
        


        sample_volume = Image(sample_volume)
        sample_volume.append({"limits": limits, "metric": "quantum_yield", "name": "Volume"})

        for scatterer in list_of_scatterers:
            sample_volume.properties += scatterer.properties

        imaged_sample = objective.resolve(sample_volume, pupil=pupil)


        # Merge with input
        if not image:
            return imaged_sample

        if not isinstance(image, list):
            image = [image]
        
        for i in range(len(image)):
            image[i] += imaged_sample
            image[i].properties += imaged_sample.properties

        return image


class Optics(Feature):

    def __init__(self,
                *args,
                 NA=0.7,
                 wavelength=0.66e-6,
                 magnification=1,
                 resolution=(1e-6, 1e-6, 1e-6),
                 refractive_index_medium=1.33,
                 upscale=2,
                 padding=(10, 10, 10, 10),
                 output_region=(None, None, None, None),
                 pupil=None,
                 **kwargs):

        auxiliary_dict = {"voxel_size": self.get_voxel_size}

        super().__init__(
            auxiliary_dict,
            *args,
            NA=NA,
            wavelength=wavelength,
            refractive_index_medium=refractive_index_medium,
            magnification=magnification,
            resolution=resolution,
            upscale=upscale,
            padding=padding,
            output_region=output_region,
            pupil=pupil,
            **kwargs
        )

    def get_voxel_size(self):

        resolution = self.properties["resolution"].current_value
        magnification = self.properties["magnification"].current_value

        if not isinstance(resolution, (list, tuple, np.ndarray)) or len(resolution) == 1:
            resolution = (resolution,) * 3
        elif len(resolution) == 2:
            resolution = (*resolution, np.min(resolution))

        return np.array(resolution) / magnification

    def pupil(self, shape,
              NA=None,
              wavelength=None,
              refractive_index_medium=None,
              voxel_size=None,
              defocus=None, 
              upscale=None,
              pupil=None,
              **kwargs):
        ''' Calculates pupil function

        Parameters
        ----------
        shape
            The shape of the pupil function
        kwargs
            The current values of the properties of the optical device
        '''
        shape = np.array(shape) 

        upscaled_shape = shape*upscale
        # Pupil radius
        R = NA / wavelength * np.array(voxel_size)[:2]

        x_radius = R[0] * upscaled_shape[0]
        y_radius = R[1] * upscaled_shape[1]

        x = (np.linspace(-(upscaled_shape[0] / 2), upscaled_shape[0] / 2 - 1, upscaled_shape[0])) / x_radius + 1e-8
        y = (np.linspace(-(upscaled_shape[1] / 2), upscaled_shape[1] / 2 - 1, upscaled_shape[1])) / y_radius + 1e-8

        W, H = np.meshgrid(y, x)
        RHO = W**2 + H**2
        RHO[RHO > 1] = 1
        pupil_function = (RHO < 1) * 1.0
        
        # Defocus
        z_shift = (2 * np.pi * refractive_index_medium / wavelength * voxel_size[2] * defocus
                   * np.sqrt(1 - (NA / refractive_index_medium * RHO)**2))


        # Downsample the upsampled pupil
        if upscale > 1:
            pupil_function = np.reshape(pupil_function, (shape[0], upscale, shape[1], upscale)).mean(axis=(3, 1))
            z_shift = np.reshape(z_shift, (shape[0], upscale, shape[1], upscale)).mean(axis=(3, 1))

        pupil_function = pupil_function * np.exp(1j * z_shift)

        pupil_function[np.isnan(pupil_function)] = 0
        pupil_function[np.isinf(pupil_function)] = 0

        if isinstance(pupil, Feature):
            pupil_function = pupil.resolve(pupil_function)
        elif isinstance(pupil, np.ndarray):
            pupil_function *= pupil
        
        return pupil_function


    def _pad_volume(self, volume, limits=None, padding=None, output_region=None, **kwargs):
        
        new_limits = np.array(limits)
        output_region = np.array(output_region) 

        new_limits[0, 0] -= padding[0]
        new_limits[0, 1] += padding[1]
        new_limits[1, 0] -= padding[2]
        new_limits[1, 1] += padding[3]

        # Replace None entries with current limit
        output_region[0] = output_region[0] or new_limits[0, 0]
        output_region[1] = output_region[1] or new_limits[0, 1]
        output_region[2] = output_region[2] or new_limits[1, 0]
        output_region[3] = output_region[3] or new_limits[1, 1]

        for i in range(2):
            new_limits[i, :] = (
                np.min([new_limits[i, 0], output_region[i]]),
                np.max([new_limits[i, 1], output_region[i + 2]]),
                )

        new_volume = np.zeros(np.diff(new_limits, axis=1)[:, 0].astype(np.int32))

        old_region = limits - new_limits
        new_volume[
            old_region[0, 0]:old_region[0, 0] + limits[0, 1] - limits[0, 0],
            old_region[1, 0]:old_region[1, 0] + limits[1, 1] - limits[1, 0],
            old_region[2, 0]:old_region[2, 0] + limits[2, 1] - limits[2, 0]
        ] = volume

        return new_volume, new_limits

    def __call__(self, sample, pupil=None):
        return Microscope(sample, self, pupil)



class Fluorescence(Optics):
    '''Optical device for incoherent light

    Stores optical parameters and convolves images with pupil functions.
    Treats the input image as an incoherent field. The output image is
    as such,

    .. math :: |image|^2 * |pupil|^2

    evaluated using the fourier transform.

    Parameters
    ----------
    NA
        The NA of the limiting aperatur
    wavelength
        The wavelength of the scattered light in meters
    voxel_size
        The pixel to meter conversion ratio
    refractive_index_medium
        The refractive index of the medium
    defocus
        The distance from the focal plane in meters
    upscale
        Upscales the pupil function for a more accurate result.
    ROI
        The region of the image to output (x,y,width,height). Default
        None returns entire image.
    '''
    
    def get(self, illuminated_volume, **kwargs):
        ''' Convolves the image with a pupil function
        '''
        
        limits = get_property(illuminated_volume, "limits")
        padded_volume, limits = self._pad_volume(illuminated_volume, limits=limits, **kwargs)

        z_limits = limits[2, :]

        output_image = Image(np.zeros((*padded_volume.shape[0:2], 1)))

        index_iterator = range(padded_volume.shape[2])
        z_iterator = np.linspace(z_limits[0], z_limits[1], num=padded_volume.shape[2], endpoint=False)


        for i, z in zip(index_iterator, z_iterator):
            image = padded_volume[:, :, i]

            if (image == 0).all():
                continue
            pupil = Image(self.pupil(image.shape, defocus=z, **kwargs))
            psf = np.square(np.abs(np.fft.ifft2(np.fft.fftshift(pupil))))
        
            optical_transfer_function = np.fft.fft2(psf)

            fourier_field = np.fft.fft2(image)
            convolved_fourier_field = fourier_field * optical_transfer_function

            # TODO: fft does not propagate properties correctly
            field = Image(np.fft.ifft2(convolved_fourier_field))

            # Discard remaining imaginary part (should be 0 up to rounding error)
            field = np.real(field)

            output_image[:, :, 0] += field
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))

        output_region[0] = None if output_region[0] is None else int(output_region[0] - limits[0, 0])
        output_region[1] = None if output_region[1] is None else int(output_region[1] - limits[1, 0])
        output_region[2] = None if output_region[2] is None else int(output_region[2] - limits[0, 0])
        output_region[3] = None if output_region[3] is None else int(output_region[3] - limits[1, 0])
        output_image = output_image[output_region[0]:output_region[2], output_region[1]:output_region[3]]
        output_image.properties = illuminated_volume.properties + pupil.properties
        return output_image


# HELPER FUNCTIONS
def get_property(feature, key, default=None):
    for property in feature.properties:
        if key in property:
            return property[key]
    return default


def get_position(feature, mode="center", return_z=False):

    num_outputs = 2 + return_z

    if mode == "corner":
        shift = (np.array(feature.shape) - 1)/ 2
    else:
        shift = np.array((num_outputs))

    position = get_property(feature, "position")

    if position is None:
        return position

    if len(position) == 3:
        if return_z:
            return position - shift
        else:
            return position[0:2] - shift[0:2]

    elif len(position) == 2:
        if return_z:
            return np.array([position[0], position[1], get_property(feature, "z", 0)]) - shift
        else:
            return position - shift[0:2]

    return position


def create_volume(list_of_scatterers, **kwargs):
    
    if not isinstance(list_of_scatterers, list):
        list_of_scatterers = [list_of_scatterers]
    volume = np.zeros((1, 1, 1))

    # x, y, z limits of the volume
    limits = np.array(((0, 1), (0, 1), (0, 1)))
    
    for scatterer in list_of_scatterers:

        

        padded_scatterer = Image(np.pad(scatterer, [(2, 2), (2, 2), (0, 0)], 'constant', constant_values=0))
        padded_scatterer.properties = scatterer.properties
        scatterer = padded_scatterer

        shape = np.array(scatterer.shape)


        position = get_position(scatterer, mode="corner", return_z=True)

        if position is None:
            RuntimeWarning("Optical device received a feature without a position property. It will be ignored.")
            continue

        x_pos = position[0] + np.arange(scatterer.shape[0]) 
        y_pos = position[1] + np.arange(scatterer.shape[1]) 

        target_x_pos = np.round(x_pos)
        target_y_pos = np.round(y_pos)

        for z in range(scatterer.shape[2]):
            scatterer_spline = RectBivariateSpline(x_pos, y_pos, scatterer[:, :, z])
            scatterer[1:-1, 1:-1, z] = scatterer_spline(target_x_pos[1:-1], target_y_pos[1:-1])

        position = np.round(position)
        new_limits = np.zeros(limits.shape, dtype=np.int32)
        for i in range(3):
            new_limits[i, :] = (
                np.min([limits[i, 0], position[i]]),
                np.max([limits[i, 1], position[i] + shape[i]]),
                )
            
        if not (np.array(new_limits) == np.array(limits)).all():
            new_volume = np.zeros(np.diff(new_limits, axis=1)[:, 0].astype(np.int32))
            old_region = limits - new_limits
            new_volume[
                old_region[0, 0]:old_region[0, 0] + limits[0, 1] - limits[0, 0],
                old_region[1, 0]:old_region[1, 0] + limits[1, 1] - limits[1, 0],
                old_region[2, 0]:old_region[2, 0] + limits[2, 1] - limits[2, 0]
            ] = volume
            volume = new_volume
            limits = new_limits

        within_volume_position = position - limits[:, 0]

        # NOTE: Maybe shouldn't be additive.
        volume[
            int(within_volume_position[0]):int(within_volume_position[0] + shape[0]),
            int(within_volume_position[1]):int(within_volume_position[1] + shape[1]),
            int(within_volume_position[2]):int(within_volume_position[2] + shape[2])
            ] += scatterer

    return volume, limits
