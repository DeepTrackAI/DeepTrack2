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
from deeptrack.image import Image, pad_image_to_fft
from deeptrack.utils import as_list

from scipy.interpolate import RectBivariateSpline


class Microscope(Feature):

    __distributed__ = False

    def __init__(self, sample, objective, pupil, *args, **kwargs):
        super().__init__(*args, sample=sample, objective=objective, pupil=pupil, **kwargs)

    def get(self, image, sample=None, objective=None, pupil=None, **kwargs):
        
        new_kwargs = objective.properties.current_value_dict()
        new_kwargs.update(kwargs)
        kwargs = new_kwargs

        list_of_scatterers = sample.resolve(**kwargs)
        if not isinstance(list_of_scatterers, list):
            list_of_scatterers = [list_of_scatterers]
        
        sample_volume, limits = create_volume(list_of_scatterers, **kwargs)
        

        sample_volume = Image(sample_volume)

        for scatterer in list_of_scatterers:
            sample_volume.properties += scatterer.properties

        imaged_sample = objective.resolve(sample_volume, pupil=pupil, limits=limits)


        # Merge with input
        if not image:
            return imaged_sample

        if not isinstance(image, list):
            image = [image]
        
        for i in range(len(image)):
            image[i] += imaged_sample
            for prop in imaged_sample.properties:
                if not any([prop["hash_key"] == prop2["hash_key"] for prop2 in image[i].properties]):
                    image[i].properties.append(prop)


        return image

## OPTICAL SYSTEMS

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
              aberration=None,
              include_aberration=True,
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

        upscaled_shape = shape * upscale
        # Pupil radius
        R = NA / wavelength * np.array(voxel_size)[:2]

        x_radius = R[0] * upscaled_shape[0]
        y_radius = R[1] * upscaled_shape[1]

        x = (np.linspace(-(upscaled_shape[0] / 2), upscaled_shape[0] / 2 - 1, upscaled_shape[0])) / x_radius + 1e-8
        y = (np.linspace(-(upscaled_shape[1] / 2), upscaled_shape[1] / 2 - 1, upscaled_shape[1])) / y_radius + 1e-8

        W, H = np.meshgrid(y, x)
        RHO = W**2 + H**2
        RHO[RHO > 1] = 1
        pupil_function = ((RHO < 1) * 1.0).astype(np.complex)

        # Defocus
        z_shift = (2 * np.pi * refractive_index_medium / wavelength * voxel_size[2]
                   * np.sqrt(1 - (NA / refractive_index_medium * RHO)**2))


        # Downsample the upsampled pupil
        if upscale > 1:
            pupil_function = np.reshape(pupil_function, (shape[0], upscale, shape[1], upscale)).mean(axis=(3, 1))
            z_shift = np.reshape(z_shift, (shape[0], upscale, shape[1], upscale)).mean(axis=(3, 1))

        pupil_function[np.isnan(pupil_function)] = 0
        pupil_function[np.isinf(pupil_function)] = 0
        pupil_function_is_nonzero = pupil_function != 0

        if include_aberration:
            pupil = pupil or aberration
            if isinstance(pupil, Feature):
                pupil_function = pupil.resolve(pupil_function, **kwargs)
            elif isinstance(pupil, np.ndarray):
                pupil_function *= pupil


        pupil_functions = []
        for z in defocus:
            pupil_at_z = Image(pupil_function)
            pupil_at_z[pupil_function_is_nonzero] *= np.exp(1j * z_shift[pupil_function_is_nonzero] * z)
            pupil_functions.append(pupil_at_z)
        
        return pupil_functions


    def _pad_volume(self, volume, limits=None, padding=None, output_region=None, **kwargs):
        if limits is None:
            limits = np.zeros((3, 2))
            
        new_limits = np.array(limits)
        output_region = np.array(output_region) 

        # Replace None entries with current limit
        output_region[0] = output_region[0] if not output_region[0] is None else new_limits[0, 0]
        output_region[1] = output_region[1] if not output_region[1] is None else new_limits[0, 1]
        output_region[2] = output_region[2] if not output_region[2] is None else new_limits[1, 0]
        output_region[3] = output_region[3] if not output_region[3] is None else new_limits[1, 1]

        for i in range(2):
            new_limits[i, :] = (
                np.min([new_limits[i, 0], output_region[i] - padding[1]]),
                np.max([new_limits[i, 1], output_region[i + 2] + padding[i + 2]]),
                )
        new_volume = np.zeros(np.diff(new_limits, axis=1)[:, 0].astype(np.int32), dtype=np.complex)

        old_region = (limits - new_limits).astype(np.int32)
        limits = limits.astype(np.int32)
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
    
    def get(self, illuminated_volume, limits=None, **kwargs):
        ''' Convolves the image with a pupil function
        '''
    
        padded_volume, limits = self._pad_volume(illuminated_volume, limits=limits, **kwargs)

        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))
    
        output_region[0] = None if output_region[0] is None else int(output_region[0] - limits[0, 0] - pad[0])
        output_region[1] = None if output_region[1] is None else int(output_region[1] - limits[1, 0] - pad[1])
        output_region[2] = None if output_region[2] is None else int(output_region[2] - limits[0, 0] + pad[2])
        output_region[3] = None if output_region[3] is None else int(output_region[3] - limits[1, 0] + pad[3])
        
        padded_volume = padded_volume[output_region[0]:output_region[2], output_region[1]:output_region[3], :]
        z_limits = limits[2, :]

        output_image = Image(np.zeros((*padded_volume.shape[0:2], 1)))

        index_iterator = range(padded_volume.shape[2])
        z_iterator = np.linspace(z_limits[0], z_limits[1], num=padded_volume.shape[2], endpoint=False)

        
        zero_plane = np.all(padded_volume == 0, axis=(0, 1), keepdims=False)
        z_values = z_iterator[~zero_plane]

        volume = pad_image_to_fft(padded_volume, axes=(0, 1))

        pupils = self.pupil(volume.shape[:2], defocus=z_values, **kwargs)

        pupil_iterator = iter(pupils)

        for i, z in zip(index_iterator, z_iterator):
            
            if zero_plane[i]:
                continue

            image = volume[:, :, i]
            pupil = Image(next(pupil_iterator))

            psf = np.square(np.abs(np.fft.ifft2(np.fft.fftshift(pupil))))
        
            optical_transfer_function = np.fft.fft2(psf)

            fourier_field = np.fft.fft2(image)
            convolved_fourier_field = fourier_field * optical_transfer_function

            # TODO: fft does not propagate properties correctly
            field = Image(np.fft.ifft2(convolved_fourier_field))

            # Discard remaining imaginary part (should be 0 up to rounding error)
            field = np.real(field) 

            output_image[:, :, 0] += field[:padded_volume.shape[0], :padded_volume.shape[1]]

        
        
        output_image = output_image[pad[0]:-pad[2], pad[1]:-pad[3]]
        try:
            output_image.properties = illuminated_volume.properties + pupil.properties
        except UnboundLocalError:
            output_image.properties = illuminated_volume.properties
        
        return output_image


class Brightfield(Optics):
    '''Optical device for coherent or partially coherent light

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
    
    def get(self, illuminated_volume, limits=None, **kwargs):
        ''' Convolves the image with a pupil function
        '''
        
        padded_volume, limits = self._pad_volume(illuminated_volume, limits=limits, **kwargs)

        pad = kwargs.get("padding", (0, 0, 0, 0))
        output_region = np.array(kwargs.get("output_region", (None, None, None, None)))
    
        output_region[0] = None if output_region[0] is None else int(output_region[0] - limits[0, 0] - pad[0])
        output_region[1] = None if output_region[1] is None else int(output_region[1] - limits[1, 0] - pad[1])
        output_region[2] = None if output_region[2] is None else int(output_region[2] - limits[0, 0] + pad[2])
        output_region[3] = None if output_region[3] is None else int(output_region[3] - limits[1, 0] + pad[3])
        
        padded_volume = padded_volume[output_region[0]:output_region[2], output_region[1]:output_region[3], :]
        z_limits = limits[2, :]

        output_image = Image(np.zeros((*padded_volume.shape[0:2], 1)))

        index_iterator = range(padded_volume.shape[2])
        z_iterator = np.linspace(z_limits[0], z_limits[1], num=padded_volume.shape[2], endpoint=False)

        
        zero_plane = np.all(padded_volume == 0, axis=(0, 1), keepdims=False)
        z_values = z_iterator[~zero_plane]

        volume = pad_image_to_fft(padded_volume, axes=(0, 1))

        voxel_size = kwargs['voxel_size']

        pupils = (self.pupil(volume.shape[:2], defocus=[1], include_aberration=False, **kwargs) + 
                  self.pupil(volume.shape[:2], defocus=[-z_limits[1]], include_aberration=True, **kwargs))

        pupil_step = np.fft.fftshift(pupils[0])
        
        if "illumination" in kwargs:
            light_in = np.ones(volume.shape[:2])
            light_in = kwargs["illumination"].resolve(light_in, **kwargs)
            light_in = np.fft.fft2(light_in)
        else:
            light_in = np.zeros(volume.shape[:2])
            light_in[0, 0] = light_in.size

        K = 2*np.pi/kwargs["wavelength"]

        for i, z in zip(index_iterator, z_iterator):
            
            light_in = light_in * pupil_step

            if zero_plane[i]:
                continue

            ri_slice = volume[:, :, i]

            light = np.fft.ifft2(light_in)
            
            light_out = light * np.exp(1j * ri_slice * voxel_size[-1] * K)

            light_in = np.fft.fft2(light_out)

        light_in_focus = light_in * np.fft.fftshift(pupils[-1])

        output_image = np.fft.ifft2(light_in_focus)[:padded_volume.shape[0], :padded_volume.shape[1]]
        output_image = np.expand_dims(output_image, axis=-1)
        output_image = Image(output_image[pad[0]:-pad[2], pad[1]:-pad[3]])
        
        
        if not kwargs.get("return_field", False):
            output_image = np.square(np.abs(output_image))

        output_image.properties = illuminated_volume.properties
        
        return output_image


class IlluminationGradient(Feature):
    def get(self, image, gradient=(0, 0), vmin=0, vmax=np.inf, **kwargs):
        
        x = np.arange(image.shape[0])
        y = np.arange(image.shape[1])
        
        X, Y = np.meshgrid(x, y)
        
        amplitude = (X * gradient[0] + Y * gradient[1]) 
        image = np.clip(image + amplitude, vmin, vmax)
        
        return image
## LIGHT SOURCES

# class Light(Feature):

#     def get(self, 
#             *,
#             output_region=None,
#             effect=1, # W / m^2
#             ): 

#         intensity = np.ones((output_region[2] - output_region[0], output_region[3] - output_region[1]))


# HELPER FUNCTIONS



def _get_position(image, mode="center", return_z=False):

    num_outputs = 2 + return_z

    if mode == "corner":
        shift = (np.array(image.shape) - 1)/ 2
    else:
        shift = np.array((num_outputs))

    position = image.get_property("position")

    if position is None:
        return position

    if len(position) == 3:
        if return_z:
            return position - shift
        else:
            return position[0:2] - shift[0:2]

    elif len(position) == 2:
        if return_z:
            outp = np.array([position[0], position[1], image.get_property("z", 0)]) - shift
            return outp
        else:
            return position - shift[0:2]

    return position


def _create_volume(list_of_scatterers, 
                  pad=(0, 0, 0, 0), 
                  output_region=(None, None, None, None), 
                  refractive_index_medium=1.33,
                  **kwargs):

    

    if not isinstance(list_of_scatterers, list):
        list_of_scatterers = [list_of_scatterers]
    volume = np.zeros((1, 1, 1), dtype=np.complex)

    # x, y, z limits of the volume
    limits =np.array([(0, 1), (0, 1), (0, 1)])

    OR = np.zeros((4, ))
    for scatterer in list_of_scatterers:

        position = get_position(scatterer, mode="corner", return_z=True)

        scatterer_value = (scatterer.get_property("value") or 
                           scatterer.get_property("intensity") or 
                           scatterer.get_property("refractive_index") - refractive_index_medium or
                           1.0)

        scatterer = scatterer * scatterer_value

        if limits is None:
            limits = np.zeros((3, 2))
            limits[:, 0] = np.round(position).astype(np.int32)
            limits[:, 1] = np.round(position).astype(np.int32) + 1

        OR[0] = np.inf if output_region[0] is None else int(output_region[0] - limits[0, 0] - pad[0])
        OR[1] = -np.inf if output_region[1] is None else int(output_region[1] - limits[1, 0] - pad[1])
        OR[2] = np.inf if output_region[2] is None else int(output_region[2] - limits[0, 0] + pad[2])
        OR[3] = -np.inf if output_region[3] is None else int(output_region[3] - limits[1, 0] + pad[3])

        

        if (position[0] + scatterer.shape[0] < OR[0]  or 
            position[0] > OR[2] or
            position[1] + scatterer.shape[1] < OR[1]  or 
            position[1] > OR[3]):
            continue

        padded_scatterer = Image(np.pad(scatterer, [(2, 2), (2, 2), (0, 0)], 'constant', constant_values=0))
        padded_scatterer.properties = scatterer.properties
        scatterer = padded_scatterer

        position = get_position(scatterer, mode="corner", return_z=True)
        shape = np.array(scatterer.shape)

        if position is None:
            RuntimeWarning("Optical device received a feature without a position property. It will be ignored.")
            continue

        x_pos = position[0] + np.arange(scatterer.shape[0]) 
        y_pos = position[1] + np.arange(scatterer.shape[1]) 

        target_x_pos = np.round(x_pos)
        target_y_pos = np.round(y_pos)

        splined_scatterer = np.zeros_like(scatterer)
        for z in range(scatterer.shape[2]):
            
            scatterer_spline = RectBivariateSpline(x_pos, y_pos, np.real(scatterer[:, :, z]))
            splined_scatterer[1:-1, 1:-1, z] = scatterer_spline(target_x_pos[1:-1], target_y_pos[1:-1])
            
            if scatterer.dtype == np.complex:
                scatterer_spline = RectBivariateSpline(x_pos, y_pos, np.imag(scatterer[:, :, z]))
                splined_scatterer[1:-1, 1:-1, z] += 1j * scatterer_spline(target_x_pos[1:-1], target_y_pos[1:-1])

        scatterer = splined_scatterer
        position = np.round(position)
        new_limits = np.zeros(limits.shape, dtype=np.int32)
        for i in range(3):
            new_limits[i, :] = (
                np.min([limits[i, 0], position[i]]),
                np.max([limits[i, 1], position[i] + shape[i]]),
                )
            
        if not (np.array(new_limits) == np.array(limits)).all():
            new_volume = np.zeros(np.diff(new_limits, axis=1)[:, 0].astype(np.int32), dtype=np.complex)
            old_region = (limits - new_limits).astype(np.int32)
            limits = limits.astype(np.int32)
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
