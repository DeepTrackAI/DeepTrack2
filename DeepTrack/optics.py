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
from DeepTrack.features import Feature
from DeepTrack.image import Image

import matplotlib.pyplot as plt

class Optics(Feature):
    '''Abstract base class for optical devices
    
    '''
    pass

class OpticalDevice(Optics):
    '''Optical device for incoherent light

    Stores optical parameters and convolves images with pupil functions.
    Treats the input image as an incoherent field. The output image is
    as such 

    .. math :: |image|^2 * |pupil|^2

    evaluated using the fourier transform. 

    Parameters
    ----------
    NA 
        The NA of the limiting aperatur
    wavelength
        The wavelength of the scattered light in meters
    pixel_size
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
    def __init__(self,
                    NA=0.7,
                    wavelength=0.66e-6,
                    pixel_size=0.1e-6,
                    refractive_index_medium=1.33,
                    defocus=0,
                    upscale=2,
                    ROI=None,
                    **kwargs):
        
        super().__init__(
            NA=NA,
            defocus=defocus,
            wavelength=wavelength,
            refractive_index_medium=refractive_index_medium,
            pixel_size=pixel_size,
            upscale=upscale,
            ROI=ROI,
            **kwargs
        )

    def get(self, image, **kwargs):
        ''' Convolves the image with a pupil function
        '''
        
        pupil = self.pupil(image.shape, **kwargs)
        psf = np.square(np.abs(np.fft.ifft2(pupil)))
        OTF = np.fft.fft2(psf)

        fourier_field = np.fft.fft2(np.square(np.abs(image)))
        convolved_fourier_field = fourier_field * OTF

        # TODO: fft does not propagate properties correctly
        field = Image(np.fft.ifft2(convolved_fourier_field))
        field.properties = image.properties


        # Discard remaining imaginary part (should be 0 up to rounding error)
        field = np.real(field)

        field = self.extract_roi(field, **kwargs)

        return field

    # TODO: Split into smaller functions
    def pupil(self, shape, 
                NA=None,
                wavelength=None,
                refractive_index_medium=None,
                pixel_size=None,
                defocus=None,
                upscale=None,
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
        R = NA / wavelength * pixel_size
        x_radius = R * upscaled_shape[0]
        y_radius = R * upscaled_shape[1]

        x = (np.linspace(-(upscaled_shape[0] / 2), upscaled_shape[0] / 2 - 1, upscaled_shape[0])) / x_radius + 1e-8
        y = (np.linspace(-(upscaled_shape[1] / 2), upscaled_shape[1] / 2 - 1, upscaled_shape[1])) / y_radius + 1e-8
        
        W, H = np.meshgrid(x, y)
        RHO = W**2 + H**2

        # Pupil magnitude. TODO: Throws warning
        pupil = (RHO < 1) / ((1 - NA**2 / refractive_index_medium**2 * RHO)**0.25) 


        # Defocus
        z_shift = 2 * np.pi * refractive_index_medium/wavelength * (1 - NA**2 / refractive_index_medium**2 * RHO)**0.5 * pixel_size * defocus


        # Downsample the upsampled pupil
        if upscale > 1:
            pupil = np.reshape(pupil, (shape[0], upscale, shape[1], upscale)).mean(axis=(3,1))
            z_shift = np.reshape(z_shift, (shape[0], upscale, shape[1], upscale)).mean(axis=(3,1))
        

        pupil = pupil*np.exp(1j * z_shift)
        
        pupil[np.isnan(pupil)] = 0
    
        return np.fft.fftshift(pupil)


    def extract_roi(self, image, ROI=None, **kwargs):        
        if ROI is None:
            return image
        
        assert len(ROI) >= 4, "ROI should be at of length 4, got {0}".format(len(ROI)) 

        return image[ROI[0]:ROI[2], ROI[1]:ROI[3]]