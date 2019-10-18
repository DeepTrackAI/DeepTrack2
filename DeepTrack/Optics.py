import numpy as np
from abc import ABC, abstractmethod
from DeepTrack.Image import Image
from DeepTrack.Features import Feature
import matplotlib.pyplot as plt
'''
    An optical device that images a fourier representation of the field. 

    Main responsibility is storing parameters of the optical system, as well
    as imaging an intensity map.
'''
class Optics(Feature):
    '''
        The function called to image an intensity map.
    '''
    def image(self, shape, Feature, **kwargs):
        if self.mode == "coherent":
            Object = Feature.fourier_field(shape, **kwargs)
            return self.pupil(Object.shape, kwargs["position"]) * Object
        else:
            Object = Feature.squared_fourier_field(shape, **kwargs)
            pupil = self.squared_pupil(Object.shape, kwargs["position"])
            return pupil * Object

    def finalize(self, shape, image):
        
        
        ROI = self.get_property("ROI")
        
        res = Image(np.fft.ifft2(image))[ROI[0]:shape[0], ROI[1]:shape[1]]
        
        if self.mode == "coherent":
            res = np.abs(np.square(res))

        res = np.real(res) # The imaginary part should be essentially zero at this point.
        res.properties=image.properties

        return res

    def squared_pupil(self,shape, position, **kwargs):
        psf = np.fft.ifft2(self.pupil(shape, position, **kwargs))
        return np.fft.fft2(np.square(np.abs(psf)))
        
    '''
        Alternative and perhaps more intuitive syntax to add an optical 
        system to a Feature tree. Equivalent to Optics + (F1 + ... + Fn)
    '''
    def __call__(self, Features):
        return Features + self 
    
    def __resolve__(self, shape, **kwargs):
        kwargs["Optics"] = self 
        return super().__resolve__(shape, **kwargs)

    def get(self, shape, Image, **kwargs):
        return self.finalize(shape, Image)
    

class BaseOpticalDevice2D(Optics):
    def __init__(self,
                    NA=0.7,
                    wavelength=0.66e-6,
                    refractive_index_medium=1.33,
                    pixel_size=0.1e-6,
                    upscale=2,
                    mode="incoherent",
                    ROI = (0,0)):
        
        
        if ROI is None:
            ROI = (0,0)

        # These will not be randomized.
        self.mode = mode
        self.upscale = upscale
        
        super().__init__(
            NA=NA,
            wavelength=wavelength,
            refractive_index_medium=refractive_index_medium,
            pixel_size=pixel_size,
            ROI=ROI
        )


    # Calculates the pupil of the optical system using the NA, wavelength and the pixel size.
    def getPupilRadius(self):
        return 

    # TODO: Split into smaller functions
    def pupil(self, shape, position):
        shape = np.array(shape)
        upscaled_shape = shape*self.upscale
        NA = self.get_property("NA")
        n = self.get_property("refractive_index_medium")
        wavelength = self.get_property("wavelength")
        pixel_size = self.get_property("pixel_size")
        R = NA / wavelength * pixel_size
        x_radius = R*upscaled_shape[0]
        y_radius = R*upscaled_shape[1]

        x = (np.linspace(-(upscaled_shape[0]/2), upscaled_shape[0] /2 - 1, upscaled_shape[0])) / x_radius + 1e-8
        y = (np.linspace(-(upscaled_shape[1]/2), upscaled_shape[1] /2 - 1, upscaled_shape[1])) / y_radius + 1e-8
        
        W, H = np.meshgrid(x,y)
        RHO = W**2 + H**2

        # Magnitude
        pupil = (RHO < 1) / ((1 - NA**2 / n**2 * RHO) ** 0.25) 

        # Plane shift
        plane_shift = -2*np.pi * R * (position[0] * W + position[1] * H)

        # Latitudal shift
        lat_shift = 2*np.pi*n/wavelength * (1 - NA**2 / n**2 * RHO) ** 0.5 * pixel_size * position[2]

        if self.upscale > 1:
            pupil = np.reshape(pupil, (shape[0], self.upscale, shape[1], self.upscale)).mean(axis=(3,1))
            plane_shift = np.reshape(plane_shift, (shape[0], self.upscale, shape[1], self.upscale)).mean(axis=(3,1))
            lat_shift = np.reshape(lat_shift, (shape[0], self.upscale, shape[1], self.upscale)).mean(axis=(3,1))
        
        pupil = pupil*np.exp(1j * (plane_shift + lat_shift))
        
        pupil[np.isnan(pupil)] = 0
    
        return np.fft.fftshift(pupil)

class ZernikeAberration:
    def __init__(self,
                    Z_index,
                    Z_coefficient):
        
        assert len(Z_index) == len(Z_coefficient), "Z_index and Z_coefficient must have same length"
