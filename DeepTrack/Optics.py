import numpy as np
from abc import ABC, abstractmethod
from DeepTrack.Backend.Image import Feature, Image
import matplotlib.pyplot as plt
'''
    An optical device that images a fourier representation of the field. 
'''
class Optics(Feature):
    def image(self, shape, Image):
        pass

    def __call__(self, Features):
        return Features + self 

    def __resolve__(self, shape, **kwargs):
        kwargs["Optics"] = self 
        return super().__resolve__(shape, **kwargs)

    def get(self, shape, Image, **kwargs):
        return self.image(shape, Image), self.__getproperties__()
    
    def __getproperties__(self):
        return {}

class BaseOpticalDevice2D(Optics):
    def __init__(self,
                    shape,
                    NA=0.7,
                    wavelength=0.66,
                    pixel_size=0.1,
                    upscale=2,
                    ROI = None):
        if ROI is None:
            ROI = (0,0)
        
        self.shape = shape
        self.NA = NA
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.upscale = upscale
        self.ROI = ROI


    # Calculates the pupil of the optical system using the NA, wavelength and the pixel size.
    def getPupilRadius(self):
        return self.pixel_size * self.NA / self.wavelength

    def getPupil(self, shape=None):
        if shape is None:
            shape = self.shape
        shape = np.array(shape)
        upscaled_shape = shape*self.upscale
        R = self.getPupilRadius()
        x_radius = R*upscaled_shape[0]
        y_radius = R*upscaled_shape[1]
        W, H = np.meshgrid(np.arange(0, upscaled_shape[0]), np.arange(0, upscaled_shape[1]))
        
        pupilMask = ((W - upscaled_shape[0] / 2) / x_radius) ** 2  + ((H - upscaled_shape[1] / 2) / (y_radius) ) **2 <= 1
        if self.upscale > 1:
            pupilMask = np.reshape(pupilMask, (shape[0], self.upscale, shape[1], self.upscale)).mean(axis=(3,1))
        pupil = pupilMask * (1 + 0j)
        return np.fft.fftshift(pupil)
    
    def image(self, shape, image):
        fourier_field = image*self.getPupil(shape=image.shape)
        # TODO: Implement custom fft2 method that correctly calls __array_wrap__.
        res = Image(np.abs(np.fft.ifft2(fourier_field)))[self.ROI[0]:shape[0], self.ROI[1]:shape[1]]
        res.properties=image.properties
        return res

    def __getproperties__(self):
        return {"type":"BaseOpticalDevice2D", "NA": self.NA, "wavelength": self.wavelength, "pixel_size": self.pixel_size}


class ZernikeAberration:
    def __init__(self,
                    Z_index,
                    Z_coefficient):
        
        assert len(Z_index) == len(Z_coefficient), "Z_index and Z_coefficient must have same length"

        
        

