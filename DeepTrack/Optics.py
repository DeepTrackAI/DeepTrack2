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
        return self.image(shape, Image)
    

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
        super().__init__(
            NA=NA,
            wavelength=wavelength,
            pixel_size=pixel_size,
            upscale=upscale,
            ROI=ROI
        )


    # Calculates the pupil of the optical system using the NA, wavelength and the pixel size.
    def getPupilRadius(self):
        return self.get_property("pixel_size") * self.get_property("NA") / self.get_property("wavelength")

    def getPupil(self, shape=None):
        if shape is None:
            shape = self.shape
        shape = np.array(shape)
        upscaled_shape = shape*self.get_property("upscale")
        R = self.getPupilRadius()
        x_radius = R*upscaled_shape[0]
        y_radius = R*upscaled_shape[1]
        W, H = np.meshgrid(np.arange(0, upscaled_shape[0]), np.arange(0, upscaled_shape[1]))
        
        pupilMask = ((W - upscaled_shape[0] / 2) / x_radius) ** 2  + ((H - upscaled_shape[1] / 2) / (y_radius) ) **2 <= 1
        if self.get_property("upscale") > 1:
            pupilMask = np.reshape(pupilMask, (shape[0], self.get_property("upscale"), shape[1], self.get_property("upscale"))).mean(axis=(3,1))
        pupil = pupilMask * (1 + 0j)
        return np.fft.fftshift(pupil)
    
    def image(self, shape, image):
        fourier_field = image*self.getPupil(shape=image.shape)
        # TODO: Implement custom fft2 method that correctly calls __array_wrap__.
        res = Image(np.abs(np.fft.ifft2(fourier_field)))[self.get_property("ROI")[0]:shape[0], self.get_property("ROI")[1]:shape[1]]
        res.properties=image.properties
        return res


class ZernikeAberration:
    def __init__(self,
                    Z_index,
                    Z_coefficient):
        
        assert len(Z_index) == len(Z_coefficient), "Z_index and Z_coefficient must have same length"

        
        

