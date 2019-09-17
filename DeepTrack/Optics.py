import numpy as np
from abc import ABC, abstractmethod
'''
    An optical device that generates the pupil based on input parameters.
    The base device is stateless, but will be extended to allow for aberrations.
'''


class BaseOpticalDevice2D:
    def __init__(self,
                    shape,
                    NA=0.7,
                    wavelength=0.66,
                    pixel_size=0.1):
        self.shape = shape
        self.NA = NA
        self.wavelength = wavelength
        self.pixel_size = pixel_size
    # Calculates the pupil of the optical system using the NA, wavelength and the pixel size.
    def getPupilRadius(self):
        return self.pixel_size * self.NA / self.wavelength

    def getPupil(self, shape=None):
        if shape is None:
            shape = self.shape
        
        R = self.getPupilRadius()
        x_radius = R*shape[0]
        y_radius = R*shape[1]
        W, H = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))

        pupilMask = ((W - shape[0] / 2) / x_radius) ** 2  + ((H - shape[1] / 2) / (y_radius) ) **2 <= 1
        
        pupil = pupilMask * (1 + 0j)
        return pupil

        

