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
    def getPupil(self):

        X = np.linspace(
            -self.pixel_size * self.shape[0] / 2,
            self.pixel_size * self.shape[0] / 2,
            num=self.shape[0],
            endpoint=True)
        
        Y = np.linspace(
            -self.pixel_size * self.shape[1] / 2,
            self.pixel_size * self.shape[1] / 2,
            num=self.shape[1],
            endpoint=True)

        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        
        sampling_frequency_x = 1/dx
        sampling_frequency_y = 1/dy

        x_radius = self.NA / (self.wavelength * sampling_frequency_x / self.shape[0])
        y_radius = self.NA / (self.wavelength * sampling_frequency_y / self.shape[1])

        W, H = np.meshgrid(np.arange(0, self.shape[0]), np.arange(0, self.shape[1]))

        pupilMask = ((W - self.shape[0] / 2) / x_radius) ** 2  + ((H - self.shape[1] / 2) / (y_radius) ) **2 <= 1
        
        pupil = pupilMask * (1 + 0j)
        return pupil

        

