'''
    Abstract base class for all particles.
'''

from abc import ABC, abstractmethod
from DeepTrack.Backend.Distributions import uniform_random
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def getIntensity():
        return 0


    
class SphericalParticle(Particle):
    def __init__(self,
            radius = 1,
            intensity = 1,
            position_distribution = None
        ):
        self.position_distribution = position_distribution

        self.radius = np.array(radius)
        if len(self.radius.shape) == 0:
            self.radius = np.array([radius])

        self.intensity = np.array(intensity)
        if len(self.intensity.shape) == 0:
            self.intensity = np.array([intensity])


    def getIntensity(self, 
                        shape,
                        NA=0.7,
                        wavelength=0.66,
                        pixel_size=0.1):
        try:
            position = self.position_distribution()
        except TypeError:
            position = np.random.rand(2)*np.array(shape[0:2])
        
        intensity =     np.random.choice(self.intensity,1)
        radius =         np.random.choice(self.radius,1) * pixel_size

        X = np.linspace(
            -pixel_size * shape[0] / 2,
            pixel_size * shape[0] / 2,
            num=shape[0],
            endpoint=True)
        
        Y = np.linspace(
            -pixel_size * shape[1] / 2,
            pixel_size * shape[1] / 2,
            num=shape[1],
            endpoint=True)

        dx = X[1] - X[0]
        dy = Y[1] - Y[0]
        
        sampling_frequency_x = 2 * np.pi / pixel_size
        sampling_frequency_y = 2 * np.pi / pixel_size

        fx = np.arange(-sampling_frequency_x/2, sampling_frequency_x/2, step = sampling_frequency_x / shape[0])
        fy = np.arange(-sampling_frequency_y/2, sampling_frequency_y/2, step = sampling_frequency_y / shape[1])
        FX, FY = np.meshgrid(fx, fy)
        RHO = np.sqrt(FX ** 2 + FY ** 2)


        particle_field = intensity * 2 * np.pi * radius * special.jn(1, radius * RHO) / RHO
        
        particle_field = particle_field * np.exp(-1j * pixel_size * (FX * (position[0] - shape[0]/2) + FY * (position[1] - shape[1]/2)))

      
        return particle_field, position



        
        



