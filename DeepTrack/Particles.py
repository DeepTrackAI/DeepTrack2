

from DeepTrack.Backend.Distributions import uniform_random, draw
from DeepTrack.Backend.Image import Output
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

import abc

'''
    Base class for all particles. 
'''
class Particle(Output):
    pass

    

'''
    Implementation of the Particle class,
    Approximates the Fourier transform of the intensity-map of a 
    spherical particle using a bessel function.

    Inputs: 
        radius                  A set of particle radii (mu) that can be simulated 
        intensity               The peak field magnitude of the the particle
        position_distribution   The distribution from which to draw th particle position 
                                (May be moved to the generator)
'''
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

    # Retrieves the fourier transformed intensity map of the spherical particle.
    def get(self, 
                Image,
                Optics):

        shape = Image.shape
        pixel_size = Optics.pixel_size

        
        if self.position_distribution is None:
            position = draw(uniform_random(shape))
        else:
            position = draw(self.position_distribution)
        intensity =     draw(self.intensity)
        radius =        draw(self.radius)
        
        sampling_frequency_x = 2 * np.pi / pixel_size
        sampling_frequency_y = 2 * np.pi / pixel_size

        fx = np.arange(-sampling_frequency_x/2, sampling_frequency_x/2, step = sampling_frequency_x / shape[0])
        fy = np.arange(-sampling_frequency_y/2, sampling_frequency_y/2, step = sampling_frequency_y / shape[1])
        FX, FY = np.meshgrid(fx, fy)
        RHO = np.sqrt(FX ** 2 + FY ** 2)
        
        
        particle_field = 40 * intensity * 2 * np.pi * special.jn(1, radius * RHO) / RHO
        particle_field = particle_field * np.exp(-1j * pixel_size * (FX * (position[0] - shape[0]/2) + FY * (position[1] - shape[1]/2)))
        pupil = Optics.getPupil()
        
        convolved_field = particle_field * pupil
        particle = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(convolved_field))))
        properties = {"type": "SphericalParticle", "position": position, "radius": radius, "intensity": intensity}
        return Image + particle, properties



        
        



