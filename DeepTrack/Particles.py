

from DeepTrack.Backend.Distributions import uniform_random, draw
from DeepTrack.Backend.Image import Feature
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

import abc

'''
    Base class for all particles. 
'''
class Particle(Feature):
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

class PointParticle(Particle):
    def __init__(self, 
            intensity=1,
            position_distribution=None):
        self.position_distribution = position_distribution
        self.intensity = intensity

    def get(self,
                Image,
                Optics):
        
        out_shape = Image.shape
        shape = np.array(out_shape) * 2
        if self.position_distribution is None:
            position = draw(uniform_random(out_shape))
        else:
            position = draw(self.position_distribution)
        intensity =     draw(self.intensity)

        shift = _get_particle_shift(position, shape, Optics)

        particle_field = intensity * np.exp(shift)

        pupil = Optics.getPupil(shape)
        
        convolved_field = particle_field * pupil
        particle = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(convolved_field)))[0:out_shape[0], 0:out_shape[1]])

        properties = {"type": "PointParticle", "position": position, "intensity": intensity}

        return Image + particle, properties


class SphericalParticle(Particle):
    def __init__(self,
            radius=1,
            intensity=1,
            position_distribution=None
        ):
        
        self.position_distribution = position_distribution
        self.radius = radius
        self.intensity = intensity
        

    # Retrieves the fourier transformed intensity map of the spherical particle.
    def get(self, 
                Image,
                Optics):
        out_shape = Image.shape
        shape = 2 * np.array(Image.shape)
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
        
        particle_field = intensity * 2 * special.jn(1, radius * RHO) / (RHO * radius)

        shift = _get_particle_shift(position, shape, Optics)
        
        particle_field = particle_field * np.exp(shift)
        pupil = Optics.getPupil(shape)
        
        convolved_field = particle_field * pupil
        particle = (np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(convolved_field))))[0:out_shape[0], 0:out_shape[1]])
        properties = {"type": "SphericalParticle", "position": position, "radius": radius, "intensity": intensity}
        return Image + particle, properties



        
        


def _get_particle_shift(position, shape, Optics):
    sampling_frequency_x = 2 * np.pi / Optics.pixel_size
    sampling_frequency_y = 2 * np.pi / Optics.pixel_size

    fx = np.arange(-sampling_frequency_x/2, sampling_frequency_x/2, step = sampling_frequency_x / shape[0])
    fy = np.arange(-sampling_frequency_y/2, sampling_frequency_y/2, step = sampling_frequency_y / shape[1])
    FX, FY = np.meshgrid(fx, fy)
    RHO = np.sqrt(FX ** 2 + FY ** 2)
    
    shift = -1j * Optics.pixel_size * (FX * (position[0] - shape[0]/2) + FY * (position[1] - shape[1]/2))
    if len(position) >= 3:
        k = 2 * np.pi / Optics.wavelength
        K_MAT = k ** 2 - RHO ** 2
        K_MAT[K_MAT < 0] = 0
        K_MAT = np.sqrt(K_MAT)
        shift = shift + 1j * position[2] * Optics.pixel_size * (K_MAT-k)
    return shift