

from DeepTrack.Distributions import uniform_random, sample
from DeepTrack.Features import Feature
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

import abc

'''
    Base class for all particles. 

    A particle will typically be defined by the fourier transform of it's intensity map. 
    This allows later classes to correctly model the optics the particle is imaged through.
'''
class Particle(Feature):

    def __input_shape__(self, shape):
        return tuple(np.array(shape)*2)

    

'''
    Implementation of the Particle class,
    Approximates the Fourier transform of the intensity-map of a 
    point particle as a constant.

    Inputs: 
        radius                  A set of particle radii (mu) that can be simulated 
        intensity               The peak field magnitude of the the particle
        position                The distribution from which to draw the particle position 
                                (May be moved to the generator)

    Properties
        position                Position of the particle [x,y(,z)], in px.
        intensity               The peak of the unaborrated particle intensity (a.u) 

'''

class PointParticle(Particle):
    __name__ = "PointParticle"
    def get(self,
                shape,
                Image,
                position=None,
                intensity=None,
                Optics=None,
                **kwargs):
        out_shape = np.array(shape) * 2
        shift = _get_particle_shift(position, out_shape, Optics)
        particle_field = intensity * np.exp(shift)
        particle = np.fft.ifftshift(particle_field)
        return Image + particle



'''
    Implementation of the Particle class,
    Approximates the Fourier transform of the intensity-map of a 
    spherical particle using a bessel function.

    Inputs: 
        radius                  A set of particle radii (mu) that can be simulated 
        intensity               The peak field magnitude of the the particle
        position                The distribution from which to draw the particle position

    Properties
        position                Position of the particle [x,y(,z)], in px.
        radius                  The radius of the particle (m) 
        intensity               The peak of the unaborrated particle intensity (a.u) 

'''


class SphericalParticle(Particle):
    __name__ = "SphericalParticle"
        
    # Retrieves the fourier transformed intensity map of the spherical particle.
    def get(self,
                shape,
                Image,
                position=None,
                intensity=None,
                radius=None,
                Optics=None,
                **kwargs):
        out_shape = np.array(shape) * 2
        pixel_size = Optics.get_property("pixel_size")
        
        sampling_frequency_x = 2 * np.pi / pixel_size
        sampling_frequency_y = 2 * np.pi / pixel_size

        fx = np.arange(-sampling_frequency_x/2, sampling_frequency_x/2, step = sampling_frequency_x / out_shape[0])
        fy = np.arange(-sampling_frequency_y/2, sampling_frequency_y/2, step = sampling_frequency_y / out_shape[1])
        FX, FY = np.meshgrid(fx, fy)
        RHO = np.sqrt(FX ** 2 + FY ** 2)
        
        particle_field = self.get_property("intensity") * 2 * special.jn(1, self.get_property("radius") * RHO) / (RHO * self.get_property("radius"))

        shift = _get_particle_shift(self.get_property("position"), out_shape, Optics)
        
        particle_field = particle_field * np.exp(shift)
        particle = np.fft.ifftshift(particle_field)
        return Image + particle


'''
    Calculates a complex field that, when element-wise multiplied with
    a fourier field, move each point in real space by a 3d vector.

    Inputs:
        position                The direction vector to shift each point with
        shape                   The shape of the output field
        Optics                  The optical system of the image (Only wavelength needed, and only if z=/=0)
'''
def _get_particle_shift(position, shape, Optics):
    sampling_frequency_x = 2 * np.pi / Optics.get_property("pixel_size")
    sampling_frequency_y = 2 * np.pi / Optics.get_property("pixel_size")

    fx = np.arange(-sampling_frequency_x/2, sampling_frequency_x/2, step = sampling_frequency_x / shape[0])
    fy = np.arange(-sampling_frequency_y/2, sampling_frequency_y/2, step = sampling_frequency_y / shape[1])
    FX, FY = np.meshgrid(fx, fy)
    RHO = np.sqrt(FX ** 2 + FY ** 2)
    
    shift = -1j * Optics.get_property("pixel_size") * (FX * (position[0]) + FY * (position[1]))
    if len(position) >= 3:
        k = 2 * np.pi / Optics.get_property("wavelength")
        K_MAT = k ** 2 - RHO ** 2
        K_MAT[K_MAT < 0] = 0
        K_MAT = np.sqrt(K_MAT)
        shift = shift + 1j * position[2] * Optics.get_property("pixel_size") * (K_MAT-k)
    return shift

