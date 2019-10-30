

from DeepTrack.features import Feature
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

import abc


class Scatterer(Feature):
    '''
    Base class for all particles. 

    A particle will typically be defined by the fourier transform of it's intensity map. 
    This allows later classes to correctly model the optics the particle is imaged through.
    ''' 

    def translate_by(self, image, position):
        
        X, Y = np.meshgrid(
            np.arange(
                -image.shape[0] / 2, 
                image.shape[0] / 2
            ),
            np.arange(
                -image.shape[1] / 2, 
                image.shape[1] / 2
            )
        )

        return image * np.exp(1j * (X * position[0] + Y * position[1]))



class PointParticle(Scatterer):
    '''
    Implementation of the Particle class,
    Approximates the Fourier transform of the intensity-map of a 
    point particle as a constant.

    Inputs: 
        intensity               The peak field magnitude of the the particle
        position                The distribution from which to draw the particle position 
                                (May be moved to the generator)

    Properties
        position                Position of the particle [x,y(,z)], in px.
        intensity               The peak of the unaborrated particle intensity (a.u) 

    '''
    __name__ = "PointParticle"
    def get(self,
                image,
                position=None,
                intensity=None,
                **kwargs):
        particle = np.ones(image.shape) * intensity
        particle = self.translate_by(particle, np.array(position) - np.array(image.shape) / 2)
        return image + particle





# class ElipticalSource(Scatterer):
#     '''
#     Implementation of the Particle class,
#     Approximates the Fourier transform of the intensity-map of a 
#     spherical particle using a bessel function.

#     Inputs: 
#         radius                  A set of particle radii (mu) that can be simulated 
#         intensity               The peak field magnitude of the the particle
#         position                The distribution from which to draw the particle position

#     Properties
#         position                Position of the particle [x,y(,z)], in px.
#         radius                  The radius of the particle (m) 
#         intensity               The peak of the unaborrated particle intensity (a.u) 

#     '''
#     __name__ = "SphericalParticle"
        
#     # Retrieves the fourier transformed intensity map of the spherical particle.
#     def fourier_field(self,
#                 shape,
#                 intensity=None,
#                 radius=None,
#                 Optics=None,
#                 **kwargs): 
#         out_shape = np.array(shape) * 2
#         pixel_size = Optics.get_property("pixel_size")
#         sampling_frequency_x = 2 * np.pi / pixel_size
#         sampling_frequency_y = 2 * np.pi / pixel_size
#         fx = np.arange(-sampling_frequency_x/2, sampling_frequency_x/2, step = sampling_frequency_x / out_shape[0])
#         fy = np.arange(-sampling_frequency_y/2, sampling_frequency_y/2, step = sampling_frequency_y / out_shape[1])
#         FX, FY = np.meshgrid(fx, fy)
#         RHO = np.sqrt(FX ** 2 + FY ** 2)

#         factor = np.sqrt(intensity) * 2 * (radius / pixel_size) ** 2

#         particle_field = factor * special.jn(1, radius * RHO) / (RHO * radius)
#         particle = np.fft.ifftshift(particle_field)
#         return particle

#     def squared_fourier_field(self, shape, intensity=None, **kwargs):
#         return self.fourier_field(shape, intensity=intensity, **kwargs) * np.sqrt(intensity)

