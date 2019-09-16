

from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Particles import Particle
from DeepTrack.Noise import Noise
from DeepTrack.Backend.Distributions import draw
import numpy as np

'''
    Base class for a generator.

    Generators combine a set of particles, an optical system and a ruleset
    to continuously create random images of particles.

    The base convolves the intensity map of the particle with an optical pupil
    to simulate particles.

    Input arguments:
        shape           Shape of the output (tuple)
        wavelength      wavelength of the illumination source in microns (number)
        pixel_size      size of the pixels in microns (number)
        NA              the effective NA of the optical systen (number)          
'''
class Generator:
    def __init__(self,
        shape=(64,64),
        wavelength = 0.68,
        pixel_size = 0.1,
        NA = 0.7
    ):
        self.shape = np.array(shape)
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.NA = NA
        self.OpticalDevice = BaseOpticalDevice2D()
        self.Particles = []
        self.Noise = []

    # Adds a particle to the set of particles that can be generated
    def add_particle(self, P):
        assert isinstance(P, Particle), "Argument supplied to add_particle is not an instance of Particle"
        
        self.Particles.append(P)

    def add_noise(self, N):
        assert isinstance(N, Noise), "Argument supplied to add_particle is not an instance of Particle"
        
        self.Noise.append(N)
    
    # Generates a single random image.
    def get(self):
        assert len(self.Particles) != 0, "Generator needs to have at least one particle. Add one using add_particle"

        Particle = draw(self.Particles)

        I, position = Particle.getIntensity(self.shape * 2,
                                        wavelength = self.wavelength,
                                        pixel_size = self.pixel_size,
                                        NA         = self.NA)

        Pupil = self.OpticalDevice.getPupil(self.shape * 2,
                                        wavelength = self.wavelength,
                                        pixel_size = self.pixel_size,
                                        NA         = self.NA)

        assert I.shape == Pupil.shape, "The output shape of the optical device and the particle needs to match"

        PhaseMask = I * Pupil
        AbsField = np.array(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(PhaseMask))))[:self.shape[0], :self.shape[1]])
        
        for N in self.Noise:
            AbsField += N
        
        return AbsField, position