

from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Particles import Particle
from DeepTrack.Noise import Noise
from DeepTrack.Backend.Distributions import draw
import numpy as np

'''
    Base class for a generator.

    Generators combine a set of particles, an optical system and a ruleset
    to continuously create random images of particles.

    This base class convolves the intensity map of the particle with an optical pupil
    to simulate particles.

    Input arguments:
        shape           Shape of the output (tuple)
        wavelength      wavelength of the illumination source in microns (number)
        pixel_size      size of the pixels in microns (number)
        NA              the effective NA of the optical systen (number)          
'''
class Generator:
    def __init__(self,
        Optics
    ):
        self.Optics = Optics
        self.Particles = []
        self.Noise = []

    # Adds a particle to the set of particles that can be generated
    def add_particle(self, P):
        assert isinstance(P, Particle), "Argument supplied to add_particle is not an instance of Particle"
        
        self.Particles.append(P)

    def add_noise(self, N):
        assert isinstance(N, Noise), "Argument supplied to add_particle is not an instance of Noise"
        
        self.Noise.append(N)
    
    # Generates a single random image.
    def get(self, Tree):
        
        Image, Properties = Tree.resolve(self.Optics)
        
        return Image, Properties

    