from DeepTrack.Distributions import sample
from DeepTrack.Features import Feature
import abc
import numpy as np

'''
Base class for the Noise object.

Creates an image of desired shape, as defined by the implementing class. 

Basic operators are overloaded to easily allow it to be added to an image 
without explicity generating a new image each time
'''

class Noise(Feature):
    pass


'''
Implementation of the Noise class to generate IID gaussian pixels.

Input arguments:
    mu:         The mean of the distribution (number, array, distribution)
    sigma       The root of the variance of the distribution. (number, array, distribution)
'''

class Gaussian(Noise):
    
    __name__ = "GaussianNoise"
    
    def __init__(self, mu=0, sigma=1, **kwargs):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(self, shape, Image, mu=0, sigma=1, **kwargs):
        mu = np.ones(shape) * mu
        sigma = np.ones(shape) * sigma
        return Image + np.random.normal(mu, sigma)
    

'''
Implementation of the Noise class to generate a random background offset.

Input arguments:
    offset:     The value of the offset (number, array, distribution)        
'''
class Offset(Noise):
    __name__ = "OffsetNoise"
    def get(self, shape, Image, offset=0, **kwargs):
        
        return Image + np.ones(shape) * offset


'''
Implementation of the Noise class to simulate photon noise, or poisson noise.

Input arguments:
    SNr:     The signal to noise ratio (number, array, distribution)      
'''
class Poisson(Noise):
    __name__ = "PoissonNoise"
    def get(self, shape, Image, SNr, **kwargs):
        peak = np.max(Image)
        rescale = SNr**2 / peak

        return np.random.poisson(Image * rescale) / rescale