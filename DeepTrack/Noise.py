from DeepTrack.Backend.Distributions import draw
from DeepTrack.Backend.Image import Feature
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
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def get(self, shape, Image, Optics=None):
        mu_val = draw(self.mu)
        sigma_val = draw(self.sigma)
        mu =    np.ones(shape) * mu_val
        sigma = np.ones(shape) * sigma_val
        return Image + np.random.normal(mu, sigma), {"type": "Gaussian", "mu": mu_val, "sigma": sigma_val}

 
'''
Implementation of the Noise class to generate a random background offset.

Input arguments:
    offset:     The value of the offset (number, array, distribution)        
'''
class Offset(Noise):
    def __init__(self,offset):
        self.offset = offset
    
    def get(self, shape, Image, Optics=None):
        offset = draw(self.offset)
        return Image + np.ones(shape) * offset, {"type": "Offset", "offset": offset}
