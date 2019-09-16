from DeepTrack.Backend.Distributions import draw
import numpy as np

'''
Base class for the Noise object.

Creates an image of desired shape, as defined by the implementing class. 

Basic operators are overloaded to easily allow it to be added to an image 
without explicity generating a new image each time
'''

class Noise:
    __array_priority__ = 2 # Avoid Numpy array distribution on operator overlaod
    def get(self, shape):
        return 0

    def __add__(self, other):
        return other + self.get(other.shape) 

    def __sub__(self, other):
        return other - self.get(other.shape) 
    
    def __mul__(self, other):
        return other * self.get(other.shape) 
    
    def __div__(self, other):
        return other / self.get(other.shape) 
    
    __rsub__ = __sub__
    __radd__ = __add__ 


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
    
    def get(self, shape):
        mu =    np.ones(shape) * draw(self.mu)
        sigma = np.ones(shape) * draw(self.sigma)
        return np.random.normal(mu, sigma)

 

class Offset(Noise):
    def __init__(self,offset):
        self.offset = offset
    
    def get(self,shape):
        return np.ones(shape) * draw(self.offset)
