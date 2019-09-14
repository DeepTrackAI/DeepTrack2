from DeepTrack.Backend.Distributions import draw
import numpy as np

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
    
    __radd__ = __add__ 

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
