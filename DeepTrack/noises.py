
from DeepTrack.features import Feature
from DeepTrack.image import Image
import abc
import numpy as np



class Noise(Feature):
    '''Base class for the Noise object.

    Creates an image of desired shape, as defined by the implementing class. 

    Basic operators are overloaded to easily allow it to be added to an image 
    without explicity generating a new image each time
    '''
    pass



class Gaussian(Noise):
    '''Adds gaussian noise to image
    Implementation of the Noise class to generate IID gaussian pixels.

    Parameters
    ----------
    mu         
        The mean of the distribution.
    sigma       
        The root of the variance of the distribution.
    '''
    __name__ = "GaussianNoise"
    def __init__(self, mu=0, sigma=1, **kwargs):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(self, shape, Image, mu=0, sigma=1, **kwargs):
        mu = np.ones(shape) * mu
        sigma = np.ones(shape) * sigma
        return Image + np.random.normal(mu, sigma)

 

class Offset(Noise):
    __name__ = "OffsetNoise"
    def get(self, Image, offset=0, **kwargs):  
        return Image + offset


class Poisson(Noise):
    def get(self, image, SNr=None, **kwargs):
        peak = np.max(image)
        rescale = SNr**2 / peak
        noised_image = Image(np.random.poisson(image * rescale) / rescale)
        noised_image.properties = image.properties
        return noised_image