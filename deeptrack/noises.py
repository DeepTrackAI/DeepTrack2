''' Features for introducing noise to images

Classes
-------
Noise
    Base abstract noise class
Offset, Add, Background
    Adds a constant value to an image
Gaussian
    Adds IID Gaussian noise to an image
Poisson
    Adds Poisson-distributed noise to an image
'''

import numpy as np
from deeptrack.features import Feature
from deeptrack.image import Image




class Noise(Feature):
    '''Base abstract noise class
    '''



class Add(Noise):
    ''' Adds a constant value to an image
    Parameters
    ----------
    offset : float
        The value to add to the image
    '''
    def __init__(self, offset, **kwargs):
        super().__init__(offset=offset, **kwargs)

    def get(self, image, offset, **kwargs):
        return image + offset
# ALIASES
Offset = Add
Background = Add


class Gaussian(Noise):
    '''Adds IID Gaussian noise to an image

    Parameters
    ----------
    mu : float
        The mean of the distribution.
    sigma : float
        The root of the variance of the distribution.
    '''
    def __init__(self, *args, mu, sigma, **kwargs):
        super().__init__(*args, mu=mu, sigma=sigma, **kwargs)

    def get(self, image, mu, sigma, **kwargs):
        mu = np.ones(image.shape) * mu
        sigma = np.ones(image.shape) * sigma
        noisy_image = image + np.random.normal(mu, sigma)
        return noisy_image



class Poisson(Noise):
    '''Adds Poisson-distributed noise to an image

    Parameters
    ----------
    snr : float
        Signal to noise ratio of the final image. The signal is determined
        by the peak value of the image.
    '''
    def get(self, image, snr=None, **kwargs):
        image[image < 0] = 0
        peak = np.max(image)
        rescale = snr**2 / peak
        noisy_image = Image(np.random.poisson(image * rescale) / rescale)
        noisy_image.properties = image.properties
        return noisy_image
