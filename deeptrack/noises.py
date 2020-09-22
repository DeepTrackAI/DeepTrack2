""" Features for introducing noise to images.

Classes
-------
Noise
    Base abstract noise class.
Offset, Background
    Adds a constant value to an image.
Gaussian
    Adds IID Gaussian noise to an image.
Poisson
    Adds Poisson-distributed noise to an image.
"""

import numpy as np
from deeptrack.features import Feature
from deeptrack.image import Image


class Noise(Feature):
    """Base abstract noise class."""


class Background(Noise):
    """Adds a constant value to an image
    Parameters
    ----------
    offset : float
        The value to add to the image
    """

    def __init__(self, offset, **kwargs):
        super().__init__(offset=offset, **kwargs)

    def get(self, image, offset, **kwargs):
        return image + offset


# ALIASES
Offset = Background


class Gaussian(Noise):
    """Adds IID Gaussian noise to an image

    Parameters
    ----------
    mu : float
        The mean of the distribution.
    sigma : float
        The root of the variance of the distribution.
    """

    def __init__(self, mu=0, sigma=1, **kwargs):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(self, image, mu, sigma, **kwargs):

        noisy_image = mu + image + np.random.randn(*image.shape) * sigma
        return noisy_image


class Poisson(Noise):
    """Adds Poisson-distributed noise to an image

    Parameters
    ----------
    snr : float
        Signal to noise ratio of the final image. The signal is determined
        by the peak value of the image.
    background : float
        Value to be be used as the background. This is used to calculate the
        signal of the image.
    """

    def __init__(self, *args, snr=100, background=0, **kwargs):
        super().__init__(*args, snr=snr, background=background, **kwargs)

    def get(self, image, snr, background, **kwargs):
        image[image < 0] = 0

        peak = np.abs(np.max(image) - background)

        rescale = snr ** 2 / peak ** 2
        noisy_image = Image(np.random.poisson(image * rescale) / rescale)
        noisy_image.properties = image.properties
        return noisy_image


## IMGAUG IMGCORRUPTLIKE
# Currently unavailable until there's a better way to implement constricted datatypes (only uint8)
# Please see https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/imgcorruptlike.py
# for source implementation

# import imgaug.augmenters as iaa
# import deeptrack as dt
# import inspect

# def init_method(self, **kwargs):
#     dt.ImgAug.__init__(self, **kwargs)

# augs = inspect.getmembers(iaa.blur, lambda x: inspect.isclass(x))

# for augname, aug in augs:

#     print(augname, aug.__module__)

#     globals()[augname] = type(augname, (aug, dt.ImgAug), {
#         "augmenter": aug,
#         "__init__": init_method})