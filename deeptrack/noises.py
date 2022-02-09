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

from .features import Feature
from .image import Image
from .types import PropertyLike


class Noise(Feature):
    """Base abstract noise class."""


class Background(Noise):
    """Adds a constant value to an image
    Parameters
    ----------
    offset : float
        The value to add to the image
    """

    def __init__(self, offset: PropertyLike[float], **kwargs):
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

    def __init__(
        self, mu: PropertyLike[float] = 0, sigma: PropertyLike[float] = 1, **kwargs
    ):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(self, image, mu, sigma, **kwargs):

        noisy_image = mu + image + np.random.randn(*image.shape) * sigma
        return noisy_image


class ComplexGaussian(Noise):
    def __init__(
        self, mu: PropertyLike[float] = 0, sigma: PropertyLike[float] = 1, **kwargs
    ):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(self, image, mu, sigma, **kwargs):

        noisy_image = (
            mu
            + image
            + (np.random.randn(*image.shape) + np.random.randn(*image.shape) * 1j)
            * sigma
        )
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

    def __init__(
        self,
        *args,
        snr: PropertyLike[float] = 100,
        background: PropertyLike[float] = 0,
        max_val=1e8,
        **kwargs
    ):
        super().__init__(
            *args, snr=snr, background=background, max_val=max_val, **kwargs
        )

    def get(self, image, snr, background, max_val, **kwargs):
        image[image < 0] = 0
        immax = np.max(image)
        peak = np.abs(immax - background)

        rescale = snr ** 2 / peak ** 2
        rescale = np.clip(rescale, 1e-10, max_val / np.abs(immax))
        try:
            noisy_image = Image(np.random.poisson(image * rescale) / rescale)
            noisy_image.merge_properties_from(image)
            return noisy_image
        except ValueError:
            raise ValueError(
                "Numpy poisson function errored due to too large value. Set max_val in dt.Poisson to a lower value to fix."
            )
