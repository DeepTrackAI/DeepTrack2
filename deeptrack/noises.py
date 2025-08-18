"""
Features for introducing noise to images.

This module provides classes to add various types of noise to images, 
including constant offsets, Gaussian noise, and Poisson-distributed noise.

Module Structure
----------------
Classes:

- `Noise`: Abstract base class for noise models.
- `Background` / `Offset`: Adds a constant value to an image.
- `Gaussian`: Adds IID Gaussian noise.
- `ComplexGaussian`: Adds complex-valued Gaussian noise.
- `Poisson`: Adds Poisson-distributed noise based on signal-to-noise ratio.

Example
-------
Add Gaussian noise to an image:

>>> import numpy as np
>>> image = np.ones((100, 100))
>>> gaussian_noise = noises.Gaussian(mu=0, sigma=0.1)
>>> noisy_image = gaussian_noise.resolve(image)

Add Poisson noise with a specified signal-to-noise ratio:

>>> poisson_noise = noises.Poisson(snr=0.5)
>>> noisy_image = poisson_noise.resolve(image)

"""

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT327

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from deeptrack import Feature, Image, PropertyLike, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

__all__ = [
    "Noise",
    "Background",
    "Offset",
    "Gaussian",
    "ComplexGaussian",
    "Poisson",
]


if TYPE_CHECKING:
    import torch


class Noise(Feature):
    """Base abstract noise class."""


#TODO ***MG*** revise Background - torch, typing, docstring, unit test
class Background(Noise):
    """Adds a constant value to an image

    Parameters
    ----------
    offset : float
        The value to add to the image
    """

    def __init__(
        self: Background,
        offset: PropertyLike[float],
        **kwargs: Any,
    ):
        super().__init__(offset=offset, **kwargs)

    def get(
        self: Background,
        image: NDArray[Any] | torch.Tensor | Image,
        offset: float,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:

        return image + offset


Offset = Background


#TODO ***JH*** revise Gaussian - torch, typing, docstring, unit test
class Gaussian(Noise):
    """Adds IID Gaussian noise to an image.

    Parameters
    ----------
    mu : float
        The mean of the Gaussian distribution.
    sigma : float
        The standard deviation of the Gaussian distribution.
    """

    def __init__(
        self: Gaussian,
        mu: PropertyLike[float] = 0,
        sigma: PropertyLike[float] = 1,
        **kwargs: Any,
    ):

        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(
        self: Gaussian,
        image: NDArray[Any] | torch.Tensor | Image,
        mu: float,
        sigma: float,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:

        noisy_image = mu + image + np.random.randn(*image.shape) * sigma

        return noisy_image


#TODO ***JH*** revise ComplexGaussian - torch, typing, docstring, unit test
class ComplexGaussian(Noise):
    """Adds complex-valued IID Gaussian noise to an image.

    Parameters
    ----------
    mu : float
        The mean of the Gaussian distribution.
    sigma : float
        The standard deviation of the Gaussian distribution.
    """

    def __init__(
        self: ComplexGaussian,
        mu: PropertyLike[float] = 0,
        sigma: PropertyLike[float] = 1,
        **kwargs: Any,
    ):

        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(
        self: ComplexGaussian,
        image: NDArray[Any] | torch.Tensor | Image,
        mu: float,
        sigma: float,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:

        real_noise = np.random.randn(*image.shape)
        imag_noise = np.random.randn(*image.shape) * 1j
        noisy_image = mu + image + (real_noise + imag_noise) * sigma
 
        return noisy_image


class Poisson(Noise):
    """Adds Poisson-distributed noise to an image.

    Poisson noise is sampled and added pixel-wise depending on the
    intensity of the pixel in the original image to achieve a desired
    signal-to-noise ratio `snr`.

    Parameters
    ----------
    snr: float
        Signal-to-noise ratio of the final image. The signal is determined
        by the peak value of the image.
    background: float
        Value to be be used as the background. This is used to calculate the
        signal of the image.
    max_val: float, optional
        Maximum allowable value to prevent overflow in noise computation.
        Default is 1e8.
    """

    def __init__(
        self: Poisson,
        *args: Any,
        snr: PropertyLike[float] = 100,
        background: PropertyLike[float] = 0,
        max_val: PropertyLike[float] = 1e8,
        **kwargs,
    ):

        super().__init__(
            *args,
            snr=snr,
            background=background,
            max_val=max_val,
            **kwargs,
        )

    def _get_numpy(
        self: Poisson,
        image: NDArray[Any] | Image,
        snr: float,
        background: float,
        max_val: float,
        **kwargs: Any,
    ) -> NDArray[Any] | Image:

        image[image < 0] = 0
        image_max = np.max(image)
        peak = np.abs(image_max - background)

        rescale = snr ** 2 / peak ** 2
        rescale = np.clip(rescale, 1e-10, max_val / np.abs(image_max))
        try:
            noisy_image = Image(np.random.poisson(image * rescale) / rescale)
            noisy_image.merge_properties_from(image)
            return noisy_image
        except ValueError:
            raise ValueError(
                "NumPy poisson function errored due to too large value. "
                "Set max_val in dt.Poisson to a lower value to fix."
            )

    def _get_torch(
        self: Poisson,
        image: torch.Tensor | Image,
        snr: float,
        background: float,
        max_val: float,
        **kwargs: Any,
    ) -> torch.Tensor | Image:

        image = torch.clamp(image, min=0)
        image_max = torch.max(image)
        peak = torch.abs(image_max - background)

        rescale = snr ** 2 / peak ** 2
        rescale = torch.clamp(rescale, 1e-10, max_val / torch.abs(image_max))
        try:
            noisy_image = Image(torch.poisson(image * rescale) / rescale)
            noisy_image.merge_properties_from(image)
            return noisy_image
        except ValueError:
            raise ValueError(
                "Torch Poisson function errored due to too large value. "
                "Set max_val in dt.Poisson to a lower value to fix."
            )

    def get(
        self: Poisson,
        image: NDArray[Any] | torch.Tensor | Image,
        snr: float,
        background: float,
        max_val: float,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:

        if self.get_backend() == "numpy":
            return self._get_numpy(image, snr, background, max_val, **kwargs,)
        elif self.get_backend() == "torch":
            return self._get_torch(image, snr, background, max_val, **kwargs,)
    
