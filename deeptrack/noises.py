"""Features for introducing noise to images.

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

from deeptrack import Feature, PropertyLike, TORCH_AVAILABLE


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
    """Base abstract noise class operating on array domains."""

    def _process_and_get(self, inputs, **properties):
        """Override Feature._process_and_get.
         
        Transparently supports ScatteredVolume / ScatteredField objects.

        Subclasses must implement `get(array, **properties)` and
        return a numpy array or torch tensor.
        """

        results = []

        # Lazy import avoids circular dependency
        try:
            from deeptrack.scatterers import ScatteredVolume, ScatteredField
            scattered_types = (ScatteredVolume, ScatteredField)
        except Exception:
            scattered_types = ()

        for x in inputs:

            # --- unwrap if scattered ---
            if scattered_types and isinstance(x, scattered_types):
                obj = x.copy()
                arr = obj.array
            else:
                obj = None
                arr = x

            # --- apply noise on array ---
            out = self.get(arr, **properties)

            # --- rewrap if needed ---
            if obj is not None:
                obj.array = out
                results.append(obj)
            else:
                results.append(out)

        return results


class Background(Noise):
    """Add a constant value to an image.

    Parameters
    ----------
    offset: float
        The value to add to the image.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Noise` class.

    Methods
    -------
    get(
        image: np.ndarray | torch.Tensor
        offset: float,
        **kwargs,
    ) -> np.ndarray | torch.Tensor
        Adds the constant offset to the input image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image with zeros:
    >>> import numpy as np
    >>>
    >>> input_image = np.zeros((2,2))

    Define the Background noise feature with offset 0.5:
    >>> noise = dt.Background(offset=0.5)

    Apply the noise to the input image and print the resulting image:
    >>> output_image = noise.resolve(input_image)
    >>> print(output_image)
    [[0.5 0.5]
    [0.5 0.5]]

    """

    def __init__(
        self: Background,
        offset: PropertyLike[float],
        **kwargs: Any,
    ):
        """Initialize the Background noise feature.

        Parameters
        ----------
        offset: PropertyLike[float]
            The constant value to be added to the image.
        **kwargs: Any
            Additional arguments passed to the parent `Noise` class.
        """
        super().__init__(offset=offset, **kwargs)

    def get(
        self: Background,
        image: np.ndarray | torch.Tensor,
        offset: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Add the given offset to the image.

        Parameters
        ----------
        image: np.ndarray | torch.Tensor
            The input image.
        offset: float
            The value to add to the image.

        Returns
        -------
        np.ndarray | torch.Tensor
            The image with offset added.
        """

        return image + offset

Offset = Background


class Gaussian(Noise):
    """Add IID Gaussian noise to an image.

    Gaussian noise is sampled from a Gaussian distribution and added pixel-wise
    to the input image.

    Parameters
    ----------
    mu: float
        The mean of the Gaussian distribution.
    sigma: float
        The standard deviation of the Gaussian distribution.

    Notes
    -----
    If the backend is NumPy, the calculations use NumPy-compatible functions,
    and the output will be a np.array. If the backend is PyTorch, the
    calculations use PyTorch-compatible functions, and the output will be a
    torch.Tensor.

    Methods
    -------
    get(
        image: np.ndarray | torch.Tensor,
        snr: float,
        background: float,
        max_val: float, optional,
        **kwargs,
    ) -> np.ndarray | torch.Tensor
        Returns an image with Gaussian noise added.

    Examples
    --------
    Add Gaussian noise to an image.
    >>> import deeptrack as dt

    Create an input image with constant values:
    >>> import numpy as np
    >>>
    >>> input_image = np.ones((2,2)) * 3
    
    Define the Gaussian noise feature with mean 1 and standard deviation 0.1:
    >>> noise = dt.Gaussian(mu=1, sigma=0.1)

    Apply the noise to the input image and print the resulting image:
    >>> output_image = noise.resolve(input_image)
    >>> print(output_image)
    [[4.01965863 4.20688642]
    [4.02184982 3.87875873]]

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
        image: np.ndarray | torch.Tensor,
        mu: float,
        sigma: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        # For a Numpy backend.
        if self.get_backend() == "numpy":
            noisy_image = mu + image + np.random.randn(*image.shape) * sigma

        # For a Torch backend.
        elif self.get_backend() == "torch":
            noisy_image = (
                mu
                + image
                + torch.randn(*image.shape, device=image.device) * sigma
            )

        return noisy_image


class ComplexGaussian(Noise):
    """Add complex-valued IID Gaussian noise to an image.

    Complex Gaussian noise is sampled by combining two independent Gaussian
    distributions for real and imaginary values and is then added pixel-wise
    to the input image.

    Parameters
    ----------
    mu: float
        The mean of the Gaussian distribution.
    sigma: float
        The standard deviation of the Gaussian distribution.

    Notes
    -----
    If the backend is NumPy, the calculations use NumPy-compatible functions,
    and the output will be a np.array. If the backend is PyTorch, the
    calculations use PyTorch-compatible functions, and the output will be a
    torch.Tensor.

    Methods
    -------
    get(
        image: np.ndarray | torch.Tensor,
        snr: float,
        background: float,
        max_val: float, optional,
        **kwargs,
    ) -> np.ndarray | torch.Tensor
        Returns an image with complex Gaussian noise added.

    Examples
    --------
    Add complex Gaussian noise to an image.

    >>> import deeptrack as dt

    Create an input image with constant values:
    >>> import numpy as np
    >>>
    >>> input_image = np.ones((2,2)) * 3
    
    Define the Gaussian noise feature with mean 1 and standard deviation 0.1:
    >>> noise = dt.ComplexGaussian(mu=1, sigma=0.1)

    Apply the noise to the input image and print the resulting image:
    >>> output_image = noise.resolve(input_image)
    >>> print(output_image)
    [[3.79975648-0.06967551j 4.09943404+0.06499738j]
    [3.99886747-0.23549974j 4.15725117-0.07847024j]]

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
        image: np.ndarray | torch.Tensor,
        mu: float,
        sigma: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        # For a Numpy backend.
        if self.get_backend() == "numpy":
            real_noise = np.random.randn(*image.shape)
            imag_noise = np.random.randn(*image.shape) * 1j
            noisy_image = mu + image + (real_noise + imag_noise) * sigma

        # For a Torch backend.
        elif self.get_backend() == "torch":
            real_noise = torch.randn(*image.shape, device=image.device)
            imag_noise = torch.randn(*image.shape, device=image.device) * 1j
            noisy_image = mu + image + (real_noise + imag_noise) * sigma

        return noisy_image


class Poisson(Noise):
    """Add Poisson-distributed noise to an image.

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

    Notes
    -----
    If the backend is NumPy, the calculations use NumPy-compatible functions,
    and the output will be a np.array. If the backend is PyTorch, the
    calculations use PyTorch-compatible functions, and the output will be a
    torch.Tensor.

    Methods
    -------
    get(
        image: np.ndarray | torch.Tensor
        snr: float,
        background: float,
        max_val: float, optional,
        **kwargs,
    ) -> np.ndarray | torch.Tensor
        Returns an image with Poisson noise added.

    Examples
    --------
    Add Poisson noise to an image.

    >>> import deeptrack as dt

    Create an input image with ones:
    >>> import numpy as np
    >>>
    >>> input_image = np.ones((2,2))
    
    Define the Poisson noise feature with a low SNR:
    >>> noise = dt.Poisson(snr=1)

    Apply the noise to the input image and print the resulting image:
    >>> output_image = noise.resolve(input_image)
    >>> print(output_image)
    [[2. 1.]
    [0. 4.]]

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

    def get(
        self: Poisson,
        image: np.ndarray | torch.Tensor | ScatteredVolume | ScatteredField,
        snr: float,
        background: float,
        max_val: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor | ScatteredVolume | ScatteredField:
        
        # # --- unwrap scattered objects ---
        # is_scattered = isinstance(image, (ScatteredVolume, ScatteredField))
        # if is_scattered:
        #     obj = image.copy()
        #     image = obj.array  # work on underlying array

        backend = self.get_backend()

        if backend == "numpy":
            image = np.clip(image, 0, None)
            image_max = np.max(image)
            peak = np.abs(image_max - background)

            rescale = snr**2 / peak**2
            rescale = np.clip(
                rescale, 1e-10, max_val / np.abs(image_max)
            )

            noisy = np.random.poisson(image * rescale) / rescale

        elif backend == "torch":
            image = torch.clamp(image, min=0)
            image_max = torch.max(image)
            peak = torch.abs(image_max - background)

            rescale = snr**2 / peak**2
            rescale = torch.clamp(
                rescale, min=1e-10, max=max_val / torch.abs(image_max)
            )

            noisy = torch.poisson(image * rescale) / rescale

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        # # --- rewrap if needed ---
        # if is_scattered:
        #     obj.array = noisy
        #     return obj

        return noisy

        # # For a numpy backend.
        # if self.get_backend() == "numpy":
        #     image[image < 0] = 0
        #     image_max = np.max(image)
        #     peak = np.abs(image_max - background)

        #     rescale = snr ** 2 / peak ** 2
        #     rescale = np.clip(
        #         rescale, 1e-10, max_val / np.abs(image_max)
        #     )
        #     try:
        #         noisy_image = np.random.poisson(image * rescale) / rescale
        #         return noisy_image
        #     except ValueError:
        #         raise ValueError(
        #             "NumPy poisson function errored due to too large value. "
        #             "Set max_val in dt.Poisson to a lower value to fix."
        #         )

        # # For a Torch backend.
        # elif self.get_backend() == "torch":
        #     image = torch.clamp(image, min=0)
        #     image_max = torch.max(image)
        #     peak = torch.abs(image_max - background)

        #     rescale = snr ** 2 / peak ** 2
        #     rescale = torch.clamp(
        #         rescale, min=1e-10, max=max_val / torch.abs(image_max)
        #     )
        #     try:
        #         noisy_image = torch.poisson(image * rescale) / rescale
        #         return noisy_image
        #     except ValueError:
        #         raise ValueError(
        #             "Torch Poisson function errored due to too large value. "
        #             "Set max_val in dt.Poisson to a lower value to fix."
        #         )