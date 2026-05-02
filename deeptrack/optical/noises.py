"""Noise models for images and array-like data.

This module provides features that add different types of noise to images,
arrays, and scattered objects. The implemented noise models include constant
offsets, Gaussian noise, complex Gaussian noise, and Poisson-distributed
noise. These features are typically used to simulate detector noise, background
signals, or stochastic measurement processes in synthetic microscopy
pipelines.
All noise models operate on both NumPy arrays and PyTorch tensors.
The active DeepTrack backend determines which implementation is used.

Module Structure
----------------
Classes:

- `Noise`: Base class for noise models.
- `Background` / `Offset`: Adds a constant value to the input image.
- `Gaussian`: Adds IID Gaussian noise.
- `ComplexGaussian`: Adds complex-valued Gaussian noise.
- `Poisson`: Adds Poisson-distributed noise based on signal-to-noise ratio.

Examples
--------
>>> import deeptrack as dt

Add Gaussian noise to an image.

>>> particle = dt.PointParticle(intensity=1)
>>> optics = dt.Fluorescence()
>>> gaussian_noise = dt.Gaussian(mu=0, sigma=0.1)
>>> noisy_image = optics(particle) >> gaussian_noise
>>> noisy_image.plot();

Add Poisson noise with a specified signal-to-noise ratio.

>>> poisson_noise = noises.Poisson(snr=0.1)
>>> noisy_image = optics(particle) >> poisson_noise
>>> noisy_image.plot();

"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from deeptrack.features import Feature
from deeptrack.types import PropertyLike
from deeptrack.backend import TORCH_AVAILABLE

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
    """Base class for noise models.

    Noise features add stochastic or deterministic perturbations to array-like
    data such as images. These features typically operate on NumPy arrays or
    PyTorch tensors and return a noisy version of the input.

    Subclasses implement the `.get(image, **kwargs)` method, which defines how
    the noise is generated and applied to a single input array.

    When a noise feature is evaluated, it applies the noise model to each
    element of the input list independently (since `__distributed__ = True`
    through inheritance from `Feature`).

    Noise features transparently support `ScatteredVolume` and
    `ScatteredField` objects. If the input element is one of these objects,
    the noise is applied to the underlying array (`element.array`) while
    preserving the container object and its metadata.

    This allows noise models to be inserted anywhere in a DeepTrack pipeline
    without breaking compatibility with scatterer-based simulations.

    Methods
    -------
    `get(image, **kwargs) -> np.ndarray | torch.Tensor`
        Abstract method implemented by subclasses to generate noise for a
        single input array.
    `_process_and_get(inputs, **properties) -> list`
        Internal method that applies noise to each element of the input list
        and preserves container objects when necessary.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Add Gaussian noise to an image

    >>> image = np.ones((64, 64))
    >>> noise = dt.Gaussian(mu=0, sigma=0.1)
    >>> noisy_image = noise(image)

    Apply noise inside a pipeline

    >>> particle = dt.PointParticle()
    >>> optics = dt.Fluorescence()
    >>> pipeline = optics(particle) >> dt.Gaussian(sigma=0.05)
    >>> image = pipeline()

    """

    def _process_and_get(
        self: Noise,
        inputs: list,
        **properties: Any,
    ) -> list:
        """Apply the noise model to a list of inputs.

        This method unwraps scattered objects to operate on their underlying
        arrays, applies the noise model using `get()`, and then restores the
        container object if needed.

        Parameters
        ----------
        inputs: list
            Input elements. Elements may be NumPy arrays, PyTorch tensors,
            or scattered objects such as `ScatteredVolume` or `ScatteredField`.
        **properties: Any
            Resolved properties passed to the noise model.

        Returns
        -------
        list
            List of noisy outputs with the same container types as the inputs.

        """

        results = []

        # Lazy import avoids circular dependency
        try:
            from deeptrack.optical.scatterers import ScatteredVolume, ScatteredField

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
    offset: PropertyLike[float]
        Constant value added to the image.
    **kwargs: Any
        Additional keyword arguments passed to the parent `Noise` class.

    Methods
    -------
    `get(image, offset, **kwargs) -> np.ndarray | torch.Tensor`
        Adds the constant offset to the input image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image with zeros

    >>> image = np.zeros((2, 2))

    Define a background offset:

    >>> noise = dt.Background(offset=0.5)

    Apply the noise

    >>> noisy = noise(image)
    >>> print(noisy)
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

    Gaussian noise is sampled from a normal distribution and added
    independently to each pixel of the input image.

    Parameters
    ----------
    mu: PropertyLike[float], optional
        Mean of the Gaussian distribution. Defaults to `0`.
    sigma: PropertyLike[float], optional
        Standard deviation of the Gaussian distribution. Defaults to `1`.

    Methods
    -------
    `get(image, mu, sigma, **kwargs) -> np.ndarray | torch.Tensor`
        Returns the input image with Gaussian noise added.

    Examples
    --------
    Add Gaussian noise to an image.

    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:

    >>> image = np.ones((2, 2)) * 3

    Define Gaussian noise:

    >>> noise = dt.Gaussian(mu=1, sigma=0.1)

    Apply the noise:

    >>> noisy = noise(image)
    >>> print(noisy)
    [[4.01965863 4.20688642]
     [4.02184982 3.87875873]]

    """

    def __init__(
        self: Gaussian,
        mu: PropertyLike[float] = 0,
        sigma: PropertyLike[float] = 1,
        **kwargs: Any,
    ):
        """Initialize the Gaussian noise feature.

        Parameters
        ----------
        mu: PropertyLike[float]
            The mean of the Gaussian distribution.
        sigma: PropertyLike[float]
            The standard deviation of the Gaussian distribution.
        **kwargs: Any
            Additional arguments passed to the parent `Noise` class.

        """

        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(
        self: Gaussian,
        image: np.ndarray | torch.Tensor,
        mu: float,
        sigma: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Add Gaussian noise to the input image.

        Parameters
        ----------
        image: np.ndarray | torch.Tensor
            The input image to which noise will be added.
        mu: float
            The mean of the Gaussian distribution.
        sigma: float
            The standard deviation of the Gaussian distribution.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray | torch.Tensor
            The input image with Gaussian noise added.

        """

        # For a Numpy backend.
        if self.get_backend() == "numpy":
            noise = np.random.randn(*image.shape)

        # For a Torch backend.
        elif self.get_backend() == "torch":
            noise = torch.randn(*image.shape, device=image.device)

        return mu + image + noise * sigma


class ComplexGaussian(Noise):
    """Add complex-valued IID Gaussian noise to an image.

    Complex Gaussian noise is generated by sampling two independent Gaussian
    distributions for the real and imaginary components and combining them into
    a complex-valued noise field that is added pixel-wise to the input image.

    Parameters
    ----------
    mu: PropertyLike[float], optional
        Mean of the Gaussian distribution. Deafults to `0`.
    sigma: PropertyLike[float], optional
        Standard deviation of the Gaussian distribution. Defaults to `1`.

    Methods
    -------
    `get(image, mu, sigma, **kwargs) -> np.ndarray | torch.Tensor`
        Returns the input image with complex Gaussian noise added.

    Examples
    --------
    Add complex Gaussian noise to an image.

    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:

    >>> image = np.ones((2, 2)) * 3

    Define complex Gaussian noise:

    >>> noise = dt.ComplexGaussian(mu=1, sigma=0.1)

    Apply the noise:

    >>> noisy = noise(image)
    >>> print(noisy)
    [[3.79975648-0.06967551j 4.09943404+0.06499738j]
     [3.99886747-0.23549974j 4.15725117-0.07847024j]]

    """

    def __init__(
        self: ComplexGaussian,
        mu: PropertyLike[float] = 0,
        sigma: PropertyLike[float] = 1,
        **kwargs: Any,
    ):
        """Initialize the complex Gaussian noise feature.

        Parameters
        ----------
        mu: PropertyLike[float]
            Mean of the Gaussian distribution.
        sigma: PropertyLike[float]
            Standard deviation of the Gaussian distribution.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Noise` class.

        """

        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def get(
        self: ComplexGaussian,
        image: np.ndarray | torch.Tensor,
        mu: float,
        sigma: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Add complex Gaussian noise to the input image.

        Parameters
        ----------
        image: np.ndarray | torch.Tensor
            Input image to which noise will be added.
        mu: float
            Mean of the Gaussian distribution.
        sigma: float
            Standard deviation of the Gaussian distribution.
        **kwargs: Any
            Additional keyword arguments passed through the feature pipeline.

        Returns
        -------
        np.ndarray | torch.Tensor
            The input image with complex Gaussian noise added.

        """

        # For a Numpy backend.
        if self.get_backend() == "numpy":
            real_noise = np.random.randn(*image.shape)
            imag_noise = np.random.randn(*image.shape)

        # For a Torch backend.
        elif self.get_backend() == "torch":
            real_noise = torch.randn(*image.shape, device=image.device)
            imag_noise = torch.randn(*image.shape, device=image.device)

        noise = real_noise + 1j * imag_noise
        return mu + image + noise * sigma


class Poisson(Noise):
    """Add Poisson-distributed noise to an image.

    Poisson noise is generated according to the pixel intensity of the input
    image and scaled to achieve a desired signal-to-noise ratio (`snr`).

    Parameters
    ----------
    snr: PropertyLike[float], optional
        Target signal-to-noise ratio of the output image. The signal is
        determined by the peak value of the input image. Defaults to `100`.
    background: PropertyLike[float], optional
        Background level used when computing the signal amplitude.
        Defaults to `0`.
    max_val: PropertyLike[float], optional
        Maximum allowable value used to prevent overflow during noise
        computation. Defaults to `1e8`.

    Methods
    -------
    `get(image, snr, background, max_val, **kwargs) -> array | tensor`
        Returns the input image with Poisson noise added.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:

    >>> image = np.ones((2, 2))

    Define Poisson noise:

    >>> noise = dt.Poisson(snr=1)

    Apply the noise:

    >>> noisy = noise(image)
    >>> print(noisy)
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
        """Initialize the Poisson noise feature.

        Parameters
        ----------
        snr: PropertyLike[float]
            Target signal-to-noise ratio of the output image. The signal is
            determined by the peak value of the input image.
        background: PropertyLike[float]
            Background level used when computing the signal amplitude.
        max_val: PropertyLike[float]
            Maximum allowable value used to prevent overflow during noise
            computation.
        **kwargs: Any
            Additional keyword arguments passed to the parent `Noise` class.

        """

        super().__init__(
            *args,
            snr=snr,
            background=background,
            max_val=max_val,
            **kwargs,
        )

    def get(
        self: Poisson,
        image: np.ndarray | torch.Tensor,
        snr: float,
        background: float,
        max_val: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Add Poisson noise to the input image.

        Parameters
        ----------
        image: np.ndarray | torch.Tensor
            Input image to which noise will be added.
        snr: float
            Target signal-to-noise ratio of the output image.
        background: float
            Background level used when computing the signal amplitude.
        max_val: float
            Maximum allowable value used to prevent overflow during noise
            computation.
        **kwargs: Any
            Additional keyword arguments passed through the feature pipeline.

        Returns
        -------
        np.ndarray | torch.Tensor
            The input image with Poisson noise added.

        """

        backend = self.get_backend()

        if backend == "numpy":
            image = np.clip(image, 0, None)
            image_max = np.max(image)
            peak = max(np.abs(image_max - background), 1e-12)

            rescale = (snr / peak) ** 2
            rescale = np.clip(
                rescale,
                1e-10,
                max_val / max(np.abs(image_max), 1e-12),
            )

            noisy = np.random.poisson(image * rescale) / rescale

        elif backend == "torch":
            image = torch.clamp(image, min=0)
            image_max = torch.max(image)
            peak = torch.abs(image_max - background)
            peak = torch.clamp(peak, min=1e-12)

            rescale = (snr / peak) ** 2
            rescale = torch.clamp(
                rescale,
                min=1e-10,
                max=max_val / torch.clamp(torch.abs(image_max), min=1e-12),
            )

            noisy = torch.poisson(image * rescale) / rescale

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        return noisy
