"""Mathematical operations and structures.

This module provides classes and utilities to perform common mathematical
operations on images, including clipping, normalization, blurring, pooling,
resizing, and morphology. All operations are implemented as subclasses of
`Feature`, enabling seamless integration with the feature-based design of the
library.

Key Features
------------
- **Clipping**

    Restrict image values to a specified range.

- **Normalization**

    Adjust image values to a common scale.

- **Blurring**

    Smooth images using various filters.

- **Pooling**

    Downsample images by applying a function to local regions.

- **Resizing**

    Change the dimensions of images.

- **Morphology**

    Binary dilation and erosion on masks.

Module Structure
-----------------

Helper functions:

- `_prepare_mask`: Normalize mask shape and channel handling for morphological
    operations.
- `isotropic_dilation`: Apply isotropic dilation to a binary mask.
- `isotropic_erosion`:Apply isotropic erosion to a binary mask.
- `move_channel_last`: Move the channel axis to the last position.
- `restore_channel_axis`:Restore the channel axis to its original position.
- `pad_image_to_fft`: Pad an image to optimal size for FFT-based operations.

Classes:

- `Average`: Compute the mean across a list of inputs.
- `Clip`: Clip values to a specified minimum and maximum.
- `NormalizeMinMax`: Perform min–max normalization.
- `NormalizeStandard`: Normalize to zero mean and unit variance.
- `NormalizeQuantile`: Normalize based on specified quantiles.
- `Blur`: Base class for blurring operations.
- `AverageBlur`: Apply mean filtering.
- `GaussianBlur`: Apply Gaussian filtering.
- `MedianBlur`: Apply median filtering.
- `Pool`: Base class for pooling operations.
- `AveragePooling`: Apply average pooling.
- `MaxPooling`: Apply max pooling.
- `MinPooling`: Apply min pooling.
- `SumPooling`: Apply sum pooling.
- `MedianPooling`: Apply median pooling.
- `Resize`: Resize images to a specified spatial size.
- `BlurCV2`: Apply OpenCV-based blurring (NumPy backend only).
- `BilateralBlur`: Apply bilateral filtering for edge-preserving smoothing.

Examples
--------
>>> import deeptrack as dt

Define a simple pipeline with mathematical operations.

Create features for clipping and normalization.

>>> clip = dt.Clip(min=0, max=200)
>>> normalize = dt.NormalizeMinMax()

Chain features together.

>>> pipeline = clip >> normalize

Process an input image.

>>> import numpy as np
>>>
>>> input_image = np.array([0, 100, 200, 400])
>>> output_image = pipeline(input_image)
>>> print(output_image)
[0., 0.5, 1., 1.]

"""

from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING

import array_api_compat as apc
import numpy as np
from scipy import ndimage
import skimage
import skimage.measure

from deeptrack import utils, OPENCV_AVAILABLE, TORCH_AVAILABLE
from deeptrack.backend import xp
from deeptrack.features import Feature
from deeptrack.types import PropertyLike

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

if OPENCV_AVAILABLE:
    import cv2

__all__ = [
    "Average",
    "Clip",
    "NormalizeMinMax",
    "NormalizeStandard",
    "NormalizeQuantile",
    "Blur",
    "AverageBlur",
    "GaussianBlur",
    "MedianBlur",
    "Pool",
    "AveragePooling",
    "MaxPooling",
    "MinPooling",
    "SumPooling",
    "MedianPooling",
    "Resize",
    "BlurCV2",
    "BilateralBlur",
    "isotropic_dilation",
    "isotropic_erosion",
    "pad_image_to_fft",
]

if TYPE_CHECKING:
    import torch
    from deeptrack.scatterers import ScatteredField, ScatteredVolume



class Average(Feature):
    """Average of input arrays.

    Computes the mean of a list of arrays along the specified axis or axes.
    By default, averaging is performed along axis 0 (the batch dimension).

    This operation is purely algebraic and does **not interpret dimensions**
    (e.g., spatial vs channel). All axes are treated uniformly and must be
    specified explicitly.

    If `features` is provided, each feature is resolved first and the results
    are averaged.

    Parameters
    ----------
    axis: int or tuple[int], optional
        Axis or axes along which to compute the average. Defaults to `0`.
    features: list[Feature] or None, optional
        List of features to resolve and average. Defaults to `None`.

    Attributes
    ----------
    __distributed__: bool = False
        Determines whether `.get(...)` is applied to each element
        independently (`True`) or to the list as a whole (`False`).

    Methods
    -------
    `get(images, axis, **kwargs) -> np.ndarray | torch.Tensor`
        Computes the average of the input images along the given axis.

    Examples
    --------
    >>> import deeptrack as dt

    Create two input images.

    >>> import numpy as np
    >>>
    >>> input_image0 = np.ones((10, 30, 20)) * 2
    >>> input_image1 = np.ones((10, 30, 20)) * 4

    Define a pipeline with the average feature along the dimension 0.

    >>> average = dt.Average(axis=0)
    >>> output_image = average([input_image0, input_image1])
    >>> output_image
    (10, 30, 20)

    Define a pipeline with the average feature along the dimension 1.

    >>> average = dt.Average(axis=1)
    >>> output_image = average([input_image0, input_image1])
    >>> output_image.shape
    (2, 30, 20)

    Define a pipeline averaging each image.

    >>> average = dt.Average(axis=(1, 2, 3))
    >>> output_image = average([input_image0, input_image1])
    >>> output_image.shape
    (2,)

    """

    __distributed__: bool = False
    features: list[Feature] | None

    def __init__(
        self: Average,
        axis: PropertyLike[int] = 0,
        features: list[Feature] | None = None,
        **kwargs: Any,
    ):
        """Initialize the parameters for averaging input features.

        Parameters
        ----------
        axis: int or tuple[int], optional
            Axis or axes along which to compute the average. Defaults to `0`.
        features: list[Feature] or None, optional
            List of features to be resolved and averaged. Defaults to `None`.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(axis=axis, **kwargs)

        if features is None:
            self.features = None
        else:
            self.features = [self.add_feature(f) for f in features]

    def get(
        self: Average,
        images: list[np.ndarray | torch.Tensor],
        axis: int | tuple[int],
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Compute the average of input images along the specified axis(es).

        This method computes the average of the input images along the
        specified axis(es).

        Parameters
        ----------
        images: array
            The input images to average.
        axis: int or tuple(int)
            Axis or axes along which to average.

        Returns
        -------
        array
            The average of the input images along the specified axis.

        """

        if self.features is not None:
            images = [feature.resolve() for feature in self.features]
        result = xp.mean(xp.stack(images), axis=axis)

        return result


class Clip(Feature):
    """Clip values of an array to a specified range.

    This feature applies elementwise clipping such that all values in the input
    are constrained to the interval [`min`, `max`].

    This operation is purely pointwise and does not interpret dimensions (e.g.,
    spatial or channel axes). The same transformation is applied independently
    to every element.

    Parameters
    ----------
    min: float, optional
        Lower bound. Values below this will be set to `min`. Defaults to
        `-inf`.
    max: float, optional
        Upper bound. Values above this will be set to `max`. Defaults to
        `+inf`.

    Returns
    -------
    np.ndarray or torch.Tensor
        Clipped array with the same shape and dtype as the input.

    Methods
    -------
    `get(image, min, max, **kwargs) -> np.ndarray | torch.Tensor`
        Clips the input image between `min` and `max`.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:

    >>> import numpy as np
    >>>
    >>> input_image = np.asarray([[10, 4], [4, -10]])

    Define a clipper feature:

    >>> clipper = dt.Clip(min=0, max=5)
    >>> output_image = clipper(input_image)
    >>> output_image
    array([[5, 4],
           [4, 0]])

    """

    def __init__(
        self: Clip,
        min: PropertyLike[float] = -xp.inf,
        max: PropertyLike[float] = +xp.inf,
        **kwargs: Any,
    ):
        """Initialize the clipping range.

        Parameters
        ----------
        min: float, optional
            Minimum allowed value. Defaults to `-xp.inf`.
        max: float, optional
            Maximum allowed value. Defaults to `+xp.inf`.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(min=min, max=max, **kwargs)

    def get(
        self: Clip,
        image: np.ndarray | torch.Tensor,
        min: float,
        max: float,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Clips the input image within the specified values.

        This method clips the input image within the specified minimum and
        maximum values.

        Parameters
        ----------
        image: array
            Input image to clip.
        min: float
            Minimum allowed value.
        max: float
            Maximum allowed value.

        Returns
        -------
        array
            The clipped image.

        """

        return xp.clip(image, min, max)


class NormalizeMinMax(Feature):
    """Min-max normalization of an array.

    Applies a linear transformation that maps input values to the range
    [`min`, `max`].

    If `featurewise=False`, normalization is applied globally over the entire
    input.

    If `featurewise=True`, normalization is applied independently along
    `channel_axis`, which is interpreted as the feature/channel dimension.

    Parameters
    ----------
    min: float, optional
        Lower bound of the output range. Default is 0.
    max: float, optional
        Upper bound of the output range. Default is 1.
    featurewise: bool, optional
        Whether to normalize each feature independently. Default is True.
    channel_axis: int or None, optional
        Axis corresponding to channels/features. If `None`, featurewise
        normalization is disabled even if `featurewise=True`. Default is -1.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized array with the same shape as input.

    Methods
    -------
    `get(image, min, max, **kwargs) -> np.ndarray | torch.Tensor`
        Normalizes the image to be within the specified range.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:

    >>> import numpy as np
    >>>
    >>> input_image = np.array([[10, 4], [4, -10]])

    Define a min-max normalizer:

    >>> normalizer = dt.NormalizeMinMax(min=-5, max=5)
    >>> output_image = normalizer(input_image)
    >>> output_image
    array([[ 5.,  2.],
           [ 2., -5.]])

    """

    def __init__(
        self: NormalizeMinMax,
        min: PropertyLike[float] = 0,
        max: PropertyLike[float] = 1,
        featurewise: bool = True,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ):
        """Initialize the min-max normalization parameters.

        Parameters
        ----------
        min: float
            Lower bound of the output range.
        max: float
            Upper bound of the output range.
        featurewise: bool
            Whether to normalize each feature independently.
        channel_axis: int or None
            Axis corresponding to channels/features.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(
            min=min,
            max=max,
            featurewise=featurewise,
            channel_axis=channel_axis,
            **kwargs,
        )

    def get(
        self: NormalizeMinMax,
        image: np.ndarray | torch.Tensor,
        min: float,
        max: float,
        featurewise: bool = True,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Normalize the input to fall between `min` and `max`.

        Parameters
        ----------
        image: np.ndarray or torch.Tensor
            Input image to normalize.
        min: float
            Lower bound of the output range.
        max: float
            Upper bound of the output range.
        featurewise: bool
            Whether to normalize each feature (channel) independently.
        channel_axis: int or None
            Axis corresponding to channels/features. If `None`, normalization
            is always global.

        Returns
        -------
        np.ndarray or torch.Tensor
            Min-max normalized image.

        """

        if featurewise and channel_axis is not None:
            ch_axis = channel_axis % image.ndim
            reduce_axes = tuple(
                ax for ax in range(image.ndim) if ax != ch_axis
            )

            img_min = xp.min(image, axis=reduce_axes, keepdims=True)
            img_max = xp.max(image, axis=reduce_axes, keepdims=True)
        else:
            img_min = xp.min(image)
            img_max = xp.max(image)

        ptp = img_max - img_min

        if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
            zero_ptp = ptp == 0
            safe_ptp = torch.where(zero_ptp, torch.ones_like(ptp), ptp)
        else:
            zero_ptp = ptp == 0
            safe_ptp = np.where(zero_ptp, np.ones_like(ptp), ptp)

        image = (image - img_min) / safe_ptp
        image = image * (max - min) + min

        # Preserve old behavior: constant images/features become 0.
        image = xp.where(zero_ptp, xp.zeros_like(image), image)
        image = xp.where(xp.isnan(image), xp.zeros_like(image), image)

        return image


class NormalizeStandard(Feature):
    """Standardize an array to zero mean and unit variance.

    Applies z-score normalization:

        output = (input - mean) / std

    where the standard deviation is computed as the **population standard
    deviation** (dividing by N, not N-1).

    Axis semantics:
    - If `featurewise=False`, normalization is applied globally.
    - If `featurewise=True` and `channel_axis` is specified, normalization is
      applied independently per channel.
    - If `featurewise=True` and `channel_axis=None`, normalization defaults to
      global behavior.

    The output always preserves the input shape.

    Parameters
    ----------
    featurewise: bool, optional
        Whether to normalize each feature independently. It default to `True`,
        which is the only behavior currently implemented.

    Returns
    -------
    np.ndarray or torch.Tensor
        Standardized array with the same shape as input.

    Methods
    -------
    `get(image: array, **kwargs: Any) -> array`
        Standardizes the input image to mean 0 and std deviation 1.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image.

    >>> import numpy as np
    >>>
    >>> input_image = np.array([[1, 2], [3, 4]], dtype=float)

    >>> standardizer = dt.NormalizeStandard()
    >>> output_image = standardizer(input_image)
    >>> output_image
    array([[-1.34164079, -0.4472136 ],
        [ 0.4472136 ,  1.34164079]])

    """

    def __init__(
        self: NormalizeStandard,
        featurewise: PropertyLike[bool] = True,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ):
        """Initialize the parameters for standardization.

        This constructor initializes the parameters for standardization.

        Parameters
        ----------
        featurewise: bool, optional
            Whether to normalize each feature independently.
        channel_axis: int or None
            Axis corresponding to channels/features.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(
            featurewise=featurewise, channel_axis=channel_axis, **kwargs
        )

    def get(
        self: NormalizeStandard,
        image: np.ndarray | torch.Tensor,
        featurewise: bool,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Standardize the input image to zero mean and unit variance.

        Applies z-score normalization:

            (image - mean) / std

        where `std` is the population standard deviation (i.e., computed with
        denominator N).

        Axis semantics:
        - If `featurewise=False`, normalization is applied globally over all
        elements.
        - If `featurewise=True` and `channel_axis` is specified, normalization
        is applied independently along each channel.
        - If `featurewise=True` and `channel_axis=None`, normalization falls
        back to global behavior.

        The output preserves the input shape.

        Parameters
        ----------
        image: np.ndarray or torch.Tensor
            Input array to standardize. Must match the selected backend.
        featurewise: bool
            Whether to normalize each channel independently.
        channel_axis: int or None, optional
            Axis corresponding to channels/features. If None, no channel-wise
            normalization is performed.
        **kwargs: Any
            Additional keyword arguments (unused).

        Returns
        -------
        np.ndarray or torch.Tensor
            Standardized array with the same shape and backend as the input.

        """

        backend = self.get_backend()

        if backend == "torch":
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            return self._get_torch(
                image,
                featurewise=featurewise,
                channel_axis=channel_axis,
                **kwargs,
            )

        elif backend == "numpy":
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "NumPy backend selected but image is not a np.ndarray"
                )

            return self._get_numpy(
                image,
                featurewise=featurewise,
                channel_axis=channel_axis,
                **kwargs,
            )

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    def _get_numpy(
        self,
        image: np.ndarray,
        featurewise: bool,
        channel_axis: int | None,
        **kwargs: Any,
    ) -> np.ndarray:
        """NumPy implementation of standardization.

        Performs z-score normalization using NumPy operations. Uses population
        standard deviation (`ddof=0`). Channels are temporarily moved to the
        last axis for computation. Numerical stability is ensured by clamping
        the standard deviation.

        Parameters
        ----------
        image: np.ndarray
            Input array.
        featurewise: bool
            Whether to normalize per channel.
        channel_axis: int or None
            Channel axis. If specified, normalization is applied independently
            across channels.

        Returns
        -------
        np.ndarray
            Standardized array with the same shape as input.

        """

        if featurewise and channel_axis is not None:
            image_moved = np.moveaxis(image, channel_axis, -1)

            axis = tuple(range(image_moved.ndim - 1))
            mean = np.mean(image_moved, axis=axis, keepdims=True)
            std = np.std(image_moved, axis=axis, keepdims=True)

            zero_std = std == 0
            safe_std = np.where(zero_std, np.ones_like(std), std)

            out = (image_moved - mean) / safe_std
            out = np.where(zero_std, np.zeros_like(out), out)

            out = np.moveaxis(out, -1, channel_axis)

        else:
            mean = np.mean(image)
            std = np.std(image)

            zero_std = std == 0
            safe_std = 1.0 if zero_std else std

            out = (image - mean) / safe_std
            out = np.zeros_like(out) if zero_std else out

        out = np.where(np.isnan(out), 0.0, out)
        return out

    def _get_torch(
        self,
        image: torch.Tensor,
        featurewise: bool,
        channel_axis: int | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """PyTorch implementation of standardization.

        Performs z-score normalization using PyTorch tensor operations. Uses
        population standard deviation (`unbiased=False`). Channels are
        temporarily moved to the last axis for computation. Numerical stability
        is ensured via `torch.clamp`.

        Parameters
        ----------
        image: torch.Tensor
            Input tensor.
        featurewise: bool
            Whether to normalize per channel.
        channel_axis: int or None
            Channel axis. If specified, normalization is applied independently
            across channels.

        Returns
        -------
        torch.Tensor
            Standardized tensor with the same shape as input.

        """

        if featurewise and channel_axis is not None:
            image_moved = image.movedim(channel_axis, -1)

            axis = tuple(range(image_moved.ndim - 1))
            mean = image_moved.mean(dim=axis, keepdim=True)
            std = image_moved.std(dim=axis, keepdim=True, unbiased=False)

            zero_std = std == 0
            safe_std = torch.where(zero_std, torch.ones_like(std), std)

            out = (image_moved - mean) / safe_std
            out = torch.where(zero_std, torch.zeros_like(out), out)

            out = out.movedim(-1, channel_axis)

        else:
            mean = image.mean()
            std = image.std(unbiased=False)

            zero_std = std == 0
            safe_std = torch.where(zero_std, torch.ones_like(std), std)

            out = (image - mean) / safe_std
            out = torch.where(zero_std, torch.zeros_like(out), out)

        out = torch.nan_to_num(out, nan=0.0)
        return out


class NormalizeQuantile(Feature):
    """Quantile-based normalization.

    Centers the input at the median and scales it using a quantile range:

        output = (image - median) / (q_high - q_low)

    Axis semantics:
    - If `featurewise=False`, quantiles are computed globally.
    - If `featurewise=True` and `channel_axis` is specified, quantiles are
      computed independently per channel.
    - If `featurewise=True` and `channel_axis=None`, normalization falls back
      to global behavior.

    The output preserves the input shape.

    Parameters
    ----------
    quantiles: tuple[float, float]
        Quantile range (q_min, q_max), with 0 < q_min < q_max < 1.
    featurewise: bool, optional
        Whether to normalize per channel. Default is True.
    channel_axis: int or None, optional
        Axis corresponding to channels. Default is -1.

    Notes
    -----
    - Not differentiable.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image.

    >>> import numpy as np
    >>>
    >>> input_image = np.array([[10, 4], [4, -10]])

    Define a quantile normalizer.

    >>> normalizer = dt.NormalizeQuantile(quantiles=(0.25, 0.75))
    >>> output_image = normalizer(input_image)
    >>> output_image
    array([[ 1.2,  0. ],
           [ 0. , -2.8]])

    """

    def __init__(
        self: NormalizeQuantile,
        quantiles: PropertyLike[tuple[float, float]] = (0.25, 0.75),
        featurewise: PropertyLike[bool] = True,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ):
        """Initialize the parameters for quantile normalization.

        This constructor initializes the parameters for quantile normalization.

        Parameters
        ----------
        quantiles: tuple[float, float], optional
            Quantile range to calculate scaling factor.
        featurewise: bool, optional
            Whether to normalize each feature independently.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(
            quantiles=quantiles,
            featurewise=featurewise,
            channel_axis=channel_axis,
            **kwargs,
        )

    def get(
        self,
        image: np.ndarray | torch.Tensor,
        quantiles: tuple[float, float],
        featurewise: bool,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ):
        backend = self.get_backend()

        if backend == "torch":
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            return self._get_torch(
                image,
                quantiles=quantiles,
                featurewise=featurewise,
                channel_axis=channel_axis,
                **kwargs,
            )

        elif backend == "numpy":
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "NumPy backend selected but image is not a np.ndarray"
                )

            return self._get_numpy(
                image,
                quantiles=quantiles,
                featurewise=featurewise,
                channel_axis=channel_axis,
                **kwargs,
            )

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    def _get_numpy(
        self: NormalizeQuantile,
        image: np.ndarray,
        quantiles: tuple[float, float],
        featurewise: bool,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ) -> np.ndarray:
        """Normalize the input image based on the specified quantiles.

        Parameters
        ----------
        image: np.ndarray or torch.Tensor
            The input image to normalize.
        quantiles: tuple[float, float]
            Quantile range to calculate scaling factor.
        featurewise: bool
            Whether to normalize each feature (channel) independently.
        channel_axis: int or None
            Axis corresponding to channels/features. If None, no channel-wise
            normalization is performed.
        kwargs: Any
            Additional keyword arguments (unused).

        Returns
        -------
        np.ndarray or torch.Tensor
            The quantile-normalized image.

        """

        q_low_val, q_high_val = quantiles

        if featurewise and channel_axis is not None:
            image_moved = np.moveaxis(image, channel_axis, -1)

            axis = tuple(range(image_moved.ndim - 1))
            q_low, q_high, median = np.quantile(
                image_moved,
                (q_low_val, q_high_val, 0.5),
                axis=axis,
                keepdims=True,
            )

            out = (image_moved - median) / np.maximum(
                q_high - q_low,
                np.asarray(1e-8, dtype=image.dtype),
            )

            out = np.moveaxis(out, -1, channel_axis)

        else:
            q_low, q_high, median = np.quantile(
                image,
                (q_low_val, q_high_val, 0.5),
            )

            out = (image - median) / np.maximum(
                q_high - q_low,
                np.asarray(1e-8, dtype=image.dtype),
            )

        out = np.where(np.isnan(out), 0.0, out)
        return out

    def _get_torch(
        self,
        image: torch.Tensor,
        quantiles: tuple[float, float],
        featurewise: bool,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Normalize the input image based on the specified quantiles.

        Parameters
        ----------
        image: torch.Tensor
            The input image to normalize.
        quantiles: tuple[float, float]
            Quantile range to calculate scaling factor.
        featurewise: bool
            Whether to normalize each feature (channel) independently.
        channel_axis: int or None
            Axis corresponding to channels/features. If None, no channel-wise
            normalization is performed.
        kwargs: Any
            Additional keyword arguments (unused).

        Returns
        -------
        torch.Tensor
            The quantile-normalized image.

        """

        q_low_val, q_high_val = quantiles

        q = torch.tensor(
            [q_low_val, q_high_val, 0.5],
            device=image.device,
            dtype=image.dtype,
        )

        if featurewise and channel_axis is not None:
            image_moved = image.movedim(channel_axis, -1)

            spatial_dims = tuple(range(image_moved.ndim - 1))

            # flatten spatial dims
            x = image_moved.reshape(-1, image_moved.shape[-1])

            q_vals = torch.quantile(x, q, dim=0)
            q_low, q_high, median = q_vals

            # reshape for broadcasting
            shape = [1] * image_moved.ndim
            shape[-1] = image_moved.shape[-1]

            q_low = q_low.view(shape)
            q_high = q_high.view(shape)
            median = median.view(shape)

            out = (image_moved - median) / torch.clamp(
                q_high - q_low,
                min=1e-8,
            )

            out = out.movedim(-1, channel_axis)

        else:
            q_low, q_high, median = torch.quantile(image, q)

            out = (image - median) / torch.clamp(
                q_high - q_low,
                min=1e-8,
            )

        out = torch.nan_to_num(out)
        return out


def move_channel_last(
    x: np.ndarray | torch.Tensor,
    channel_axis: int | None,
) -> tuple[np.ndarray | torch.Tensor, int | None]:
    """Move the channel axis to the last position.

    Helper function to move the channel axis to the last position for both
    NumPy and PyTorch tensors. If `channel_axis` is `None`, the input is
    returned unchanged.

    Parameters
    ----------
    x: np.ndarray or torch.Tensor
        Input array or tensor.
    channel_axis: int or None
        Axis corresponding to channels/features. If None, no movement is
        performed.

    Returns
    -------
    tuple[np.ndarray or torch.Tensor, int or None]
        A tuple containing the array/tensor with the channel axis moved to the
        last position and the original channel axis index (or None if no
        movement was done).

    """

    if channel_axis is None:
        return x, None

    original_axis = channel_axis

    if isinstance(x, np.ndarray):
        x = np.moveaxis(x, channel_axis, -1)
    elif isinstance(x, torch.Tensor):
        x = x.movedim(channel_axis, -1)
    else:
        raise TypeError("Unsupported type")

    return x, original_axis


def restore_channel_axis(
    x: np.ndarray | torch.Tensor,
    original_axis: int | None,
) -> np.ndarray | torch.Tensor:
    """Restore the channel axis to its original position.

    Helper function to restore the channel axis to its original position after
    processing. If `original_axis` is `None`, the input is returned unchanged.

    Parameters
    ----------
    x: np.ndarray or torch.Tensor
        Input array or tensor.
    original_axis: int or None
        Original axis index for the channel dimension. If None, no movement is
        performed.

    Returns
    -------
    np.ndarray or torch.Tensor
        The array/tensor with the channel axis restored to its original
        position.

    """

    if original_axis is None:
        return x

    if isinstance(x, np.ndarray):
        return np.moveaxis(x, -1, original_axis)
    elif isinstance(x, torch.Tensor):
        return x.movedim(-1, original_axis)
    else:
        raise TypeError("Unsupported type")


class Blur(Feature):
    """Backend-dispatched abstract base class for blurring operations.

    This class defines a unified interface for applying blur filters across
    multiple computational backends (NumPy and PyTorch). Subclasses are
    responsible for implementing the backend-specific logic via `_get_numpy`
    and `_get_torch`.

    Subclasses must implement at least one of:
        `_get_numpy(image, **kwargs)`
        `_get_torch(image, **kwargs)`

    Methods
    -------
    `get(image, **kwargs) -> np.ndarray | torch.Tensor`
        Applies the blur using the selected backend.

    """

    def get(
        self: Blur,
        image: np.ndarray | torch.Tensor | ScatteredVolume | ScatteredField,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Apply the blur filter to the input image using the selected backend.

        This method applies the blur filter to the input image using the
        selected backend. It dispatches to the appropriate backend-specific
        implementation based on the type of the input image and the configured
        backend. It also handles unwrapping of scattered objects if necessary.

        Parameters
        ----------
        image: np.ndarray or torch.Tensor or ScatteredVolume or ScatteredField
            The input image to blur. Must be compatible with the selected
            backend. If a scattered object is provided, the blur will be
            applied to its underlying array.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray or torch.Tensor or ScatteredVolume or ScatteredField
            The blurred image, with the same shape and backend as the input.

        """

        backend = self.get_backend()
        from deeptrack.scatterers import ScatteredVolume, ScatteredField

        is_scattered = isinstance(image, (ScatteredVolume, ScatteredField))
        if is_scattered:
            obj = image.copy()
            image = obj.array  # operate on underlying array

        if backend == "torch":
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            result = self._get_torch(
                image,
                **kwargs,
            )

        elif backend == "numpy":
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "NumPy backend selected but image is not a np.ndarray"
                )

            result = self._get_numpy(
                image,
                **kwargs,
            )

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        if is_scattered:
            obj.array = result
            return obj

        return result

    def _get_numpy(self, image: np.ndarray, **kwargs):
        raise NotImplementedError

    def _get_torch(self, image: torch.Tensor, **kwargs):
        raise NotImplementedError


class AverageBlur(Blur):
    """Blur an image by computing simple means over neighbourhoods.

    Applies a uniform (mean) filter over spatial dimensions.

    If `channel_axis` is specified, the blur is applied independently
    per channel. Otherwise, all dimensions (including channels, if present)
    are treated as spatial, and the filter is applied across them.

    Parameters
    ----------
    ksize: int
        Kernel size for the blur operation.
    channel_axis: int or None
        The axis representing the channel dimension. If `None`, channels are
        not treated separately and the same blurring is applied across all
        dimensions.

    Methods
    -------
    `get(image, ksize, channel_axis, **kwargs) --> np.ndarray | torch.Tensor`
        Applies the average blurring filter to the input image.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image.

    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(32, 32)

    Define an average blur feature.

    >>> average_blur = dt.AverageBlur(ksize=3, channel_axis=None)
    >>> output_image = average_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    """

    def __init__(
        self: AverageBlur,
        ksize: int = 3,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize the parameters for averaging input features.

        This constructor initializes the parameters for averaging input
        features.

        Parameters
        ----------
        ksize: int
            Kernel size for the blur operation.
        channel_axis: int | None
            The axis representing the channel dimension.
        **kwargs: Any
            Additional keyword arguments.

        """

        self.ksize = int(ksize)
        self.channel_axis = channel_axis
        super().__init__(**kwargs)

    def _get_numpy(
        self: AverageBlur, image: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Apply average blurring using SciPy's uniform_filter.

        This method applies average blurring to the input image using
        SciPy's `uniform_filter`.

        Parameters
        ----------
        image: np.ndarray
            The input image to blur.
        **kwargs: Any
            Additional keyword arguments for `uniform_filter`.

        Returns
        -------
        np.ndarray
            The blurred image.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)

        if ch_axis is not None:
            k = (self.ksize,) * (x.ndim - 1) + (1,)
        else:
            k = (self.ksize,) * x.ndim

        out = ndimage.uniform_filter(
            x,
            size=k,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
            origin=kwargs.get("origin", 0),
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self: AverageBlur, image: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Apply average blurring using PyTorch's avg_pool.

        This method applies average blurring to the input image using
        PyTorch's `avg_pool` functions.

        Parameters
        ----------
        image: torch.Tensor
            The input image to blur.
        **kwargs: Any
            Additional keyword arguments for padding.

        Returns
        -------
        torch.Tensor
            The blurred image.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)

        # move channels → first (torch conv convention)
        if ch_axis is not None:
            x = x.movedim(-1, 0)  # C, ...
        else:
            x = x.unsqueeze(0)  # 1, ...

        x = x.unsqueeze(0)  # 1, C, ...

        spatial_dims = x.ndim - 2
        k = (self.ksize,) * spatial_dims

        # padding
        pad = []
        for kk in reversed(k):
            p = kk // 2
            pad.extend([p, p])

        x = F.pad(
            x,
            tuple(pad),
            mode=kwargs.get("mode", "reflect"),
            value=kwargs.get("cval", 0),
        )

        # pooling
        if spatial_dims == 1:
            out = F.avg_pool1d(x, k, stride=1)
        elif spatial_dims == 2:
            out = F.avg_pool2d(x, k, stride=1)
        elif spatial_dims == 3:
            out = F.avg_pool3d(x, k, stride=1)
        else:
            raise NotImplementedError(f"{spatial_dims}D not supported")

        out = out.squeeze(0)

        if ch_axis is not None:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return restore_channel_axis(out, ch_axis)


class GaussianBlur(Blur):
    """Apply a Gaussian blur over spatial dimensions.

    The image is convolved with a Gaussian kernel with standard deviation
    `sigma`. If `channel_axis` is specified, the blur is applied independently
    per channel. Otherwise, all dimensions (including channels, if present)
    are treated as spatial, and the filter is applied across them.
    The implementation uses separable convolution for efficiency.
    For large `sigma` relative to the image size, the output approaches
      the global mean of the image.

    Parameters
    ----------
    sigma: float
        Standard deviation of the Gaussian kernel.
    channel_axis: int or None, default=-1
        Axis corresponding to channels. Set to None to treat all dimensions
        as spatial.

    Methods
    -------
    `get(image, sigma, channel_axis, **kwargs) --> array | tensor`
            Apply Gaussian blurring to the input image using the selected
            backend.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:

    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(32, 32)

    Define a Gaussian blur feature.

    >>> gaussian_blur = dt.GaussianBlur(sigma=2, channel_axis=None)
    >>> output_image = gaussian_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Visualize the input and output images.

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(input_image, cmap='gray')
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(output_image, cmap='gray')
    >>> plt.show()

    """

    def __init__(
        self: GaussianBlur,
        sigma: PropertyLike[float] = 2,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ):
        """Initialize the parameters for Gaussian blurring.

        Parameters
        ----------
        sigma: float
            Standard deviation of the Gaussian kernel.
        channel_axis: int or None
            Axis corresponding to channels/features. If `None`, all dimensions
            are treated as spatial.
        **kwargs: Any
            Additional keyword arguments.

        """

        self.channel_axis = channel_axis
        super().__init__(sigma=sigma, **kwargs)

    def _get_numpy(
        self: GaussianBlur, image: np.ndarray, sigma: float, **kwargs
    ) -> np.ndarray:
        """Apply Gaussian blurring using SciPy's gaussian_filter.

        Apply Gaussian blur using SciPy's `gaussian_filter`. The `sigma`
        parameter is expanded to match the number of dimensions, with zero for
        the channel dimension if `channel_axis` is specified. The blur is
        applied across all spatial dimensions, and independently per channel if
        `channel_axis` is set.

        Parameters
        ----------
        image: np.ndarray
            The input image to blur.
        sigma: float
            Standard deviation of the Gaussian kernel.
        **kwargs: Any
            Additional keyword arguments for `gaussian_filter`.

        Returns
        -------
        np.ndarray
            The blurred image.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)

        if ch_axis is not None:
            sigma_full = (sigma,) * (x.ndim - 1) + (0,)
        else:
            sigma_full = (sigma,) * x.ndim

        out = ndimage.gaussian_filter(
            x,
            sigma=sigma_full,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self: GaussianBlur,
        image: torch.Tensor,
        sigma: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply Gaussian blurring using separable convolution.

        Applies Gaussian blur using separable 1D convolutions. Channels are
        processed independently if `channel_axis` is set. Otherwise, the same
        blur is applied across all dimensions. The method handles edge cases
        such as zero sigma (no blur) and large sigma (approaching global mean).

        Parameters
        ----------
        image: torch.Tensor
            The input image to blur.
        sigma: float
            Standard deviation of the Gaussian kernel.
        **kwargs: Any
            Additional keyword arguments for padding and convolution.

        Returns
        -------
        torch.Tensor
            The blurred image.

        """

        if sigma == 0:
            return image.clone()

        x, ch_axis = move_channel_last(image, self.channel_axis)

        # move channels → first (C, ...)
        if ch_axis is not None:
            x = x.movedim(-1, 0)
        else:
            x = x.unsqueeze(0)

        x = x.unsqueeze(0)  # (1, C, ...)

        spatial_dims = x.ndim - 2
        spatial_sizes = x.shape[2:]

        radius = int(np.ceil(3 * sigma))
        if radius == 0:
            return image.clone()

        # --- SAFE GUARD: avoid invalid reflect padding ---
        if any(radius >= s for s in spatial_sizes):
            # Gaussian with very large sigma → constant mean
            dims = tuple(range(2, x.ndim))
            mean = x.mean(dim=dims, keepdim=True)
            x = mean.expand_as(x)

            out = x.squeeze(0)
            if ch_axis is not None:
                out = out.movedim(0, -1)
            else:
                out = out.squeeze(0)

            return restore_channel_axis(out, ch_axis)

        # --- build 1D Gaussian kernel ---
        coords = torch.arange(
            -radius,
            radius + 1,
            device=x.device,
            dtype=x.dtype,
        )
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        C = x.shape[1]

        mode = kwargs.get("mode", "reflect")
        cval = kwargs.get("cval", 0.0)

        if spatial_dims == 2:
            kx = kernel.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
            ky = kernel.view(1, 1, -1, 1).repeat(C, 1, 1, 1)

            x = F.pad(x, (radius, radius, 0, 0), mode=mode, value=cval)
            x = F.conv2d(x, kx, groups=C)

            x = F.pad(x, (0, 0, radius, radius), mode=mode, value=cval)
            x = F.conv2d(x, ky, groups=C)

        elif spatial_dims == 1:
            k = kernel.view(1, 1, -1).repeat(C, 1, 1)
            x = F.pad(x, (radius, radius), mode=mode, value=cval)
            x = F.conv1d(x, k, groups=C)

        elif spatial_dims == 3:
            raise NotImplementedError("3D GaussianBlur not implemented yet")

        else:
            raise NotImplementedError(f"{spatial_dims}D not supported")

        out = x.squeeze(0)

        if ch_axis is not None:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return restore_channel_axis(out, ch_axis)


class MedianBlur(Blur):
    """Apply a median filter over spatial dimensions.

    Each pixel is replaced by the median of its neighborhood defined by
    `ksize`. Median filtering is effective at removing impulsive noise
    (e.g., salt-and-pepper) while preserving edges.

    If `channel_axis` is specified, the filter is applied independently
    per channel. Otherwise, all dimensions (including channels, if present)
    are treated as spatial and the filter is applied across them.

    NumPy backend uses `scipy.ndimage.median_filter`. Torch backend uses
    explicit unfolding and is significantly slower. Median filtering is not
    differentiable.

    Parameters
    ----------
    ksize: int
        Size of the median filter window (must be odd).
    channel_axis: int or None, default=-1
        Axis corresponding to channels. Set to None to treat all dimensions
        as spatial.

    Methods
    -------
    `get(image, ksize, channel_axis, **kwargs) --> array | tensor`
            Applies the median filter to the input image using the selected
            backend.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:

    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(32, 32)

    Define a median blur feature:

    >>> median_blur = dt.MedianBlur(ksize=3, channel_axis=None)
    >>> output_image = median_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Visualize the input and output images:

    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(input_image, cmap='gray')
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(output_image, cmap='gray')
    >>> plt.show()

    """

    def __init__(
        self: MedianBlur,
        ksize: PropertyLike[int] = 3,
        channel_axis: int | None = -1,
        **kwargs: Any,
    ):
        if isinstance(ksize, int) and ksize % 2 == 0:
            raise ValueError("MedianBlur requires an odd kernel size.")
        self.channel_axis = channel_axis
        super().__init__(ksize=ksize, **kwargs)

    def _get_numpy(
        self,
        image: np.ndarray,
        ksize: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply median filtering using SciPy's `median_filter`.

        The filter is applied over spatial dimensions, and independently
        per channel if `channel_axis` is specified.

        Parameters
        ----------
        image: np.ndarray
            The input image to blur.
        ksize: int
            Size of the median filter window (must be odd).
        **kwargs: Any
            Additional keyword arguments for `median_filter`.

        Returns
        -------
        np.ndarray
            The blurred image.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)

        if ch_axis is not None:
            size = (ksize,) * (x.ndim - 1) + (1,)
        else:
            size = (ksize,) * x.ndim

        out = ndimage.median_filter(
            x,
            size=size,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self,
        image: torch.Tensor,
        ksize: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply median filtering using explicit unfolding.

        Local neighborhoods are extracted using `unfold`, and the median is
        computed over each neighborhood. Channels are processed independently
        if `channel_axis` is specified.

        This implementation is significantly slower than the NumPy backend.

        Parameters
        ----------
        image: torch.Tensor
             The input image to blur.
        ksize: int
            Size of the median filter window (must be odd).
        **kwargs: Any
            Additional keyword arguments for padding.

        Returns
        -------
        torch.Tensor
            The blurred image.

        """

        if ksize % 2 == 0:
            raise ValueError("MedianBlur requires odd kernel size.")

        if ksize == 1:
            return image.clone()

        x, ch_axis = move_channel_last(image, self.channel_axis)

        # channels → first
        if ch_axis is not None:
            x = x.movedim(-1, 0)
        else:
            x = x.unsqueeze(0)

        x = x.unsqueeze(0)  # (1, C, ...)

        spatial_dims = x.ndim - 2
        pad = ksize // 2

        pad_tuple = []
        for _ in range(spatial_dims):
            pad_tuple = [pad, pad] + pad_tuple
        pad_tuple = tuple(pad_tuple)

        mode = kwargs.get("mode", "reflect")
        cval = kwargs.get("cval", 0)

        if mode == "constant":
            x = F.pad(x, pad_tuple, mode="constant", value=cval)
        else:
            x = F.pad(x, pad_tuple, mode=mode)

        # unfold
        if spatial_dims == 2:
            x = x.unfold(2, ksize, 1).unfold(3, ksize, 1)
        elif spatial_dims == 3:
            x = x.unfold(2, ksize, 1).unfold(3, ksize, 1).unfold(4, ksize, 1)
        else:
            raise NotImplementedError

        # compute median
        x = x.contiguous().view(*x.shape[:-spatial_dims], -1)
        x = x.median(dim=-1).values

        x = x.squeeze(0)

        if ch_axis is not None:
            x = x.movedim(0, -1)
        else:
            x = x.squeeze(0)

        return restore_channel_axis(x, ch_axis)


class Pool(Feature):
    """Abstract base class for pooling operations.

    Pooling reduces the spatial resolution of an array by aggregating values
    over local neighborhoods defined by `ksize`.

    The pooling window is specified as:
    - int → same size in all spatial dimensions
    - (px, py) → 2D pooling
    - (px, py, pz) → 3D pooling

    If `channel_axis` is specified, pooling is applied independently per
    channel. Otherwise, all dimensions (including channels, if present) are
    treated as spatial.

    Input dimensions are cropped (from the origin) to be divisible by the
    pooling size before applying pooling. Cropping is not centered: excess
    elements are removed from the right/bottom (or back).

    Subclasses must  implement `_get_numpy` and/or `_get_torch`, respect
    `channel_axis` and call `_crop_to_multiple`.

    Parameters
    ----------
    ksize: int or tuple
        Size of the pooling window.
    channel_axis: int or None, default=None
        Axis corresponding to channels. Set to None to treat all dimensions
        as spatial.

    Methods
    -------
    `get(image, ksize, channel_axis, **kwargs) --> array | tensor`
        Apply the pooling operation to the input image using the selected
        backend.

    """

    def __init__(
        self: Pool,
        ksize: PropertyLike[int | tuple[int, int] | tuple[int, int, int]] = 2,
        channel_axis: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the parameters for pooling operations.

        Parameters
        ----------
        ksize: int or tuple
            Size of the pooling window. Can be an int (same size for all
            spatial dimensions) or a tuple specifying the size for each spatial
            dimension.
        channel_axis: int or None
            Axis corresponding to channels. If None, all dimensions are treated
            as spatial.
        **kwargs: Any
            Additional keyword arguments.

        """

        self.ksize = self._normalize_ksize(ksize)
        self.channel_axis = channel_axis
        super().__init__(**kwargs)

    @staticmethod
    def _normalize_ksize(
        ksize: int | tuple[int, int] | tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Normalize the ksize parameter to a 3D tuple.

        This method takes the `ksize` parameter, which can be specified as an
        int (for uniform pooling) or a tuple (for dimension-specific pooling),
        and normalizes it to a 3D tuple of the form (px, py, pz). For 2D
        pooling, the tuple is expanded to (px, py, 1).

        Parameters
        ----------
        ksize: int or tuple
            The kernel size for pooling. Can be an int (same size for all
            spatial dimensions) or a tuple specifying the size for each spatial
            dimension.

        Returns
        -------
        tuple[int, int, int]
            A normalized 3D tuple representing the pooling window size in the
            format (px, py, pz). For 2D pooling, the tuple is expanded to
            (px, py, 1).

        """
        if isinstance(ksize, int):
            return (ksize, ksize, ksize)

        if isinstance(ksize, tuple):
            if len(ksize) == 2:
                return (ksize[0], ksize[1], 1)
            if len(ksize) == 3:
                return tuple(int(k) for k in ksize)

        raise TypeError("ksize must be int, (px, py), or (px, py, pz)")

    def get(
        self: Pool,
        image: np.ndarray | torch.Tensor | ScatteredVolume | ScatteredField,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Apply the pooling operation to the input image.

        This method applies the pooling operation to the input image using the
        selected backend. It dispatches to the appropriate backend-specific
        implementation based on the type of the input image and the configured
        backend. It also handles unwrapping of scattered objects if necessary.

        Parameters
        ----------
        image: np.ndarray or torch.Tensor or ScatteredVolume or ScatteredField
            The input image to pool. Must be compatible with the selected
            backend. If a scattered object is provided, the pooling will be
            applied to its underlying array.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray or torch.Tensor or ScatteredVolume or ScatteredField
            The pooled image, with reduced spatial resolution.

        """

        backend = self.get_backend()
        from deeptrack.scatterers import ScatteredVolume, ScatteredField

        is_scattered = isinstance(image, (ScatteredVolume, ScatteredField))
        if is_scattered:
            obj = image.copy()
            image = obj.array

        if backend == "torch":
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            result = self._get_torch(
                image,
                **kwargs,
            )

        elif backend == "numpy":
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "NumPy backend selected but image is not a np.ndarray"
                )

            result = self._get_numpy(
                image,
                **kwargs,
            )

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        if is_scattered:
            obj.array = result
            return obj

        return result

    def _get_pool_size(
        self: Pool,
        x: np.ndarray | torch.Tensor,
        has_channels: bool = False,
    ) -> tuple[int, int] | tuple[int, int, int]:
        """Return pooling window size matching input dimensionality.

        Parameters
        ----------
        x: np.ndarray or torch.Tensor
            Input array or tensor for which to determine the pooling window
            size.
        has_channels: bool
            Whether the input has a channel dimension.

        Returns
        -------
        tuple[int, int] or tuple[int, int, int]
            The pooling window size corresponding to the spatial dimensions of
            the input. Returns (px, py) for 2D inputs and (px, py, pz) for 3D
            inputs.

        """

        px, py, pz = self.ksize

        spatial_dims = x.ndim - (1 if has_channels else 0)

        if spatial_dims == 2:
            return (px, py)

        if spatial_dims == 3:
            return (px, py, pz)

        raise NotImplementedError("Only 2D or 3D inputs supported")

    def _crop_to_multiple(
        self: Pool,
        array: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Crop the input array.

        Crop the input array from the origin (top-left/front) to ensure that
        each spatial dimension is divisible by the pooling size. This is not a
        centered crop. The cropping is performed from the origin.

        Parameters
        ----------
        array: np.ndarray or torch.Tensor
            The input array to crop. Must be compatible with the selected
            backend.

        Returns
        -------
        np.ndarray or torch.Tensor
            The cropped array, with spatial dimensions adjusted to be divisible
            by the pooling size.

        """

        # assumes array is already channel-last if channels exist
        has_channels = self.channel_axis is not None
        pool = self._get_pool_size(array, has_channels)

        # 2D
        if len(pool) == 2:
            px, py = pool
            H, W = array.shape[:2]
            crop_h = (H // px) * px
            crop_w = (W // py) * py
            return array[:crop_h, :crop_w, ...]

        # 3D
        elif len(pool) == 3:
            px, py, pz = pool
            H, W, Z = array.shape[:3]
            crop_h = (H // px) * px
            crop_w = (W // py) * py
            crop_z = (Z // pz) * pz
            return array[:crop_h, :crop_w, :crop_z, ...]

        else:
            raise NotImplementedError("Unsupported dimensionality")

    def _get_numpy(self, image: np.ndarray, **kwargs):
        raise NotImplementedError

    def _get_torch(self, image: torch.Tensor, **kwargs):
        raise NotImplementedError


class AveragePooling(Pool):
    """Average pooling over spatial dimensions.

    Reduces spatial resolution by computing the mean over non-overlapping
    blocks of size `ksize`.

    The interpretation of dimensions depends on `channel_axis`:
    - If `channel_axis` is specified, pooling is applied only over spatial
      dimensions and independently per channel.
    - If `channel_axis=None`, all dimensions are treated as spatial and are
      pooled jointly.

    Input arrays are cropped from the origin so that each spatial dimension
    is divisible by the pooling size. Cropping is not centered.

    This implementation is consistent across NumPy and PyTorch backends.

    Parameters
    ----------
    ksize: int or tuple
        Pooling window size. Can be:
        - int → same size for all spatial dimensions
        - (px, py) → 2D pooling
        - (pz, px, py) → 3D pooling
    channel_axis: int or None, default=None
        Axis corresponding to channels. If None, all dimensions are treated
        as spatial.

    Notes
    -----
    - Channels are never pooled when `channel_axis` is specified.
    - The operation is equivalent to strided average pooling with stride equal
      to kernel size.
    - Behavior matches `skimage.measure.block_reduce` (NumPy) and
      `torch.nn.functional.avg_pool*` (PyTorch).

    Examples
    --------
    >>> import deeptrack as dt

    >>> import numpy as np
    >>>
    >>> image = np.ones((4, 4, 3))
    >>> pool = dt.AveragePooling(ksize=2, channel_axis=-1)
    >>> out = pool(image)
    >>> out.shape
    (2, 2, 3)

    >>> pool = dt.AveragePooling(ksize=2, channel_axis=None)
    >>> out = pool(image)
    >>> out.shape
    (2, 2, 1)

    """

    def _get_numpy(
        self: AveragePooling,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply average pooling using block reduction.

        This implementation uses `skimage.measure.block_reduce` to compute
        local means over non-overlapping blocks. Channel dimensions, if
        present, are excluded from pooling by using a block size of 1 along
        that axis.

        Parameters
        ----------
        image: np.ndarray
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        np.ndarray
            Downsampled array with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        if has_channels:
            block_size = pool + (1,)
        else:
            block_size = pool

        out = skimage.measure.block_reduce(
            x,
            block_size=block_size,
            func=np.mean,
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self: AveragePooling,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply average pooling using PyTorch pooling operators.

        The input is reshaped to match PyTorch's expected layout:
        (N, C, spatial...). Pooling is performed using `avg_pool2d` or
        `avg_pool3d` depending on dimensionality, with kernel size equal to
        stride to ensure non-overlapping pooling.

        Parameters
        ----------
        image: torch.Tensor
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        # ---- reshape to torch format ----
        if has_channels:
            x = x.movedim(-1, 0)  # C, H, W
        else:
            x = x.unsqueeze(0)  # 1, H, W

        x = x.unsqueeze(0)  # 1, C, H, W

        # ---- pooling ----
        if len(pool) == 2:
            out = F.avg_pool2d(x, pool, pool)
        elif len(pool) == 3:
            out = F.avg_pool3d(x, pool, pool)
        else:
            raise NotImplementedError

        # ---- restore ----
        out = out.squeeze(0)

        if has_channels:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return restore_channel_axis(out, ch_axis)


class MaxPooling(Pool):
    """Max pooling over spatial dimensions.

    Reduces spatial resolution by taking the maximum over non-overlapping
    blocks of size `ksize`.

    The interpretation of dimensions depends on `channel_axis`:

    - If `channel_axis` is specified, pooling is applied independently per
      channel and never across channels.
    - If `channel_axis=None`, all dimensions are treated as spatial.

    Input arrays are cropped from the origin so that each spatial dimension
    is divisible by the pooling size.

    Works with both NumPy and PyTorch backends.

    Parameters
    ----------
    ksize: int or tuple
        Pooling window size.
    channel_axis: int or None, default=None
        Axis corresponding to channels.

    Notes
    -----
    - Equivalent to standard max pooling with stride equal to kernel size.
    - Preserves extrema and is non-linear (unlike average pooling).

    Examples
    --------
    >>> import deeptrack as dt

    >>> import numpy as np
    >>>
    >>> image = np.random.rand(4, 4, 3)
    >>> pool = dt.MaxPooling(ksize=2, channel_axis=-1)
    >>> out = pool(image)
    >>> out.shape
    (2, 2, 3)

    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply max pooling using block reduction.

        This implementation uses `skimage.measure.block_reduce` to compute
        local maxima over non-overlapping blocks. Channel dimensions, if present,
        are excluded from pooling by using a block size of 1 along that axis.

        Parameters
        ----------
        image: np.ndarray
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        np.ndarray
            Downsampled array with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        if has_channels:
            block_size = pool + (1,)
        else:
            block_size = pool

        out = skimage.measure.block_reduce(
            x,
            block_size=block_size,
            func=np.max,
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply max pooling using PyTorch pooling operators.

        The input is reshaped to match PyTorch's expected layout:
        (N, C, spatial...). Pooling is performed using `max_pool2d` or
        `max_pool3d` depending on dimensionality, with kernel size equal to
        stride to ensure non-overlapping pooling.

        Parameters
        ----------
        image: torch.Tensor
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with reduced spatial dimensions.

        """
        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        # ---- reshape to torch format ----
        if has_channels:
            x = x.movedim(-1, 0)  # C, H, W
        else:
            x = x.unsqueeze(0)  # 1, H, W

        x = x.unsqueeze(0)  # 1, C, H, W

        # ---- pooling ----
        if len(pool) == 2:
            out = F.max_pool2d(x, pool, pool)
        elif len(pool) == 3:
            out = F.max_pool3d(x, pool, pool)
        else:
            raise NotImplementedError

        # ---- restore ----
        out = out.squeeze(0)

        if has_channels:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return restore_channel_axis(out, ch_axis)


class MinPooling(Pool):
    """Min pooling over spatial dimensions.

    Reduces spatial resolution by taking the minimum over non-overlapping
    blocks of size `ksize`.

    The interpretation of dimensions depends on `channel_axis`:

    - If `channel_axis` is specified, pooling is applied independently per
      channel and never across channels.
    - If `channel_axis=None`, all dimensions are treated as spatial.

    Input arrays are cropped from the origin so that each spatial dimension
    is divisible by the pooling size.

    Works with both NumPy and PyTorch backends.

    Parameters
    ----------
    ksize: int or tuple
        Pooling window size.
    channel_axis: int or None, default=None
        Axis corresponding to channels.

    Notes
    -----
    - Equivalent to standard min pooling with stride equal to kernel size.
    - Preserves extrema and is non-linear (unlike average pooling).

    Examples
    --------
    >>> import deeptrack as dt

    >>> import numpy as np
    >>>
    >>> image = np.random.rand(4, 4, 3)
    >>> pool = dt.MinPooling(ksize=2, channel_axis=-1)
    >>> out = pool(image)
    >>> out.shape
    (2, 2, 3)

    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply min pooling using block reduction.

        This implementation uses `skimage.measure.block_reduce` to compute
        local minima over non-overlapping blocks. Channel dimensions, if present,
        are excluded from pooling by using a block size of 1 along that axis.

        Parameters
        ----------
        image: np.ndarray
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        np.ndarray
            Downsampled array with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        if has_channels:
            block_size = pool + (1,)
        else:
            block_size = pool

        out = skimage.measure.block_reduce(
            x,
            block_size=block_size,
            func=np.min,
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply min pooling using PyTorch pooling operators.

        The input is reshaped to match PyTorch's expected layout:
        (N, C, spatial...). Pooling is performed using `-max_pool2d(-x, ...)`
        or `-max_pool3d(-x, ...)` depending on dimensionality, with kernel size
        equal to stride to ensure non-overlapping pooling.

        Parameters
        ----------
        image: torch.Tensor
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with reduced spatial dimensions.

        """
        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        # ---- reshape to torch format ----
        if has_channels:
            x = x.movedim(-1, 0)  # C, H, W
        else:
            x = x.unsqueeze(0)  # 1, H, W

        x = x.unsqueeze(0)  # 1, C, H, W

        # ---- pooling ----
        if len(pool) == 2:
            out = -F.max_pool2d(-x, pool, pool)
        elif len(pool) == 3:
            out = -F.max_pool3d(-x, pool, pool)
        else:
            raise NotImplementedError

        # ---- restore ----
        out = out.squeeze(0)

        if has_channels:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return restore_channel_axis(out, ch_axis)


class SumPooling(Pool):
    """Sum pooling over spatial dimensions.

    Reduces spatial resolution by taking the sum over non-overlapping
    blocks of size `ksize`.

    The interpretation of dimensions depends on `channel_axis`:

    - If `channel_axis` is specified, pooling is applied independently per
      channel and never across channels.
    - If `channel_axis=None`, all dimensions are treated as spatial.

    Input arrays are cropped from the origin so that each spatial dimension
    is divisible by the pooling size.

    Works with both NumPy and PyTorch backends.

    Parameters
    ----------
    ksize: int or tuple
        Pooling window size.
    channel_axis: int or None, default=None
        Axis corresponding to channels.

    Notes
    -----
    - Equivalent to standard sum pooling with stride equal to kernel size.
    - Linear operation (unlike max/min pooling).

    Examples
    --------
    >>> import deeptrack as dt

    >>> import numpy as np
    >>>
    >>> image = np.random.rand(4, 4, 3)
    >>> pool = dt.SumPooling(ksize=2, channel_axis=-1)
    >>> out = pool(image)
    >>> out.shape
    (2, 2, 3)

    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply sum pooling using block reduction.

        This implementation uses `skimage.measure.block_reduce` to compute
        local sums over non-overlapping blocks. Channel dimensions, if present,
        are excluded from pooling by using a block size of 1 along that axis.

        Parameters
        ----------
        image: np.ndarray
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        np.ndarray
            Downsampled array with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        if has_channels:
            block_size = pool + (1,)
        else:
            block_size = pool

        out = skimage.measure.block_reduce(
            x,
            block_size=block_size,
            func=np.sum,
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply sum pooling using PyTorch pooling operators.

        The input is reshaped to match PyTorch's expected layout:
        (N, C, spatial...). Pooling is performed using `avg_pool2d` or
        `avg_pool3d` depending on dimensionality multiplied by the kernel size,
        with kernel size equal to stride to ensure non-overlapping pooling.

        Parameters
        ----------
        image: torch.Tensor
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        # ---- reshape to torch format ----
        if has_channels:
            x = x.movedim(-1, 0)  # C, H, W
        else:
            x = x.unsqueeze(0)  # 1, H, W

        x = x.unsqueeze(0)  # 1, C, H, W

        # ---- pooling ----
        if len(pool) == 2:
            out = F.avg_pool2d(x, pool, pool)
        elif len(pool) == 3:
            out = F.avg_pool3d(x, pool, pool)
        else:
            raise NotImplementedError

        kernel_volume = 1
        for p in pool:
            kernel_volume *= p
        out = out * kernel_volume

        # ---- restore ----
        out = out.squeeze(0)

        if has_channels:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return restore_channel_axis(out, ch_axis)


class MedianPooling(Pool):
    """Median pooling over spatial dimensions.

    Reduces spatial resolution by taking the median over non-overlapping
    blocks of size `ksize`.

    The interpretation of dimensions depends on `channel_axis`:

    - If `channel_axis` is specified, pooling is applied independently per
      channel and never across channels.
    - If `channel_axis=None`, all dimensions are treated as spatial.

    Input arrays are cropped from the origin so that each spatial dimension
    is divisible by the pooling size.

    Works with both NumPy and PyTorch backends.

    Parameters
    ----------
    ksize: int or tuple
        Pooling window size.
    channel_axis: int or None, default=None
        Axis corresponding to channels.

    Notes
    -----
    - Equivalent to standard median pooling with stride equal to kernel size.
    - Preserves central tendency and is non-linear (unlike average pooling).

    Examples
    --------
    >>> import deeptrack as dt

    >>> import numpy as np
    >>>
    >>> image = np.random.rand(4, 4, 3)
    >>> pool = dt.MedianPooling(ksize=2, channel_axis=-1)
    >>> out = pool(image)
    >>> out.shape
    (2, 2, 3)

    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply median pooling using block reduction.

        This implementation uses `skimage.measure.block_reduce` to compute
        local medians over non-overlapping blocks. Channel dimensions, if
        present, are excluded from pooling by using a block size of 1 along
        that axis.

        Parameters
        ----------
        image: np.ndarray
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        np.ndarray
            Downsampled array with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        if has_channels:
            block_size = pool + (1,)
        else:
            block_size = pool

        out = skimage.measure.block_reduce(
            x,
            block_size=block_size,
            func=np.median,
        )

        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply median pooling using PyTorch pooling operators.

        The input is reshaped to match PyTorch's expected layout:
        (N, C, spatial...). Pooling is performed by unfolding the input into
        non-overlapping blocks and computing the median along the last
        dimension. PyTorch does not have a built-in median pooling operator.

        Parameters
        ----------
        image: torch.Tensor
            The input image to pool.
        **kwargs: Any
            Additional keyword arguments for pooling.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with reduced spatial dimensions.

        """

        x, ch_axis = move_channel_last(image, self.channel_axis)
        has_channels = ch_axis is not None

        x = self._crop_to_multiple(x)
        pool = self._get_pool_size(x, has_channels)

        # ---------- helper ----------
        def _median_lastdim(x):
            vals, _ = torch.sort(x, dim=-1)
            n = vals.shape[-1]
            mid = n // 2
            if n % 2 == 1:
                return vals[..., mid]
            else:
                return (vals[..., mid - 1] + vals[..., mid]) / 2

        # ---------- reshape to (C, spatial...) ----------
        if has_channels:
            x = x.movedim(-1, 0)  # (C, ...)
        else:
            x = x.unsqueeze(0)  # (1, ...)

        spatial_dims = x.ndim - 1  # exclude channel dim

        # ---------- 2D ----------
        if spatial_dims == 2:
            px, py = pool

            x = x.unfold(1, px, px).unfold(2, py, py)
            x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], -1)

            out = _median_lastdim(x)

        # ---------- 3D ----------
        elif spatial_dims == 3:
            px, py, pz = pool

            x = x.unfold(1, px, px).unfold(2, py, py).unfold(3, pz, pz)
            x = x.contiguous().view(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1
            )

            out = _median_lastdim(x)

        else:
            raise NotImplementedError(f"{spatial_dims}D not supported")

        # ---------- restore ----------
        if has_channels:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return restore_channel_axis(out, ch_axis)


class Resize(Feature):
    """Resize an image to a specified spatial size.

    Resizes the spatial dimensions of an input array or tensor to a target
    size specified by `dsize`. The size is given as (width, height), while
    the output follows standard array layout (height, width).

    The operation supports both NumPy arrays and PyTorch tensors:

    - NumPy backend: uses `cv2.resize`
    - PyTorch backend: uses `torch.nn.functional.interpolate`

    Channel handling follows the `channel_axis` convention:

    - If `channel_axis` is specified, resizing is applied only to spatial
    dimensions and independently for each channel.
    - If `channel_axis=None` and the input has more than two dimensions,
    the last axis is treated as the channel dimension.

    Parameters
    ----------
    dsize: tuple[int, int]
        Target output size given as (width, height). This convention is
        backend-independent and applies equally to NumPy and PyTorch inputs.
    channel_axis: int or None, default=None
        Axis corresponding to channels in the input image. If None and
        dimension > 2, the last channel dimension is used.
    **kwargs: Any
        Additional keyword arguments.

    Methods
    -------
    `get(image, dsize, **kwargs) -> array | tensor`
        Resize the input image to the specified size using the selected
        backend.

    Examples
    --------
    >>> import numpy as np

    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(16, 16)
    >>> feature = dt.math.Resize(dsize=(8, 4))
    >>> resized_image = feature.resolve(input_image)
    >>> resized_image.shape
    (4, 8)

    >>> import numpy as np
    >>> input_image = np.random.rand(16, 16, 16)
    >>> feature = dt.math.Resize(dsize=(8, 4), channel_axis=1)
    >>> resized_image = feature.resolve(input_image)
    >>> resized_image.shape
    (4, 16, 8)

    """

    def __init__(
        self: Resize,
        dsize: PropertyLike[tuple[int, int]] = (256, 256),
        channel_axis: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the parameters for the Resize feature.

        Parameters
        ----------
        dsize: PropertyLike[tuple[int, int]]
            The target size. dsize is always (width, height) for both backends.
            Default is (256, 256).
        channel_axis: int | None, default=None
            The axis corresponding to the channels in the input image. If None
            and the input has more than two dimensions, the last channel
            dimension is used.
        **kwargs: Any
            Additional keywords arguments.

        """

        self.channel_axis = channel_axis
        super().__init__(dsize=dsize, **kwargs)

    def get(
        self: Resize,
        image: np.ndarray | torch.Tensor | ScatteredVolume | ScatteredField,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Resize the input image to a specified spatial size.

        This method dispatches to the appropriate backend implementation
        (NumPy or PyTorch) and applies resizing to the spatial dimensions
        of the input.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor or ScatteredVolume or ScatteredField
            The input image to resize. If a scattered object is provided, the
            resizing is applied to its internal array/tensor.
        dsize : tuple[int, int]
            Target output size given as (width, height). This convention is
            backend-independent and applies to both NumPy and PyTorch inputs.
        **kwargs : Any
            Additional keyword arguments passed to the underlying resize
            implementation:
            - NumPy backend: forwarded to `cv2.resize`
            - PyTorch backend: forwarded to
            `torch.nn.functional.interpolate` (if supported)

        Returns
        -------
        np.ndarray or torch.Tensor or ScatteredVolume or ScatteredField
            The resized image, with the same type and layout as the input.

        """

        backend = self.get_backend()

        from deeptrack.scatterers import ScatteredVolume, ScatteredField

        is_scattered = isinstance(image, (ScatteredVolume, ScatteredField))
        if is_scattered:
            obj = image.copy()
            image = obj.array

        if backend == "torch":
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            result = self._get_torch(
                image,
                dsize=dsize,
                **kwargs,
            )

        elif backend == "numpy":
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "NumPy backend selected but image is not a np.ndarray"
                )

            result = self._get_numpy(
                image,
                dsize=dsize,
                **kwargs,
            )

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        if is_scattered:
            obj.array = result
            return obj

        return result

    def _get_numpy(
        self,
        image: np.ndarray,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> np.ndarray:
        """Resize the input image using OpenCV.

        Parameters
        ----------
        image: np.ndarray
            The input image to resize.
        dsize: tuple[int, int]
            Target output size given as (width, height).
        **kwargs: Any
            Additional keyword arguments for `cv2.resize`.

        Returns
        -------
        np.ndarray
            The resized image.

        """

        target_w, target_h = map(int, dsize)

        # --- normalize channel handling ---
        x, ch_axis = move_channel_last(image, self.channel_axis)

        out = utils.safe_call(
            cv2.resize,
            positional_args=[x, (target_w, target_h)],
            **kwargs,
        )
        return restore_channel_axis(out, ch_axis)

    def _get_torch(
        self,
        image: torch.Tensor,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Resize the input image using PyTorch's interpolation functions.

        Parameters
        ----------
        image: torch.Tensor
            The input image to resize.
        dsize: tuple[int, int]
            Target output size given as (width, height).
        **kwargs: Any
            Additional keyword arguments for `torch.nn.functional.interpolate`.

        Returns
        -------
        torch.Tensor
            The resized image.

        """

        target_w, target_h = map(int, dsize)

        # --- normalize channel handling ---
        x, ch_axis = move_channel_last(image, self.channel_axis)

        # --- dtype safety ---
        orig_dtype = x.dtype
        if not torch.is_floating_point(x):
            x = x.float()

        # --- 2D (H, W) ---
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            out = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

            out = out.squeeze(0).squeeze(0)

            if out.dtype != orig_dtype:
                out = out.to(orig_dtype)

            return restore_channel_axis(out, ch_axis)

        # --- 3D ---
        if x.ndim == 3:

            # (H, W, C) → (1, C, H, W)
            x = x.movedim(-1, 0).unsqueeze(0)

            out = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

            out = out.squeeze(0).movedim(0, -1)

            if out.dtype != orig_dtype:
                out = out.to(orig_dtype)

            return restore_channel_axis(out, ch_axis)

        raise ValueError(f"Unsupported tensor shape {image.shape}")


class BlurCV2(Feature):
    """Apply a blurring filter using OpenCV (`cv2`).

    Applies an OpenCV-based blurring or filtering operation to an input image.
    The provided `filter_function` must be compatible with OpenCV and accept
    the input image via the `src` argument (e.g., `cv2.GaussianBlur`,
    `cv2.bilateralFilter`).

    Parameters
    ----------
    filter_function : Callable or str
        OpenCV-compatible filtering function. If a string is provided,
        it is resolved as an attribute of `cv2`.
    mode : str, default="reflect"
        Border handling mode. Supported values are:
        {'reflect', 'wrap', 'constant', 'mirror', 'nearest'}.
        These are internally mapped to OpenCV border types.
    **kwargs : Any
        Additional keyword arguments passed directly to the filtering function.

    Methods
    -------
    `get(image: np.ndarray, **kwargs: Any) --> array`
        Applies the blurring filter to the input image.

    Notes
    -----
    BlurCV2 is NumPy-only and does not support PyTorch tensors.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:

    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(32, 32)

    Define a blur feature using the Gaussian blur function:

    >>> import cv2
    >>>
    >>> blur = dt.BlurCV2(
    ...     filter_function=cv2.GaussianBlur,
    ...     ksize=(5, 5),
    ...     sigmaX=1,
    ...     mode='reflect',
    ... )
    >>> output_image = blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    """

    _MODE_TO_BORDER = {
        "reflect": "BORDER_REFLECT",
        "wrap": "BORDER_WRAP",
        "constant": "BORDER_CONSTANT",
        "mirror": "BORDER_REFLECT_101",
        "nearest": "BORDER_REPLICATE",
    }

    def __init__(
        self: BlurCV2,
        filter_function: Callable | str,
        mode: PropertyLike[str] = "reflect",
        **kwargs: Any,
    ):
        """Initialize the OpenCV-based blur feature.

        Parameters
        ----------
        filter_function : Callable or str
            OpenCV-compatible filtering function.
        mode : str, default="reflect"
            Border handling mode.
        **kwargs : Any
            Additional keyword arguments passed to the filtering function.

        """

        if not OPENCV_AVAILABLE:
            raise ImportError(
                "OpenCV not installed on device. Since OpenCV is an optional "
                f"dependency of DeepTrack2. To use {self.__class__.__name__}, "
                "you need to install it manually."
            )

        self.filter = filter_function
        self.mode = mode
        super().__init__(mode=mode, **kwargs)

    def get(
        self: BlurCV2,
        image: np.ndarray,
        mode: str,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies the blurring filter to the input image.

        This method applies the blurring filter to the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to blur. Must be a NumPy array.
        **kwargs: Any
            Additional parameters for the blurring function.

        Returns
        -------
        np.ndarray
            The blurred image.

        """

        kwargs.pop("name", None)
        kwargs.pop("borderType", None)

        if apc.is_torch_array(image):
            raise TypeError(
                "BlurCV2 only supports NumPy arrays. "
                "Use GaussianBlur / AverageBlur for Torch."
            )

        import cv2

        filter_fn = (
            getattr(cv2, self.filter)
            if isinstance(self.filter, str)
            else self.filter
        )

        try:
            border_attr = self._MODE_TO_BORDER[mode]
        except KeyError as e:
            raise ValueError(f"Unsupported border mode '{mode}'") from e

        try:
            border = getattr(cv2, border_attr)
        except AttributeError as e:
            raise RuntimeError(
                f"OpenCV missing border constant '{border_attr}'"
            ) from e

        return filter_fn(
            src=image,
            borderType=border,
            **kwargs,
        )


class BilateralBlur(BlurCV2):
    """Apply bilateral filtering using OpenCV (`cv2.bilateralFilter`).

    Bilateral filtering smooths homogeneous regions while preserving edges
    by combining spatial and intensity-based weighting.

    Parameters
    ----------
    d: int
        Diameter of the pixel neighborhood. If set to a non-positive value,
        it is computed automatically from `sigma_space`.
    sigma_color: float
        Standard deviation in the intensity (color) space. Larger values
        result in stronger mixing of pixels with different intensities.
    sigma_space: float
        Standard deviation in the spatial domain. Larger values allow
        influence from more distant pixels.
    **kwargs: Any
        Additional keyword arguments passed to `cv2.bilateralFilter`.

    Notes
    -----
    - This feature supports only NumPy arrays.
    - PyTorch tensors are not supported.
    - Parameter names are mapped to OpenCV conventions:
      `sigma_color → sigmaColor`, `sigma_space → sigmaSpace`.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:

    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(32, 32)

    Define a bilateral blur feature:

    >>> import cv2
    >>>
    >>> bilateral_blur = dt.BilateralBlur(
    ...     d=5,
    ...     sigma_color=50,
    ...     sigma_space=50,
    ...     mode='reflect',
    ... )
    >>> output_image = bilateral_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    """

    def __init__(
        self: BilateralBlur,
        d: PropertyLike[int] = 3,
        sigma_color: PropertyLike[float] = 50,
        sigma_space: PropertyLike[float] = 50,
        **kwargs: Any,
    ):
        """Initialize the bilateral blur feature.

        Parameters
        ----------
        d: int
            Diameter of the pixel neighborhood.
        sigma_color: float
            Standard deviation in the intensity domain.
        sigma_space: float
            Standard deviation in the spatial domain.
        **kwargs: Any
            Additional keyword arguments passed to `cv2.bilateralFilter`.

        """

        super().__init__(
            filter_function="bilateralFilter",
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            **kwargs,
        )


def _prepare_mask(
    mask: np.ndarray | torch.Tensor,
    channel_axis: int | None,
) -> tuple[np.ndarray | torch.Tensor, bool, bool]:
    """Standardize mask shape and channel handling for morphology.

    This function normalizes the input mask representation and determines
    whether the operation should be applied channel-wise.

    Behavior:
    - If `channel_axis` is specified, the mask is treated as multi-channel
      and processed independently along that axis.
    - If `channel_axis is None`, the mask is treated as a scalar field:
        - (H, W) → 2D image
        - (H, W, Z) → 3D volume
    - A singleton channel `(H, W, 1)` is treated as 2D during computation
      and restored after processing.

    Parameters
    ----------
    mask: np.ndarray or torch.Tensor
        Input mask with shape (H, W) or (H, W, D).
    channel_axis: int or None
        Axis corresponding to channels. If None, no channel interpretation
        is applied.

    Returns
    -------
    mask: np.ndarray or torch.Tensor
        Possibly reshaped mask used for computation.
    channelwise: bool
        Whether to apply the operation independently along a channel axis.
    restore_channel: bool
        Whether to restore a singleton channel dimension in the output.

    Raises
    ------
    ValueError
        If the input is not 2D or 3D.

    """

    if mask.ndim < 2 or mask.ndim > 3:
        raise ValueError(f"Mask must be 2D or 3D. Got shape {mask.shape}")

    # --- explicit channel handling ---
    if channel_axis is not None:
        return mask, True, False

    # --- implicit singleton channel ---
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask[..., 0], False, True

    return mask, False, False


def isotropic_dilation(
    mask: np.ndarray | torch.Tensor,
    radius: float,
    *,
    backend: str = "numpy",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    channel_axis: int | None = None,
) -> np.ndarray | torch.Tensor:
    """Apply binary dilation to a mask.

    Performs morphological dilation using a structuring element of radius
    `radius`. Output is always boolean. Shape is preserved.

    - If `channel_axis is None`:
        - (H, W) → treated as a 2D mask
        - (H, W, Z) → treated as a 3D volume
    - If `channel_axis` is specified:
        - Operation is applied independently for each channel
    - Singleton channel:
        - (H, W, 1) is treated as 2D and restored after processing

    **NumPy backend**
    Uses `skimage.morphology.isotropic_dilation`, based on Euclidean distance.
    An additional safeguard ensures that empty masks remain empty. This avoids
    boundary artifacts present in `skimage.morphology.isotropic_dilation`.

    **Torch backend**
    Uses convolution with a full kernel (square/cubic neighborhood),
    corresponding to Chebyshev distance. This is not strictly isotropic.

    Parameters
    ----------
    mask : np.ndarray or torch.Tensor
        Input mask. Non-zero values are treated as foreground (`mask > 0`).
    radius : float
        Radius of the structuring element. If `radius <= 0`, the input
        is returned unchanged.
    backend : {"numpy", "torch"}, default="numpy"
        Backend used for computation.
    device : torch.device, optional
        Device used for torch backend.
    dtype : torch.dtype, optional
        Data type for torch computations.
    channel_axis : int or None, optional
        Axis corresponding to channels.

    Returns
    -------
    np.ndarray or torch.Tensor
        Dilated mask (boolean) with the same shape as the input.

    """

    if radius <= 0:
        return mask

    mask, channelwise, restore_channel = _prepare_mask(mask, channel_axis)

    if channelwise:
        xp = np if backend == "numpy" else __import__("torch")

        # move channel axis to last
        mask_moved = (
            np.moveaxis(mask, channel_axis, -1)
            if backend == "numpy"
            else mask.movedim(channel_axis, -1)
        )

        outputs = [
            isotropic_dilation(
                mask_moved[..., c],
                radius,
                backend=backend,
                device=device,
                dtype=dtype,
                channel_axis=None,  # IMPORTANT: recursion must disable channel axis
            )
            for c in range(mask_moved.shape[-1])
        ]

        out = xp.stack(outputs, axis=-1)

        # move axis back
        if backend == "numpy":
            out = np.moveaxis(out, -1, channel_axis)
        else:
            out = out.movedim(-1, channel_axis)

        return out

    if backend == "numpy":
        from skimage.morphology import isotropic_dilation as sk_iso_dil

        mask = mask > 0
        if not np.any(mask):  # fixes a corner case
            return np.zeros_like(mask, dtype=bool)

        out = sk_iso_dil(mask, radius)
        if restore_channel:
            return out[..., None]
        return out

    r = int(np.ceil(radius))

    if mask.ndim == 2:
        kernel = torch.ones(
            (1, 1, 2 * r + 1, 2 * r + 1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv2d(x, kernel, padding=r)

    elif mask.ndim == 3:
        kernel = torch.ones(
            (1, 1, 2 * r + 1, 2 * r + 1, 2 * r + 1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv3d(x, kernel, padding=r)

    else:
        raise ValueError("Mask must be 2D or 3D")

    out = y[0, 0] > 0
    if restore_channel:
        return out[..., None]
    return out


def isotropic_erosion(
    mask: np.ndarray | torch.Tensor,
    radius: float,
    *,
    backend: str = "numpy",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    channel_axis: int | None = None,
) -> np.ndarray | torch.Tensor:
    """Apply binary erosion to a mask.

    Performs morphological erosion using a structuring element of radius
    `radius`. Output is always boolean. Shape is preserved.

    - If `channel_axis is None`:
        - (H, W) → treated as a 2D mask
        - (H, W, Z) → treated as a 3D volume
    - If `channel_axis` is specified:
        - Operation is applied independently for each channel
    - Singleton channel:
        - (H, W, 1) is treated as 2D and restored after processing

    **NumPy backend**
    Uses `skimage.morphology.isotropic_erosion`, based on Euclidean distance.

    **Torch backend**
    Uses convolution with a full kernel (square/cubic neighborhood),
    corresponding to Chebyshev distance. This is not strictly isotropic.

    Parameters
    ----------
    mask : np.ndarray or torch.Tensor
        Input mask. Non-zero values are treated as foreground (`mask > 0`).
    radius : float
        Radius of the structuring element. If `radius <= 0`, the input
        is returned unchanged.
    backend : {"numpy", "torch"}, default="numpy"
        Backend used for computation.
    device : torch.device, optional
        Device used for torch backend.
    dtype : torch.dtype, optional
        Data type for torch computations.
    channel_axis : int or None, optional
        Axis corresponding to channels.

    Returns
    -------
    np.ndarray or torch.Tensor
        Eroded mask (boolean) with the same shape as the input.

    """

    if radius <= 0:
        return mask

    mask, channelwise, restore_channel = _prepare_mask(mask, channel_axis)

    if channelwise:
        xp = np if backend == "numpy" else __import__("torch")

        # move channel axis to last
        mask_moved = (
            np.moveaxis(mask, channel_axis, -1)
            if backend == "numpy"
            else mask.movedim(channel_axis, -1)
        )

        outputs = [
            isotropic_erosion(
                mask_moved[..., c],
                radius,
                backend=backend,
                device=device,
                dtype=dtype,
                channel_axis=None,  # IMPORTANT: recursion must disable channel axis
            )
            for c in range(mask_moved.shape[-1])
        ]

        out = xp.stack(outputs, axis=-1)

        # move axis back
        if backend == "numpy":
            out = np.moveaxis(out, -1, channel_axis)
        else:
            out = out.movedim(-1, channel_axis)

        return out

    if backend == "numpy":
        from skimage.morphology import isotropic_erosion as sk_iso_ero

        mask = mask > 0
        out = sk_iso_ero(mask, radius)
        if restore_channel:
            return out[..., None]
        return out

    r = int(np.ceil(radius))

    if mask.ndim == 2:
        kernel = torch.ones(
            (1, 1, 2 * r + 1, 2 * r + 1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv2d(x, kernel, padding=r)

    elif mask.ndim == 3:
        kernel = torch.ones(
            (1, 1, 2 * r + 1, 2 * r + 1, 2 * r + 1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv3d(x, kernel, padding=r)

    else:
        raise ValueError("Mask must be 2D or 3D")

    required = kernel.numel()
    out = y[0, 0] >= required
    if restore_channel:
        return out[..., None]

    return out


_FASTEST_SIZES = []
for n in range(1, 10):
    for a in range(1, n):  # Start at 1 -> at least one factor of 2
        _FASTEST_SIZES.append(2**a * 3 ** (n - a - 1))
_FASTEST_SIZES = np.unique(_FASTEST_SIZES)


def pad_image_to_fft(
    image: np.ndarray | torch.Tensor,
    axes: Iterable[int] = (0, 1),
) -> np.ndarray | torch.Tensor:
    """Pad an image to improve Fast Fourier Transform (FFT) performance.
    Padding is applied at the end of each axis (no centering).

    Preserves backend:
    - NumPy input → NumPy output
    - Torch input → Torch output (preserves autograd compatibility)

    This function pads an image by adding zeros to the end of specified axes
    so that their lengths match the nearest larger size in `_FASTEST_SIZES`.
    Sizes are chosen as products of small prime factors, which are efficient
    for FFT algorithms.

    Parameters
    ----------
    image: np.ndarray | torch.Tensor
        The input image to pad.
    axes : iterable of int, optional
        Axes along which to apply padding. Negative axes are supported.

    Returns
    -------
    np.ndarray | torch.Tensor
        The padded image with dimensions optimized for FFT performance.

    Raises
    ------
    ValueError
        If no suitable size is found in `_FASTEST_SIZES` for any axis length.

    Examples
    --------
    >>> import numpy as np
    >>> from deeptrack.image import pad_image_to_fft

    Pad a NumPy array:

    >>> img = np.zeros((5, 11))
    >>> padded_img = pad_image_to_fft(img)
    >>> print(padded_img.shape)
    (6, 12)

    """

    def _closest(dim: int) -> int:
        for size in _FASTEST_SIZES:
            if size >= dim:
                return size
        raise ValueError(
            f"No suitable size found in _FASTEST_SIZES={_FASTEST_SIZES} "
            f"for dimension {dim}."
        )

    shape = list(image.shape)
    new_shape = list(shape)

    for axis in axes:
        new_shape[axis] = _closest(shape[axis])

    pad_sizes = [(0, new - old) for old, new in zip(shape, new_shape)]

    # --- NumPy backend ---
    if isinstance(image, np.ndarray):
        return np.pad(image, pad_sizes, mode="constant")

    # --- Torch backend ---
    if isinstance(image, torch.Tensor):
        # torch.nn.functional.pad expects reversed flat list
        pad = []
        for before, after in reversed(pad_sizes):
            pad.extend([before, after])

        return torch.nn.functional.pad(image, pad, mode="constant", value=0.0)

    raise TypeError(f"Unsupported type: {type(image)}")
