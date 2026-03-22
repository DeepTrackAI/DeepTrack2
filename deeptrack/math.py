"""Mathematical operations and structures.

This module provides classes and utilities to perform common mathematical
operations and transformations on images, including clipping, normalization,
blurring, and pooling. These are implemented as subclasses of `Feature` for
seamless integration with the feature-based design of the library. Each
`Feature` supports lazy evaluation and can be composed using operators (e.g.,
`>>` for chaining), enabling efficient and readable construction of image
processing pipelines.

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

Module Structure
-----------------
Classes:

- `Clip`: Clip the input values within a specified minimum and maximum range.

- `NormalizeMinMax`: Perform min-max normalization on images.

- `NormalizeStandard`: Normalize images to have mean 0 and standard
    deviation 1.

- `NormalizeQuantile`: Normalize images based on specified quantiles.

- `Blur`: Apply a blurring filter to the image.

- `AverageBlur`: Apply average blurring to the image.

- `GaussianBlur`: Apply Gaussian blurring to the image.

- `MedianBlur`: Apply median blurring to the image.

- `Pool`: Apply a pooling function to downsample the image.

- `AveragePooling`: Apply average pooling to the image.

- `MaxPooling`: Apply max-pooling to the image.

- `MinPooling`: Apply min-pooling to the image.

- `SumPooling`: Apply sum pooling to the image.

- `MedianPooling`: Apply median pooling to the image.

- `Resize`: Resize the image to a specified size.

- `BlurCV2`: Apply a blurring filter using OpenCV2.

- `BilateralBlur`: Apply bilateral blurring to preserve edges while smoothing.

Examples
--------
Define a simple pipeline with mathematical operations:
>>> import deeptrack as dt
>>> import numpy as np

Create features for clipping and normalization:
>>> clip = dt.Clip(min=0, max=200)
>>> normalize = dt.NormalizeMinMax()

Chain features together:
>>> pipeline = clip >> normalize

Process an input image:
>>> input_image = np.array([0, 100, 200, 400])
>>> output_image = pipeline(input_image)
>>> print(output_image)
[0., 0.5, 1., 1.]

"""

#TODO ***??*** revise class docstring
#TODO ***??*** revise DTAT381

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Tuple, TYPE_CHECKING
import warnings

import array_api_compat as apc
import numpy as np
from scipy import ndimage
import skimage
import skimage.measure

from deeptrack import image, utils, OPENCV_AVAILABLE, TORCH_AVAILABLE
from deeptrack.features import Feature
from deeptrack.types import PropertyLike
from deeptrack.backend import xp, config

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
]

if TYPE_CHECKING:
    import torch


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
        Axis or axes along which to compute the average. It defaults to 0.
    features: list[Feature] or None, optional
        List of features to resolve and average. It defaults to None.

    Attributes
    ----------
    __distributed__ : bool = False
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
        axis: int or tuple[int]
            Axis or axes along which to compute the average. It defaults to 0.
        features: list[Feature] or None, optional
            List of features to be resolved and averaged. It defaults to None.
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

    This operation is purely pointwise and does **not interpret dimensions**
    (e.g., spatial or channel axes). The same transformation is applied
    independently to every element.

    Parameters
    ----------
    min: float, optional
        Lower bound. Values below this will be set to `min`. It defaults to
        `-inf`.
    max: float, optional
        Upper bound. Values above this will be set to `max`. It defaults to
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
            Minimum allowed value. It defaults to `-xp.inf`.
        max: float, optional
            Maximum allowed value. It defaults to `+xp.inf`.
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
    min : float, optional
        Lower bound of the output range. Default is 0.
    max : float, optional
        Upper bound of the output range. Default is 1.
    featurewise : bool, optional
        Whether to normalize each feature independently. Default is True.
    channel_axis : int or None, optional
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
            **kwargs
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
            reduce_axes = tuple(ax for ax in range(image.ndim) if ax != ch_axis)

            img_min = xp.min(image, axis=reduce_axes, keepdims=True)
            img_max = xp.max(image, axis=reduce_axes, keepdims=True)
        else:
            img_min = xp.min(image)
            img_max = xp.max(image)

        ptp = img_max - img_min
        eps = xp.asarray(1e-8, dtype=image.dtype)
        ptp = xp.maximum(ptp, eps)

        out = (image - img_min) / ptp
        out = out * (max - min) + min
        out = xp.where(xp.isnan(out), xp.zeros_like(out), out)

        return out


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

        super().__init__(featurewise=featurewise, **kwargs)

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

            std = np.maximum(std, np.asarray(1e-8, dtype=image.dtype))
            out = (image_moved - mean) / std

            out = np.moveaxis(out, -1, channel_axis)

        else:
            mean = np.mean(image)
            std = np.std(image)

            std = np.maximum(std, np.asarray(1e-8, dtype=image.dtype))
            out = (image - mean) / std

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
        image : torch.Tensor
            Input tensor.
        featurewise : bool
            Whether to normalize per channel.
        channel_axis : int or None
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

            std = torch.clamp(std, min=1e-8)
            out = (image_moved - mean) / std

            out = out.movedim(-1, channel_axis)

        else:
            mean = image.mean()
            std = image.std(unbiased=False)

            std = torch.clamp(std, min=1e-8)
            out = (image - mean) / std

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
    quantiles : tuple[float, float]
        Quantile range (q_min, q_max), with 0 < q_min < q_max < 1.
    featurewise : bool, optional
        Whether to normalize per channel. Default is True.
    channel_axis : int or None, optional
        Axis corresponding to channels. Default is -1.

    Notes
    -----
    - Not differentiable.
    
    Examples
    --------
    >>> import deeptrack as dt

    Create an input image.
    >>> import numpy as np
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
            # ---- HARD GUARD: torch only ----
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
            # ---- HARD GUARD: numpy only ----
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
) -> Tuple[np.ndarray | torch.Tensor, int | None]:
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
    Tuple[np.ndarray or torch.Tensor, int or None]
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
        The array/tensor with the channel axis restored to its original position.

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
        image: np.ndarray | torch.Tensor,        
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Apply the blur filter to the input image using the selected backend.
        
        This method applies the blur filter to the input image using the
        selected backend. It dispatches to the appropriate backend-specific
        implementation based on the type of the input image and the configured
        backend. It also handles unwrapping of scattered objects if necessary.

        Parameters
        ----------
        image: np.ndarray or torch.Tensor
            The input image to blur. Must be compatible with the selected 
            backend.

        Returns
        -------
        np.ndarray or torch.Tensor
            The blurred image, with the same shape and backend as the input.
                
        """

        backend = self.get_backend()
        from deeptrack.scatterers import ScatteredVolume, ScatteredField

        is_scattered = isinstance(image, (ScatteredVolume, ScatteredField))
        if is_scattered:
            obj = image.copy()
            image = obj.array  # operate on underlying array

        if backend == "torch":
            # ---- HARD GUARD: torch only ----
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            result = self._get_torch(
                image,
                **kwargs,
            )

        elif backend == "numpy":
            # ---- HARD GUARD: numpy only ----
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

    def _get_numpy(self, image: np.ndarray,  **kwargs):
        raise NotImplementedError

    def _get_torch(self, image: torch.Tensor,  **kwargs):
        raise NotImplementedError


class AverageBlur(Blur):
    """Blur an image by computing simple means over neighbourhoods.

    Applies a uniform (mean) filter over spatial dimensions.

    If `channel_axis` is specified, the blur is applied independently
    per channel. Otherwise, all dimensions are treated as spatial.

    Parameters
    ----------
    ksize: int
        Kernel size for the pooling operation.
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
    >>> import numpy as np

    Create an input image.
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
        **kwargs: Any
    ) -> None:
        """Initialize the parameters for averaging input features.

        This constructor initializes the parameters for averaging input
        features.

        Parameters
        ----------
        ksize: int
            Kernel size for the pooling operation.
        channel_axis: int | None
            The axis representing the channel dimension.
        **kwargs: Any
            Additional keyword arguments.

        """

        self.ksize = int(ksize)
        self.channel_axis = channel_axis
        super().__init__(**kwargs)

    def _get_numpy(
        self: AverageBlur, 
        image: np.ndarray, 
        **kwargs: Any
    ) -> np.ndarray:
        """Apply average blurring using SciPy's uniform_filter.

        This method applies average blurring to the input image using
        SciPy's `uniform_filter`.
        
        Parameters
        ----------
        image: np.ndarray
            The input image to blur.
        **kwargs: dict[str, Any]
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
        self: AverageBlur, 
        image: torch.Tensor, 
        **kwargs: Any
    ) -> torch.Tensor:
        """Apply average blurring using PyTorch's avg_pool.

        This method applies average blurring to the input image using
        PyTorch's `avg_pool` functions.

        Parameters
        ----------
        image: torch.Tensor
            The input image to blur.
        **kwargs: dict[str, Any]
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
            x = x.unsqueeze(0)    # 1, ...

        x = x.unsqueeze(0)        # 1, C, ...

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
    """Applies a Gaussian blur to images using Gaussian kernels.

    This class blurs images by convolving them with a Gaussian filter, which
    smooths the image and reduces high-frequency details. The level of blurring
    is controlled by the standard deviation (`sigma`) of the Gaussian kernel.

    Parameters
    ----------
    sigma: float
        Standard deviation of the Gaussian kernel.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a Gaussian blur feature:
    >>> gaussian_blur = dt.GaussianBlur(sigma=2)
    >>> output_image = gaussian_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Visualize the input and output images:
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(input_image, cmap='gray')
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(output_image, cmap='gray')
    >>> plt.show()

    """

    def __init__(self: GaussianBlur, sigma: PropertyLike[float] = 2, **kwargs: Any):
        """Initialize the parameters for Gaussian blurring.

        This constructor initializes the parameters for Gaussian blurring.

        Parameters
        ----------
        sigma: float
            Standard deviation of the Gaussian kernel.
        **kwargs: Any
            Additional keyword arguments.

        """

        # self.sigma = float(sigma)
        # super().__init__(None, **kwargs)
        super().__init__(sigma=sigma, **kwargs)

    # NumPy backend
    def _get_numpy(
        self,
        image: np.ndarray,
        sigma: float,
        **kwargs: Any,
    ) -> np.ndarray:
        return ndimage.gaussian_filter(
            image,
            sigma=sigma,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

    # Torch backend
    def _get_torch(
        self,
        image: torch.Tensor,
        sigma: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        if sigma == 0:
            return image.clone()

        spatial = image.shape[-2:]
        if sigma >= 0.5 * max(spatial):
            mean = image.mean(dim=(-2, -1), keepdim=True)
            return mean.expand_as(image)

        mode = kwargs.get("mode", "reflect")
        if mode not in {"reflect", "constant", "replicate"}:
            raise ValueError(f"Unsupported mode '{mode}' for torch GaussianBlur")

        cval = kwargs.get("cval", 0.0)

        radius = int(np.ceil(3 * sigma))
        if radius == 0:
            return image.clone()

        coords = torch.arange(
            -radius,
            radius + 1,
            device=image.device,
            dtype=image.dtype,
        )
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        if image.ndim == 2:
            x_in = image.unsqueeze(0).unsqueeze(0)
            has_channels = False
        elif image.ndim == 3:
            x_in = image.movedim(-1, 0).unsqueeze(0)
            has_channels = True
        else:
            raise NotImplementedError(
                "GaussianBlur torch backend only supports 2D or HWC tensors."
            )

        C = x_in.shape[1]
        kx = kernel.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        ky = kernel.view(1, 1, -1, 1).repeat(C, 1, 1, 1)

        if mode == "constant":
            x_in = F.pad(x_in, (radius, radius, 0, 0), mode="constant", value=cval)
            x_in = F.conv2d(x_in, kx, groups=C)
            x_in = F.pad(x_in, (0, 0, radius, radius), mode="constant", value=cval)
            x_in = F.conv2d(x_in, ky, groups=C)
        else:
            x_in = F.pad(x_in, (radius, radius, 0, 0), mode=mode)
            x_in = F.conv2d(x_in, kx, groups=C)
            x_in = F.pad(x_in, (0, 0, radius, radius), mode=mode)
            x_in = F.conv2d(x_in, ky, groups=C)

        out = x_in.squeeze(0)
        if has_channels:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return out

#TODO ***JH*** revise MedianBlur - torch, typing, docstring, unit test
class MedianBlur(Blur):
    """Applies a median blur.

    This class replaces each pixel of the input image with the median value of
    its neighborhood. The `ksize` parameter determines the size of the
    neighborhood used to calculate the median filter. The median filter is
    useful for reducing noise while preserving edges. It is particularly
    effective for removing salt-and-pepper noise from images.

    - NumPy backend: `scipy.ndimage.median_filter`
    - Torch backend: explicit unfolding followed by `torch.median`

    Parameters
    ----------
    ksize: int
        Kernel size.
    **kwargs: dict
        Additional parameters sent to the blurring function.

    Notes
    -----
    Torch median blurring is significantly more expensive than mean or
    Gaussian blurring due to explicit tensor unfolding.

    Median blur is not differentiable. This is typically acceptable, as the
    operation is intended for denoising and preprocessing rather than as a
    trainable network layer.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a median blur feature:
    >>> median_blur = dt.MedianBlur(ksize=3)
    >>> output_image = median_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Visualize the input and output images:
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
        **kwargs: Any,
    ):
        if isinstance(ksize, int) and ksize % 2 == 0:
            raise ValueError("MedianBlur requires an odd kernel size.")
        super().__init__(ksize=ksize, **kwargs)

    # ---------- NumPy backend ----------

    def _get_numpy(
        self,
        image: np.ndarray,
        ksize: int,
        **kwargs: Any,
    ) -> np.ndarray:
        return ndimage.median_filter(
            image,
            size=ksize,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

    # ---------- Torch backend ----------

    def _get_torch(
        self,
        image: torch.Tensor,
        ksize: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        if ksize % 2 == 0:
            raise ValueError("MedianBlur requires odd kernel size.")

        if ksize == 1:
            return image.clone()

        has_channels = (image.ndim == 3)

        if has_channels:
            image = image.movedim(-1, 0)
        else:
            image = image.unsqueeze(0)

        image = image.unsqueeze(0)  # (1, C, ...)

        spatial_dims = image.ndim - 2
        pad = ksize // 2

        pad_tuple = []
        for _ in range(spatial_dims):
            pad_tuple = [pad, pad] + pad_tuple
        pad_tuple = tuple(pad_tuple)

        image = F.pad(image, pad_tuple, mode=kwargs.get("mode", "reflect"))

        if spatial_dims == 2:
            x = image.unfold(2, ksize, 1).unfold(3, ksize, 1)
        elif spatial_dims == 3:
            x = image.unfold(2, ksize, 1).unfold(3, ksize, 1).unfold(4, ksize, 1)
        else:
            raise NotImplementedError

        x = x.contiguous().view(*x.shape[:-spatial_dims], -1)
        x = x.median(dim=-1).values

        x = x.squeeze(0)
        if has_channels:
            x = x.movedim(0, -1)
        else:
            x = x.squeeze(0)

        return x

#TODO ***CM*** revise typing, docstring, unit test
class Pool(Feature):
    """Abstract base class for pooling features."""

    def __init__(
        self,
        ksize: PropertyLike[int | tuple[int, int] | tuple[int, int, int]] = 2,
        **kwargs: Any,
    ):
        self.ksize = self._normalize_ksize(ksize)
        super().__init__(**kwargs)

    @staticmethod
    def _normalize_ksize(ksize) -> tuple[int, int, int]:
        if isinstance(ksize, int):
            return (ksize, ksize, ksize)

        if isinstance(ksize, tuple):
            if len(ksize) == 2:
                kx, ky = ksize
                return (kx, ky, 1)
            if len(ksize) == 3:
                return ksize

        raise TypeError(
            "ksize must be int, (kx, ky), or (kz, kx, ky)"
        )


    def get(
        self,
        image: np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        
        backend = self.get_backend()

        if backend == "torch":
            # ---- HARD GUARD: torch only ----
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            return self._get_torch(
                image,
                **kwargs,
            )

        elif backend == "numpy":
            # ---- HARD GUARD: numpy only ----
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "NumPy backend selected but image is not a np.ndarray"
                )

            return self._get_numpy(
                image,
                **kwargs,
            )
        
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    def _get_pool_size(self, array):
        px, py, pz = self.ksize

        if array.ndim == 2:
            return px, py, 1

        if array.ndim == 3:
            # treat as channels-last only if explicitly small AND typical
            if array.shape[-1] in (1, 3, 4):
                return px, py, 1

        return px, py, pz


    def _crop_center(self, array):
        px, py, pz = self._get_pool_size(array)

        # 2D or effectively 2D (channels-last)
        if array.ndim < 3 or pz == 1:
            H, W = array.shape[:2]
            crop_h = (H // px) * px
            crop_w = (W // py) * py
            return array[:crop_h, :crop_w, ...]

        # 3D volume
        H, W, Z = array.shape[:3]
        crop_h = (H // px) * px
        crop_w = (W // py) * py
        crop_z = (Z // pz) * pz
        return array[:crop_h, :crop_w, :crop_z, ...]

    def _get_numpy(self, image: np.ndarray, **kwargs):
        raise NotImplementedError

    def _get_torch(self, image: torch.Tensor, **kwargs):
        raise NotImplementedError


class AveragePooling(Pool):
    """Average pooling feature.

    Downsamples the input by applying mean pooling over non-overlapping
    blocks of size `ksize`, preserving the center of the image and never
    pooling over channel dimensions.

    Works with NumPy and PyTorch backends.
    """

    # ---------- NumPy backend ----------

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        # 2D or effectively 2D (channels-last)
        if image.ndim < 3 or pz == 1:
            block_size = (px, py) + (1,) * (image.ndim - 2)
        else:
            # 3D volume (optionally with channels)
            block_size = (px, py, pz) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.mean,
        )

    # ---------- Torch backend ----------

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        # ---------- 2D ----------
        if image.ndim == 2:
            x = image.unsqueeze(0).unsqueeze(0)
            pooled = F.avg_pool2d(x, (px, py), (px, py))
            return pooled.squeeze(0).squeeze(0)

        # ---------- 3D tensor ----------
        elif image.ndim == 3:

            # (H, W, C)
            if image.shape[-1] in (1, 3, 4) or pz == 1:
                x = image.movedim(-1, 0).unsqueeze(0)  # (1,C,H,W)

                pooled = F.avg_pool2d(x, (px, py), (px, py))

                return pooled.squeeze(0).movedim(0, -1)

            # (H, W, Z)
            else:
                x = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,Z,H,W)

                pooled = F.avg_pool3d(x, (pz, px, py), (pz, px, py))

                return pooled.squeeze(0).squeeze(0).permute(1, 2, 0)

        # ---------- 4D tensor ----------
        elif image.ndim == 4:
            # (H, W, Z, C)
            x = image.permute(3, 2, 0, 1).unsqueeze(0)  # (1,C,Z,H,W)

            pooled = F.avg_pool3d(x, (pz, px, py), (pz, px, py))

            return pooled.squeeze(0).permute(2, 3, 1, 0)

        else:
            raise NotImplementedError(f"Unsupported shape {image.shape}")


class MaxPooling(Pool):
    """Max pooling feature.

    Downsamples the input by applying max pooling over non-overlapping
    blocks of size `ksize`, preserving the center of the image and never
    pooling over channel dimensions.

    Works with NumPy and PyTorch backends.

    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        if image.ndim < 3 or pz == 1:
            block_size = (px, py) + (1,) * (image.ndim - 2)
        else:
            block_size = (px, py, pz) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.max,
        )

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        # ---------- 2D ----------
        if image.ndim == 2:
            x = image.unsqueeze(0).unsqueeze(0)
            pooled = F.max_pool2d(x, (px, py), (px, py))
            return pooled.squeeze(0).squeeze(0)

        # ---------- 3D tensor ----------
        elif image.ndim == 3:

            # (H, W, C)
            if image.shape[-1] in (1, 3, 4) or pz == 1:
                x = image.movedim(-1, 0).unsqueeze(0)  # (1,C,H,W)

                pooled = F.max_pool2d(x, (px, py), (px, py))

                return pooled.squeeze(0).movedim(0, -1)

            # (H, W, Z)
            else:
                x = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

                pooled = F.max_pool3d(x, (pz, px, py), (pz, px, py))

                return pooled.squeeze(0).squeeze(0).permute(1, 2, 0)

        # ---------- 4D ----------
        elif image.ndim == 4:
            # (H, W, Z, C)
            x = image.permute(3, 2, 0, 1).unsqueeze(0)

            pooled = F.max_pool3d(x, (pz, px, py), (pz, px, py))

            return pooled.squeeze(0).permute(2, 3, 1, 0)

        else:
            raise NotImplementedError(f"Unsupported shape {image.shape}")


class MinPooling(Pool):
    """Min pooling feature.

    Downsamples the input by applying min pooling over non-overlapping
    blocks of size `ksize`, preserving the center of the image and never
    pooling over channel dimensions.

    Works with NumPy and PyTorch backends.

    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        if image.ndim < 3 or pz == 1:
            block_size = (px, py) + (1,) * (image.ndim - 2)
        else:
            block_size = (px, py, pz) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.min,
        )

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        # ---------- 2D ----------
        if image.ndim == 2:
            x = image.unsqueeze(0).unsqueeze(0)
            pooled = -F.max_pool2d(-x, (px, py), (px, py))
            return pooled.squeeze(0).squeeze(0)

        # ---------- 3D tensor ----------
        elif image.ndim == 3:

            # (H, W, C)
            if image.shape[-1] in (1, 3, 4) or pz == 1:
                x = image.movedim(-1, 0).unsqueeze(0)  # (1,C,H,W)

                pooled = -F.max_pool2d(-x, (px, py), (px, py))

                return pooled.squeeze(0).movedim(0, -1)

            # (H, W, Z)
            else:
                x = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

                pooled = -F.max_pool3d(-x, (pz, px, py), (pz, px, py))

                return pooled.squeeze(0).squeeze(0).permute(1, 2, 0)

        # ---------- 4D ----------
        elif image.ndim == 4:
            # (H, W, Z, C)
            x = image.permute(3, 2, 0, 1).unsqueeze(0)

            pooled = -F.max_pool3d(-x, (pz, px, py), (pz, px, py))

            return pooled.squeeze(0).permute(2, 3, 1, 0)

        else:
            raise NotImplementedError(f"Unsupported shape {image.shape}")


class SumPooling(Pool):
    """Sum pooling feature.

    Downsamples the input by applying sum pooling over non-overlapping
    blocks of size `ksize`, preserving the center of the image and never
    pooling over channel dimensions.

    Works with NumPy and PyTorch backends.
    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        if image.ndim < 3 or pz == 1:
            block_size = (px, py) + (1,) * (image.ndim - 2)
        else:
            block_size = (px, py, pz) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.sum,
        )

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        # ---------- 2D ----------
        if image.ndim == 2:
            x = image.unsqueeze(0).unsqueeze(0)
            pooled = F.avg_pool2d(x, (px, py), (px, py)) * (px * py)
            return pooled.squeeze(0).squeeze(0)

        # ---------- 3D tensor ----------
        elif image.ndim == 3:

            # (H, W, C)
            if image.shape[-1] in (1, 3, 4) or pz == 1:
                x = image.movedim(-1, 0).unsqueeze(0)

                pooled = F.avg_pool2d(x, (px, py), (px, py)) * (px * py)

                return pooled.squeeze(0).movedim(0, -1)

            # (H, W, Z)
            else:
                x = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

                pooled = F.avg_pool3d(x, (pz, px, py), (pz, px, py)) * (px * py * pz)

                return pooled.squeeze(0).squeeze(0).permute(1, 2, 0)

        # ---------- 4D ----------
        elif image.ndim == 4:
            # (H, W, Z, C)
            x = image.permute(3, 2, 0, 1).unsqueeze(0)

            pooled = F.avg_pool3d(x, (pz, px, py), (pz, px, py)) * (px * py * pz)

            return pooled.squeeze(0).permute(2, 3, 1, 0)

        else:
            raise NotImplementedError(f"Unsupported shape {image.shape}")


class MedianPooling(Pool):
    """Median pooling feature.

    Downsamples the input by applying median pooling over non-overlapping
    blocks of size `ksize`, preserving the center of the image and never
    pooling over channel dimensions.

    Notes
    -----
    - NumPy backend uses `skimage.measure.block_reduce`
    - Torch backend performs explicit unfolding followed by `median`
    - Torch median pooling is significantly more expensive than mean/max

    Median pooling is not differentiable and should not be used inside
    trainable neural networks requiring gradient-based optimization.
    
    """

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        if image.ndim < 3 or pz == 1:
            block_size = (px, py) + (1,) * (image.ndim - 2)
        else:
            block_size = (px, py, pz) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.median,
        )

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch

        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        # ---------- helper inline pattern ----------
        def _median_lastdim(x):
            vals, _ = torch.sort(x, dim=-1)
            n = vals.shape[-1]
            mid = n // 2
            if n % 2 == 1:
                return vals[..., mid]
            else:
                return (vals[..., mid - 1] + vals[..., mid]) / 2


        # ---------- 2D ----------
        if image.ndim == 2:
            x = image.unfold(0, px, px).unfold(1, py, py)
            x = x.contiguous().view(x.shape[0], x.shape[1], -1)

            return _median_lastdim(x)


        # ---------- 3D ----------
        elif image.ndim == 3:

            # (H, W, C)
            if image.shape[-1] in (1, 3, 4) or pz == 1:
                x = image.permute(2, 0, 1)  # (C,H,W)

                x = x.unfold(1, px, px).unfold(2, py, py)
                x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], -1)

                out = _median_lastdim(x)
                return out.permute(1, 2, 0)

            # (H, W, Z)
            else:
                x = image.permute(2, 0, 1)  # (Z,H,W)

                x = x.unfold(0, pz, pz).unfold(1, px, px).unfold(2, py, py)
                x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], -1)

                out = _median_lastdim(x)
                return out.permute(1, 2, 0)


        # ---------- 4D ----------
        elif image.ndim == 4:
            # (H, W, Z, C)
            x = image.permute(3, 2, 0, 1)  # (C,Z,H,W)

            x = x.unfold(1, pz, pz).unfold(2, px, px).unfold(3, py, py)
            x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)

            out = _median_lastdim(x)
            return out.permute(2, 3, 1, 0)


        else:
            raise NotImplementedError(f"Unsupported shape {image.shape}")

class Resize(Feature):
    """Resize an image to a specified size.

    `Resize` resizes images following the channels-last semantic
    convention.

    The operation supports both NumPy arrays and PyTorch tensors:
    - NumPy arrays are resized using OpenCV (`cv2.resize`).
    - PyTorch tensors are resized using `torch.nn.functional.interpolate`.

    In all cases, the input is interpreted as having spatial dimensions
    first and an optional channel dimension last.

    Parameters
    ----------
    dsize : PropertyLike[tuple[int, int]]
        Target output size given as (width, height). This convention is
        backend-independent and applies equally to NumPy and PyTorch inputs.

    **kwargs : Any
        Additional keyword arguments forwarded to the underlying resize
        implementation:
        - NumPy backend: passed to `cv2.resize`.
        - PyTorch backend: passed to
        `torch.nn.functional.interpolate`.

    Methods
    -------
    get(
        image: np.ndarray | torch.Tensor,
        dsize: tuple[int, int],
        **kwargs
    ) -> np.ndarray | torch.Tensor
        Resize the input image to the specified size.

    Examples
    --------
    NumPy example:

    >>> import numpy as np
    >>> input_image = np.random.rand(16, 16)
    >>> feature = dt.math.Resize(dsize=(8, 4))   # (width=8, height=4)
    >>> resized_image = feature.resolve(input_image)
    >>> resized_image.shape
    (4, 8)

    PyTorch example:

    >>> import torch
    >>> input_image = torch.rand(16, 16)         
    >>> feature = dt.math.Resize(dsize=(8, 4))
    >>> resized_image = feature.resolve(input_image)
    >>> resized_image.shape
    torch.Size([4, 8])

    Notes
    -----
    - Resize follows channels-last semantics, consistent with other features
    such as Pool and Blur.
    - Torch tensors with channels-first layout (e.g. (C, H, W) or
    (N, C, H, W)) are not supported and must be converted to
    channels-last format before resizing.
    - For PyTorch tensors, bilinear interpolation is used with
    `align_corners=False`, closely matching OpenCV’s default behavior.

    """


    def __init__(
        self: Resize,
        dsize: PropertyLike[tuple[int, int]] = (256, 256),
        **kwargs: Any,
    ):
        """Initialize the parameters for the Resize feature.

        Parameters
        ----------
        dsize: PropertyLike[tuple[int, int]]
            The target size. dsize is always (width, height) for both backends. 
            Default is (256, 256).
        **kwargs: Any
            Additional arguments passed to the parent `Feature` class.

        """

        super().__init__(dsize=dsize, **kwargs)

    def get(
        self: Resize,
        image: np.ndarray | torch.Tensor,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Resize the input image to the specified size.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image following channels-last semantics.

            Supported shapes are:
            - (H, W)
            - (H, W, C)
            - (Z, H, W)
            - (Z, H, W, C)

            For PyTorch tensors, channels-first layouts such as (C, H, W) or
            (N, C, H, W) are not supported and must be converted to
            channels-last format before calling `Resize`.

        dsize : tuple[int, int]
            Desired output size given as (width, height). This convention is
            backend-independent and applies to both NumPy and PyTorch inputs.

        **kwargs : Any
            Additional keyword arguments passed to the underlying resize
            implementation:
            - NumPy backend: forwarded to `cv2.resize`.
            - PyTorch backend: forwarded to `torch.nn.functional.interpolate`.

        Returns
        -------
        np.ndarray or torch.Tensor
            The resized image, with the same type and dimensionality layout as
            the input image.

        Notes
        -----
        - Resize follows the same channels-last semantic convention as other
        features in `deeptrack.math`.
        - For PyTorch tensors, resizing uses bilinear interpolation with
        `align_corners=False`, which closely matches OpenCV’s default behavior.

        """

        backend = self.get_backend()

        if backend == "torch":
            # ---- HARD GUARD: torch only ----
            if not isinstance(image, torch.Tensor):
                raise TypeError(
                    "Torch backend selected but image is not a torch.Tensor"
                )

            return self._get_torch(
                image,
                dsize=dsize,
                **kwargs,
            )

        elif backend == "numpy":
            # ---- HARD GUARD: numpy only ----
            if not isinstance(image, np.ndarray):
                raise TypeError(
                    "NumPy backend selected but image is not a np.ndarray"
                )

            return self._get_numpy(
                image,
                dsize=dsize,
                **kwargs,
            )
        
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    def _get_numpy(
        self,
        image: np.ndarray,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> np.ndarray:

        target_w, target_h = map(int, dsize)

        # --- 2D ---
        if image.ndim == 2:
            return utils.safe_call(
                cv2.resize,
                positional_args=[image, (target_w, target_h)],
                **kwargs,
            )

        # --- (H, W, C) ---
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            return utils.safe_call(
                cv2.resize,
                positional_args=[image, (target_w, target_h)],
                **kwargs,
            )

        # --- (H, W, Z) ---
        if image.ndim == 3:
            return np.stack([
                self._get_numpy(image[..., i], dsize, **kwargs)
                for i in range(image.shape[-1])
            ], axis=-1)

        # --- 4D cases ---
        if image.ndim == 4:

            # (H, W, Z, C)
            if image.shape[-1] in (1, 3, 4):
                return np.stack([
                    self._get_numpy(image[..., i, :], dsize, **kwargs)
                    for i in range(image.shape[2])
                ], axis=2)

            # (Z, H, W, C)
            if image.shape[-1] in (1, 3, 4):
                return np.stack([
                    self._get_numpy(image[i], dsize, **kwargs)
                    for i in range(image.shape[0])
                ], axis=0)

            # fallback (treat first dim as batch/Z)
            return np.stack([
                self._get_numpy(image[i], dsize, **kwargs)
                for i in range(image.shape[0])
            ], axis=0)

        raise ValueError(f"Unsupported shape {image.shape}")

    def _get_torch(
        self,
        image: torch.Tensor,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        target_w, target_h = map(int, dsize)

        # --- 2D ---
        if image.ndim == 2:
            x = image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            x = F.interpolate(x, size=(target_h, target_w),
                            mode="bilinear", align_corners=False)
            return x.squeeze(0).squeeze(0)

        # --- (H, W, C) ---
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            x = image.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
            x = F.interpolate(x, size=(target_h, target_w),
                            mode="bilinear", align_corners=False)
            return x.squeeze(0).permute(1, 2, 0)

        # --- (H, W, Z) ---
        if image.ndim == 3:
            return torch.stack([
                self._get_torch(image[..., i], dsize, **kwargs)
                for i in range(image.shape[-1])
            ], dim=-1)

        # --- 4D cases ---
        if image.ndim == 4:

            # (H, W, Z, C)
            if image.shape[-1] in (1, 3, 4):
                return torch.stack([
                    self._get_torch(image[..., i, :], dsize, **kwargs)
                    for i in range(image.shape[2])
                ], dim=2)

            # (Z, H, W, C)
            if image.shape[-1] in (1, 3, 4):
                return torch.stack([
                    self._get_torch(image[i], dsize, **kwargs)
                    for i in range(image.shape[0])
                ], dim=0)

            # fallback (treat first dim as batch/Z)
            return torch.stack([
                self._get_torch(image[i], dsize, **kwargs)
                for i in range(image.shape[0])
            ], dim=0)

        raise ValueError(f"Unsupported tensor shape {image.shape}")

#TODO ***JH*** revise BlurCV2 - torch, typing, docstring, unit test
class BlurCV2(Feature):
    """Apply a blurring filter using OpenCV2.

    This class applies a blurring filter to an image using OpenCV2. The
    filter_function must be an OpenCV-compatible function that accepts a src
    keyword argument (e.g., cv2.GaussianBlur, cv2.bilateralFilter, etc.).

    Parameters
    ----------
    filter_function: Callable
        The blurring function to apply.
    mode: str
        Border mode for handling boundaries (e.g., 'reflect').

    Methods
    -------
    `get(image: np.ndarray, **kwargs: Any) --> np.ndarray`
        Applies the blurring filter to the input image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> import cv2

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a blur feature using the Gaussian blur function:
    >>> blur = dt.BlurCV2(
    ...     filter_function=cv2.GaussianBlur,
    ...     ksize=(5, 5),
    ...     sigmaX=1,
    ...     mode='reflect',
    ... )
    >>> output_image = blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Notes
    -----
    BlurCV2 is NumPy-only and does not support PyTorch tensors.
    This class is intended for OpenCV-specific filters that are
    not available in the backend-agnostic math layer.

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
        """Initialize the parameters for blurring input features.

        This constructor initializes the parameters for blurring input
        features.

        Parameters
        ----------
        filter_function: Callable
            The blurring function to apply.
        mode: str
            Border mode for handling boundaries (e.g., 'reflect').
        **kwargs: Any
            Additional keyword arguments.

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

        filter_fn = getattr(cv2, self.filter) if isinstance(self.filter, str) else self.filter

        try:
            border_attr = self._MODE_TO_BORDER[mode]
        except KeyError as e:
            raise ValueError(f"Unsupported border mode '{mode}'") from e

        try:
            border = getattr(cv2, border_attr)
        except AttributeError as e:
            raise RuntimeError(f"OpenCV missing border constant '{border_attr}'") from e

        # preserve legacy behavior
        kwargs.pop("name", None)

        return filter_fn(
            src=image,
            borderType=border,
            **kwargs,
        )


#TODO ***JH*** revise BilateralBlur - torch, typing, docstring, unit test
class BilateralBlur(BlurCV2):
    """Blur an image using a bilateral filter.

    Bilateral filters blur homogenous areas while trying to
    preserve edges.

    Parameters
    ----------
    d: int
        Diameter of each pixel neighborhood with value range.
    sigma_color: float
        Filter sigma in the color space with value range. A
        large value of the parameter means that farther colors within the
        pixel neighborhood (see `sigma_space`) will be mixed together,
        resulting in larger areas of semi-equal color.
    sigma_space: float
        Filter sigma in the coordinate space with value range. A
        large value of the parameter means that farther pixels will influence
        each other as long as their colors are close enough (see
        `sigma_color`).
    **kwargs: dict
        Additional parameters sent to the blurring function.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> import cv2

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a bilateral blur feature:
    >>> bilateral_blur = dt.BilateralBlur(
    ...     d=5,
    ...     sigma_color=50,
    ...     sigma_space=50,
    ...     mode='reflect',
    ... )
    >>> output_image = bilateral_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Notes
    -----
    BilateralBlur is NumPy-only and does not support PyTorch tensors.

    """

    def __init__(
        self: BilateralBlur,
        d: PropertyLike[int] = 3,
        sigma_color: PropertyLike[float] = 50,
        sigma_space: PropertyLike[float] = 50,
        **kwargs: Any,
    ):
        """Initialize the parameters for bilateral blurring.

        This constructor initializes the parameters for bilateral blurring.

        Parameters
        ----------
        d: int
            Diameter of each pixel neighborhood with value range.
        sigma_color: number
            Filter sigma in the color space with value range. A
            large value of the parameter means that farther colors within the
            pixel neighborhood (see `sigma_space`) will be mixed together,
            resulting in larger areas of semi-equal color.
        sigma_space: number
            Filter sigma in the coordinate space with value range. A
            large value of the parameter means that farther pixels will influence
            each other as long as their colors are close enough (see
            `sigma_color`).
        **kwargs: dict
            Additional parameters sent to the blurring function.

        """

        super().__init__(
            filter_function="bilateralFilter",
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            **kwargs,
        )

# def isotropic_dilation(
#     mask: np.ndarray | torch.Tensor,
#     radius: float,
#     *,
#     backend: Literal["numpy", "torch"],
#     device=None,
#     dtype=None,
# ) -> np.ndarray | torch.Tensor:
#     """ Binary dilation using isotropic kernel.

    
#     - NumPy backend uses a true Euclidean ball.
#     - Torch backend uses a cubic structuring element (approximate).
#     - Operation is non-differentiable.

#     """

#     def _validate_mask(mask):
#         if mask.ndim >= 3 and mask.shape[-1] in (1, 3, 4):
#             raise ValueError(
#                 "Morphological operations expect binary masks, not channel images. "
#                 f"Got shape {mask.shape} which looks like (H, W, C)."
#             )

#     if radius <= 0:
#         return mask
    
#     _validate_mask(mask)

#     if backend == "numpy":
#         from skimage.morphology import isotropic_dilation
#         return isotropic_dilation(mask, radius)

#     r = int(np.ceil(radius))

#     if mask.ndim == 2:
#         # --- 2D ---
#         kernel = torch.ones(
#             (1, 1, 2*r+1, 2*r+1),
#             device=device or mask.device,
#             dtype=dtype or torch.float32,
#         )
#         x = mask.to(kernel.dtype)[None, None]
#         y = F.conv2d(x, kernel, padding=r)

#     elif mask.ndim == 3:
#         # --- 3D ---
#         kernel = torch.ones(
#             (1, 1, 2*r+1, 2*r+1, 2*r+1),
#             device=device or mask.device,
#             dtype=dtype or torch.float32,
#         )
#         x = mask.to(kernel.dtype)[None, None]
#         y = F.conv3d(x, kernel, padding=r)

#     else:
#         raise ValueError("Mask must be 2D or 3D for torch backend")

#     return (y[0, 0] > 0)

# def isotropic_erosion(
#     mask: np.ndarray | torch.Tensor,
#     radius: float,
#     *,
#     backend: Literal["numpy", "torch"],
#     device=None,
#     dtype=None,
# ) -> np.ndarray | torch.Tensor:
#     """ 
#     Binary erosion using an isotropic (NumPy) or box-shaped (Torch) kernel.
    
#     Notes
#     -----
#     - NumPy backend uses a true Euclidean ball.
#     - Torch backend uses a cubic structuring element (approximate).
#     - Torch backend supports 3D masks only.
#     - Operation is non-differentiable.

#     """

#     def _validate_mask(mask):
#         if mask.ndim >= 3 and mask.shape[-1] in (1, 3, 4):
#             raise ValueError(
#                 "Morphological operations expect binary masks, not channel images. "
#                 f"Got shape {mask.shape} which looks like (H, W, C)."
#             )

#     if radius <= 0:
#         return mask
    
#     _validate_mask(mask)

#     if backend == "numpy":
#         from skimage.morphology import isotropic_erosion
#         return isotropic_erosion(mask, radius)

#     r = int(np.ceil(radius))

#     if mask.ndim == 2:
#         kernel = torch.ones(
#             (1, 1, 2*r+1, 2*r+1),
#             device=device or mask.device,
#             dtype=dtype or torch.float32,
#         )
#         x = mask.to(kernel.dtype)[None, None]
#         y = F.conv2d(x, kernel, padding=r)

#     elif mask.ndim == 3:
#         kernel = torch.ones(
#             (1, 1, 2*r+1, 2*r+1, 2*r+1),
#             device=device or mask.device,
#             dtype=dtype or torch.float32,
#         )
#         x = mask.to(kernel.dtype)[None, None]
#         y = F.conv3d(x, kernel, padding=r)

#     else:
#         raise ValueError("Mask must be 2D or 3D for torch backend")

#     required = kernel.numel()
#     return (y[0, 0] >= required)


def _move_channel_last(
    x: np.ndarray | torch.Tensor, 
    channel_axis: int | None,
) -> tuple[np.ndarray | torch.Tensor, int | None]:
    """Move the channel axis to the last position if specified."""

    if channel_axis is None:
        return x, None
    return xp.moveaxis(x, channel_axis, -1), channel_axis

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
    mask : np.ndarray or torch.Tensor
        Input mask with shape (H, W) or (H, W, D).
    channel_axis : int or None
        Axis corresponding to channels. If None, no channel interpretation
        is applied.

    Returns
    -------
    mask : np.ndarray or torch.Tensor
        Possibly reshaped mask used for computation.
    channelwise : bool
        Whether to apply the operation independently along a channel axis.
    restore_channel : bool
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
    """Apply isotropic binary dilation to a mask.

    Performs morphological dilation using a structuring element of radius
    `radius`.
    - If `channel_axis is None`:
        - (H, W) → treated as 2D
        - (H, W, Z) → treated as 3D volume
    - If `channel_axis` is specified:
        - Operation is applied independently along that axis
    - Singleton channel:
        - (H, W, 1) is treated as 2D and restored after processing

    Parameters
    ----------
    mask : np.ndarray or torch.Tensor
        Input mask. Values are interpreted as binary (`mask > 0`).
    radius : float
        Radius of the structuring element. If `radius <= 0`, the input
        is returned unchanged.
    backend : {"numpy", "torch"}, optional
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
        Dilated mask with the same shape as the input.

    Notes
    -----
    **NumPy backend**
        Uses `skimage.morphology.isotropic_dilation`, which implements
        true Euclidean (distance-transform-based) isotropic dilation.

    **Torch backend**
        Uses convolution with a full (square/cubic) kernel:
        - 2D: `conv2d`
        - 3D: `conv3d`
        A pixel is activated if any element in the neighborhood is active.

        This is **not strictly isotropic** (Chebyshev distance), and
        results may differ from the NumPy implementation, especially
        for small radii.

    - Output is always boolean.
    - Shape is preserved in all cases.
    
    """

    if radius <= 0:
        return mask

    mask, channelwise, restore_channel = _prepare_mask(mask, channel_axis)
    
    if channelwise:
        xp = np if backend == "numpy" else __import__("torch")

        # move channel axis to last
        mask_moved = np.moveaxis(mask, channel_axis, -1) if backend == "numpy" else mask.movedim(channel_axis, -1)

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
        from skimage.morphology import isotropic_dilation
        out =  isotropic_dilation(mask, radius)
        if restore_channel:
            return out[..., None]
        return out

    r = int(np.ceil(radius))

    if mask.ndim == 2:
        kernel = torch.ones(
            (1, 1, 2*r+1, 2*r+1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv2d(x, kernel, padding=r)

    elif mask.ndim == 3:
        kernel = torch.ones(
            (1, 1, 2*r+1, 2*r+1, 2*r+1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv3d(x, kernel, padding=r)

    else:
        raise ValueError("Mask must be 2D or 3D")

    out = (y[0, 0] > 0)
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
    """Apply isotropic binary erosion to a mask.

    Performs morphological erosion using a structuring element of radius
    `radius`.
    - If `channel_axis is None`:
        - (H, W) → treated as 2D
        - (H, W, Z) → treated as 3D volume
    - If `channel_axis` is specified:
        - Operation is applied independently along that axis
    - Singleton channel:
        - (H, W, 1) is treated as 2D and restored after processing

    Parameters
    ----------
    mask : np.ndarray or torch.Tensor
        Input mask. Values are interpreted as binary (`mask > 0`).
    radius : float
        Radius of the structuring element. If `radius <= 0`, the input
        is returned unchanged.
    backend : {"numpy", "torch"}, optional
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
        Eroded mask with the same shape as the input.

    Notes
    -----
    **NumPy backend**
        Uses `skimage.morphology.isotropic_erosion`, based on distance
        transforms and true Euclidean geometry.

    **Torch backend**
        Uses convolution with a full kernel:
        - A pixel is preserved only if **all elements** in the neighborhood
          are active.

        This corresponds to a box-shaped structuring element and differs
        from true isotropic erosion.

    Differences vs NumPy:
        - Torch uses Chebyshev distance (square/cube neighborhood)
        - NumPy uses Euclidean distance
        - Results may differ near boundaries and for small radii

    - Output is always boolean.
    - Shape is preserved in all cases.
    
    """
    
    if radius <= 0:
        return mask

    mask, channelwise, restore_channel = _prepare_mask(mask, channel_axis)
    
    if channelwise:
        xp = np if backend == "numpy" else __import__("torch")

        # move channel axis to last
        mask_moved = np.moveaxis(mask, channel_axis, -1) if backend == "numpy" else mask.movedim(channel_axis, -1)

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
        from skimage.morphology import isotropic_erosion
        out =  isotropic_erosion(mask, radius)
        if restore_channel:
            return out[..., None]
        return out
    
    r = int(np.ceil(radius))

    if mask.ndim == 2:
        kernel = torch.ones(
            (1, 1, 2*r+1, 2*r+1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv2d(x, kernel, padding=r)

    elif mask.ndim == 3:
        kernel = torch.ones(
            (1, 1, 2*r+1, 2*r+1, 2*r+1),
            device=device or mask.device,
            dtype=dtype or torch.float32,
        )
        x = mask.to(kernel.dtype)[None, None]
        y = F.conv3d(x, kernel, padding=r)

    else:
        raise ValueError("Mask must be 2D or 3D")

    required = kernel.numel()
    out = (y[0, 0] >= required)
    if restore_channel:
        return out[..., None]

    return out