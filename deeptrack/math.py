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

from typing import Any, Callable, Dict, Tuple, TYPE_CHECKING

import array_api_compat as apc
import numpy as np
from scipy import ndimage
import skimage
import skimage.measure

from deeptrack import utils, OPENCV_AVAILABLE, TORCH_AVAILABLE
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
    """Average of input images.

    Computes the average of input images along the specified axis or axes.
    By default, averaging is performed along axis 0 (the batch dimension).

    If `features` is specified, each feature in the list is first resolved,
    and their results are averaged.

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
    get(images: list[array], axis: int or tuple[int], **kwargs: Any) -> array
        Computes the average of the input images along the given axis.

    Examples
    --------
    >>> import deeptrack as dt

    Create two input images:
    >>> import numpy as np
    >>>
    >>> input_image1 = np.random.rand(10, 30, 20)
    >>> input_image2 = np.random.rand(10, 30, 20)

    Define a pipeline with the average feature along the batch dimension:
    >>> average = dt.Average(axis=0)
    >>> output_image = average([input_image1, input_image2])
    >>> output_image.shape
    (10, 30, 20)

    Define a pipeline with the average feature along the first image
    dimension:
    >>> average = dt.Average(axis=1)
    >>> output_image = average([input_image1, input_image2])
    >>> output_image.shape
    (2, 30, 20)

    Define a pipeline averaging each image:
    >>> average = dt.Average(axis=(1, 2, 3))
    >>> output_image = average([input_image1, input_image2])
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
    """Clip the input from a minimum to a maximum value.

    This feature clips all values in the input image such that they fall within
    the specified range [`min`, `max`].

    Parameters
    ----------
    min: float, optional
        Lower bound. Values below this will be set to `min`. It defaults to
        `-np.inf`.
    max: float, optional
        Upper bound. Values above this will be set to `max`. It defaults to
        `+np.inf`.

    Methods
    -------
    get(image: array, min: float, max: float, **kwargs: Any) -> array
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
    """Image normalization using min-max scaling.

    It applies a linear transformation that maps the input to the range [`min`,
    `max`].

    It uses the global minimum and maximum of the image to perform scaling.
    If the image has no dynamic range (`ptp = 0`), the output is set to 0.

    Parameters
    ----------
    min: float, optional
        Lower bound of the transformation. It defaults to 0.
    max: float, optional
        Upper bound of the transformation. It defaults to 1.
    featurewise: bool, optional
        Whether to normalize each feature independently. It default to `True`.

    Methods
    -------
    get(image: array, min: float, max: float, **kwargs: Any) -> array
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
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(min=min, max=max, featurewise=featurewise, **kwargs)

    def get(
        self: NormalizeMinMax,
        image: np.ndarray | torch.Tensor,
        min: float,
        max: float,
        featurewise: bool = True,
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

        Returns
        -------
        np.ndarray or torch.Tensor
            Min-max normalized image.

        """

        has_channels = image.ndim >= 3 and image.shape[-1] <= 4

        if featurewise and has_channels:
            # reduce over spatial dimensions only
            axis = tuple(range(image.ndim - 1))
            img_min = xp.min(image, axis=axis, keepdims=True)
            img_max = xp.max(image, axis=axis, keepdims=True)
        else:
            # global normalization
            img_min = xp.min(image)
            img_max = xp.max(image)

        ptp = img_max - img_min
        eps = xp.asarray(1e-8, dtype=image.dtype)
        ptp = xp.maximum(ptp, eps)

        image = (image - img_min) / ptp
        image = image * (max - min) + min

        image = xp.where(xp.isnan(image), xp.zeros_like(image), image)
        return image



class NormalizeStandard(Feature):
    """Image normalization using standardization.

    Standardizes the input image to have zero mean and unit standard
    deviation. Uses the population standard deviation (divides by N).

    Parameters
    ----------
    featurewise: bool, optional
        Whether to normalize each feature independently. It default to `True`,
        which is the only behavior currently implemented.

    Methods
    -------
    get(image: array, **kwargs: Any) -> array
        Standardizes the input image to mean 0 and std deviation 1.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:
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
        **kwargs: Any,
    ):
        """Initialize the parameters for standardization.

        This constructor initializes the parameters for standardization.

        Parameters
        ----------
        featurewise: bool, optional
            Whether to normalize each feature independently.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(featurewise=featurewise, **kwargs)

    def get(
        self: NormalizeStandard,
        image: np.ndarray | torch.Tensor,
        featurewise: bool,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Normalizes the input image to have mean 0 and standard deviation 1.

        Parameters
        ----------
        image: np.ndarray or torch.Tensor
            The input image to normalize.
        featurewise: bool
            Whether to normalize each feature (channel) independently.

        Returns
        -------
        np.ndarray or torch.Tensor
            The standardized image.
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
                featurewise=featurewise,
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
                featurewise=featurewise,
                **kwargs,
            )

        else:
            raise RuntimeError(f"Unknown backend: {backend}")


    # ------ NumPy backend ------
    
    def _get_numpy(
        self,
        image: np.ndarray,
        featurewise: bool,
        **kwargs: Any,
    ) -> np.ndarray:

        has_channels = image.ndim >= 3 and image.shape[-1] <= 4

        if featurewise and has_channels:
            axis = tuple(range(image.ndim - 1))
            mean = np.mean(image, axis=axis, keepdims=True)
            std = np.std(image, axis=axis, keepdims=True)  # population std
        else:
            mean = np.mean(image)
            std = np.std(image)

        std = np.maximum(std, 1e-8)

        out = (image - mean) / std
        out = np.where(np.isnan(out), 0.0, out)

        return out

    # ------ Torch backend ------

    def _get_torch(
        self,
        image: torch.Tensor,
        featurewise: bool,
        **kwargs: Any,
    ) -> torch.Tensor:

        has_channels = image.ndim >= 3 and image.shape[-1] <= 4

        if featurewise and has_channels:
            axis = tuple(range(image.ndim - 1))
            mean = image.mean(dim=axis, keepdim=True)
            std = image.std(dim=axis, keepdim=True, unbiased=False)
        else:
            mean = image.mean()
            std = image.std(unbiased=False)

        std = torch.clamp(std, min=1e-8)

        out = (image - mean) / std
        out = torch.nan_to_num(out, nan=0.0)

        return out


class NormalizeQuantile(Feature):
    """Image normalization using quantiles.

    Centers the image at the median and scales it such that the values at the
    specified lower and upper quantiles are mapped to −1 and +1, respectively.

    Parameters
    ----------
    quantiles : tuple[float, float]
        Quantile range used to compute the scaling factor. Must satisfy
        0.0 < q_min < q_max < 1.0.
    featurewise : bool, optional
        Whether to normalize each feature independently. Defaults to `True`.
        Currently, `True` is the only supported behavior.

    Methods
    -------
    get(image: array, quantiles: tuple[float, float], **kwargs) -> array
        Normalizes the input based on the given quantile range.

    Notes
    -----
    This operation is not differentiable. When used inside a gradient-based
    model, it will block gradient flow. Use with care if end-to-end 
    differentiability is required.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:
    >>> import numpy as np
    >>>
    >>> input_image = np.array([[10, 4], [4, -10]])

    Define a quantile normalizer:
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
            **kwargs,
        )

    def get(
        self,
        image: np.ndarray | torch.Tensor,
        quantiles: tuple[float, float],
        featurewise: bool,
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
                **kwargs,
            )

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    # ------ NumPy backend ------
    def _get_numpy(
        self: NormalizeQuantile,
        image: np.ndarray,
        quantiles: tuple[float, float],
        featurewise: bool,
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

        Returns
        -------
        np.ndarray or torch.Tensor
            The quantile-normalized image.
        
        """

        q_low_val, q_high_val = quantiles

        has_channels = image.ndim >= 3 and image.shape[-1] <= 4

        if featurewise and has_channels:
            axis = tuple(range(image.ndim - 1))
            q_low, q_high, median = np.quantile(
                image,
                (q_low_val, q_high_val, 0.5),
                axis=axis,
                keepdims=True,
            )
        else:
            q_low, q_high, median = np.quantile(
                image,
                (q_low_val, q_high_val, 0.5),
            )

        scale = q_high - q_low
        eps = np.asarray(1e-8, dtype=image.dtype)
        scale = np.maximum(scale, eps)

        image = (image - median) / scale 
        image = np.where(np.isnan(image), np.zeros_like(image), image)
        return image
    
    def _get_torch(
        self,
        image: torch.Tensor,
        quantiles: tuple[float, float],
        featurewise: bool,
        **kwargs: Any,
    ):
        q_low_val, q_high_val = quantiles

        if featurewise:
            if image.ndim < 3:
                # No channels → global quantile
                q = torch.tensor(
                    [q_low_val, q_high_val, 0.5],
                    device=image.device,
                    dtype=image.dtype,
                )
                q_low, q_high, median = torch.quantile(image, q)
            else:
                # channels-last: (..., C)
                spatial_dims = image.ndim - 1
                C = image.shape[-1]

                # flatten spatial dims
                x = image.reshape(-1, C)   # (N, C)

                q = torch.tensor(
                    [q_low_val, q_high_val, 0.5],
                    device=image.device,
                    dtype=image.dtype,
                )

                q_vals = torch.quantile(x, q, dim=0)
                q_low, q_high, median = q_vals

                # reshape for broadcasting
                shape = [1] * image.ndim
                shape[-1] = C
                q_low = q_low.view(shape)
                q_high = q_high.view(shape)
                median = median.view(shape)

        else:
            q = torch.tensor(
                [q_low_val, q_high_val, 0.5],
                device=image.device,
                dtype=image.dtype,
            )
            q_low, q_high, median = torch.quantile(image, q)

        scale = q_high - q_low
        scale = torch.clamp(scale, min=1e-8)

        image = (image - median) / scale
        image = torch.nan_to_num(image)

        return image


#TODO ***CM*** revise typing, docstring, unit test
class Blur(Feature):
    """Abstract blur feature with backend-dispatched implementations.
    
    This class serves as a base for blur features that support multiple
    backends (e.g., NumPy, Torch). Subclasses should implement backend-specific
    blurring logic via `_get_numpy` and/or `_get_torch` methods.
    
    Methods
    -------
    get(image: np.ndarray | torch.Tensor, **kwargs) -> np.ndarray | torch.Tensor
        Applies the appropriate backend-specific blurring method.
        
    _blur(xp, image: array, **kwargs) -> array
        Internal method that dispatches to the correct backend-specific blur
        implementation.

    """


    def get(
        self,
        image: np.ndarray | torch.Tensor,
        **kwargs,
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

    def _get_numpy(self, image: np.ndarray, **kwargs):
        raise NotImplementedError

    def _get_torch(self, image: torch.Tensor, **kwargs):
        raise NotImplementedError



#TODO ***CM*** revise AverageBlur - torch, typing, docstring, unit test
class AverageBlur(Blur):
    """Blur an image by computing simple means over neighbourhoods.

    Performs a (N-1)D convolution if the last dimension is smaller than
    the kernel size.

    Parameters
    ----------
    ksize: int
        Kernel size for the pooling operation.

    Methods
    -------
    `get(image: np.ndarray | torch.Tensor, ksize: int, **kwargs: Any) --> np.ndarray | torch.Tensor`
        Applies the average blurring filter to the input image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define an average blur feature:
    >>> average_blur = dt.AverageBlur(ksize=3)
    >>> output_image = average_blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    """

    def __init__(
        self: AverageBlur, 
        ksize: int = 3, 
        **kwargs: Any
    ) -> None:
        """Initialize the parameters for averaging input features.

        This constructor initializes the parameters for averaging input
        features.

        Parameters
        ----------
        ksize: int
            Kernel size for the pooling operation.
        **kwargs: Any
            Additional keyword arguments.

        """

        self.ksize = int(ksize)
        super().__init__(**kwargs)

    @staticmethod
    def _kernel_shape(shape: tuple[int, ...], ksize: int) -> tuple[int, ...]:
        # If last dim is channel and smaller than kernel, do not blur channels
        if shape[-1] < ksize:
            return (ksize,) * (len(shape) - 1) + (1,)
        return (ksize,) * len(shape)

    # ---------- NumPy backend ----------
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

        k = self._kernel_shape(image.shape, self.ksize)
        return ndimage.uniform_filter(
            image,
            size=k,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
            origin=kwargs.get("origin", 0),
            axes=tuple(range(len(k))),
        )

    # ---------- Torch backend ----------
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

        k = self._kernel_shape(tuple(image.shape), self.ksize)

        last_dim_is_channel = len(k) < image.ndim
        if last_dim_is_channel:
            image = image.movedim(-1, 0)   # C, ...
        else:
            image = image.unsqueeze(0)     # 1, ...

        # add batch dimension
        image = image.unsqueeze(0)         # 1, C, ...

        # symmetric padding
        pad = []
        for kk in reversed(k):
            p = kk // 2
            pad.extend([p, p])
        image = F.pad(
            image,
            tuple(pad),
            mode=kwargs.get("mode", "reflect"),
            value=kwargs.get("cval", 0),
        )

        # pooling by dimensionality
        if image.ndim == 3:
            out = F.avg_pool1d(image, kernel_size=k, stride=1)
        elif image.ndim == 4:
            out = F.avg_pool2d(image, kernel_size=k, stride=1)
        elif image.ndim == 5:
            out = F.avg_pool3d(image, kernel_size=k, stride=1)
        else:
            raise NotImplementedError(
                f"Input dimensionality {image.ndim - 2} not supported"
            )

        # restore layout
        out = out.squeeze(0)
        if last_dim_is_channel:
            out = out.movedim(0, -1)
        else:
            out = out.squeeze(0)

        return out


#TODO ***CM*** revise typing, docstring, unit test
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

        self.sigma = float(sigma)
        super().__init__(None, **kwargs)

    # ---------- NumPy backend ----------

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return ndimage.gaussian_filter(
            image,
            sigma=self.sigma,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

    # ---------- Torch backend ----------

    @staticmethod
    def _gaussian_kernel_1d(
        sigma: float,
        device,
        dtype,
    ) -> torch.Tensor:
        radius = int(np.ceil(3 * sigma))
        x = torch.arange(
            -radius,
            radius + 1,
            device=device,
            dtype=dtype,
        )
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        kernel_1d = self._gaussian_kernel_1d(
            self.sigma,
            device=image.device,
            dtype=image.dtype,
        )

        # channel-last handling
        last_dim_is_channel = image.ndim >= 3
        if last_dim_is_channel:
            image = image.movedim(-1, 0)  # C, ...
        else:
            image = image.unsqueeze(0)    # 1, ...

        # add batch dimension
        image = image.unsqueeze(0)        # 1, C, ...

        spatial_dims = image.ndim - 2
        C = image.shape[1]

        for d in range(spatial_dims):
            k = kernel_1d
            shape = [1] * spatial_dims
            shape[d] = -1
            k = k.view(1, 1, *shape)
            k = k.repeat(C, 1, *([1] * spatial_dims))

            pad = [0, 0] * spatial_dims
            radius = k.shape[2 + d] // 2
            pad[-(2 * d + 2)] = radius
            pad[-(2 * d + 1)] = radius
            pad = tuple(pad)

            image = F.pad(
                image,
                pad,
                mode=kwargs.get("mode", "reflect"),
            )

            if spatial_dims == 1:
                image = F.conv1d(image, k, groups=C)
            elif spatial_dims == 2:
                image = F.conv2d(image, k, groups=C)
            elif spatial_dims == 3:
                image = F.conv3d(image, k, groups=C)
            else:
                raise NotImplementedError(
                    f"{spatial_dims}D Gaussian blur not supported"
                )

        # restore layout
        image = image.squeeze(0)
        if last_dim_is_channel:
            image = image.movedim(0, -1)
        else:
            image = image.squeeze(0)

        return image


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
        self.ksize = int(ksize)
        super().__init__(None, **kwargs)

    # ---------- NumPy backend ----------

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return ndimage.median_filter(
            image,
            size=self.ksize,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

    # ---------- Torch backend ----------

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        k = self.ksize
        if k % 2 == 0:
            raise ValueError("MedianBlur requires an odd kernel size.")

        last_dim_is_channel = image.ndim >= 3
        if last_dim_is_channel:
            image = image.movedim(-1, 0)   # C, ...
        else:
            image = image.unsqueeze(0)     # 1, ...

        # add batch dimension
        image = image.unsqueeze(0)         # 1, C, ...

        spatial_dims = image.ndim - 2
        pad = k // 2

        pad_tuple = []
        for _ in range(spatial_dims):
            pad_tuple.extend([pad, pad])
        pad_tuple = tuple(reversed(pad_tuple))

        image = F.pad(
            image,
            pad_tuple,
            mode=kwargs.get("mode", "reflect"),
        )

        if spatial_dims == 1:
            x = image.unfold(2, k, 1)
        elif spatial_dims == 2:
            x = image.unfold(2, k, 1).unfold(3, k, 1)
        elif spatial_dims == 3:
            x = (
                image
                .unfold(2, k, 1)
                .unfold(3, k, 1)
                .unfold(4, k, 1)
            )
        else:
            raise NotImplementedError(
                f"{spatial_dims}D median blur not supported"
            )

        x = x.contiguous().view(*x.shape[:-spatial_dims], -1)
        x = x.median(dim=-1).values

        x = x.squeeze(0)
        if last_dim_is_channel:
            x = x.movedim(0, -1)
        else:
            x = x.squeeze(0)

        return x

#TODO ***CM*** revise typing, docstring, unit test
class Pool(Feature):
    """Abstract base class for pooling features."""


    def __init__(
        self,
        ksize: PropertyLike[int] = 2,
        **kwargs: Any,
    ):
        self.ksize = int(ksize)
        super().__init__(**kwargs)

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


    # ---------- shared helpers ----------

    def _get_pool_size(self, array) -> tuple[int, int, int]:
        k = self.ksize

        if array.ndim == 2:
            return k, k, 1

        if array.ndim == 3:
            if array.shape[-1] <= 4:   # channel heuristic
                return k, k, 1
            return k, k, k

        if array.ndim == 4:
            return k, k, k

        raise ValueError(f"Unsupported array shape {array.shape}")

    def _crop_center(self, array):
        px, py, pz = self._get_pool_size(array)

        # 2D or effectively 2D (channels-last)
        if array.ndim < 3 or pz == 1:
            H, W = array.shape[:2]
            crop_h = (H // px) * px
            crop_w = (W // py) * py
            return array[:crop_h, :crop_w, ...]

        # 3D volume
        Z, H, W = array.shape[:3]
        crop_z = (Z // pz) * pz
        crop_h = (H // px) * px
        crop_w = (W // py) * py
        return array[:crop_z, :crop_h, :crop_w, ...]

    # ---------- abstract backends ----------

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
            block_size = (pz, px, py) + (1,) * (image.ndim - 3)

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

        is_3d = image.ndim >= 3 and pz > 1

        # Flatten extra (channel / feature) dimensions into C
        if not is_3d:
            extra = image.shape[2:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(1, C, image.shape[0], image.shape[1])
            kernel = (px, py)
            stride = (px, py)
            pooled = F.avg_pool2d(x, kernel, stride)
        else:
            extra = image.shape[3:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(
                1, C,
                image.shape[0],
                image.shape[1],
                image.shape[2],
            )
            kernel = (pz, px, py)
            stride = (pz, px, py)
            pooled = F.avg_pool3d(x, kernel, stride)

        # Restore original layout
        return pooled.reshape(pooled.shape[2:] + extra)


class MaxPooling(Pool):
    """Max pooling feature.

    Downsamples the input by applying max pooling over non-overlapping
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
            block_size = (pz, px, py) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.max,
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

        is_3d = image.ndim >= 3 and pz > 1

        # Flatten extra (channel / feature) dimensions into C
        if not is_3d:
            extra = image.shape[2:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(1, C, image.shape[0], image.shape[1])
            kernel = (px, py)
            stride = (px, py)
            pooled = F.max_pool2d(x, kernel, stride)
        else:
            extra = image.shape[3:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(
                1, C,
                image.shape[0],
                image.shape[1],
                image.shape[2],
            )
            kernel = (pz, px, py)
            stride = (pz, px, py)
            pooled = F.max_pool3d(x, kernel, stride)

        # Restore original layout
        return pooled.reshape(pooled.shape[2:] + extra)


class MinPooling(Pool):
    """Min pooling feature.

    Downsamples the input by applying min pooling over non-overlapping
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
            block_size = (pz, px, py) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.min,
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

        is_3d = image.ndim >= 3 and pz > 1

        # Flatten extra (channel / feature) dimensions into C
        if not is_3d:
            extra = image.shape[2:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(1, C, image.shape[0], image.shape[1])
            kernel = (px, py)
            stride = (px, py)

            # min(x) = -max(-x)
            pooled = -F.max_pool2d(-x, kernel, stride)
        else:
            extra = image.shape[3:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(
                1, C,
                image.shape[0],
                image.shape[1],
                image.shape[2],
            )
            kernel = (pz, px, py)
            stride = (pz, px, py)

            pooled = -F.max_pool3d(-x, kernel, stride)

        # Restore original layout
        return pooled.reshape(pooled.shape[2:] + extra)


class SumPooling(Pool):
    """Sum pooling feature.

    Downsamples the input by applying sum pooling over non-overlapping
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
            block_size = (pz, px, py) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.sum,
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

        is_3d = image.ndim >= 3 and pz > 1

        # Flatten extra (channel / feature) dimensions into C
        if not is_3d:
            extra = image.shape[2:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(1, C, image.shape[0], image.shape[1])
            kernel = (px, py)
            stride = (px, py)
            pooled = F.avg_pool2d(x, kernel, stride) * (px * py)
        else:
            extra = image.shape[3:]
            C = int(np.prod(extra)) if extra else 1
            x = image.reshape(
                1, C,
                image.shape[0],
                image.shape[1],
                image.shape[2],
            )
            kernel = (pz, px, py)
            stride = (pz, px, py)
            pooled = F.avg_pool3d(x, kernel, stride) * (pz * px * py)

        # Restore original layout
        return pooled.reshape(pooled.shape[2:] + extra)


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

    # ---------- NumPy backend ----------

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
            block_size = (pz, px, py) + (1,) * (image.ndim - 3)

        return skimage.measure.block_reduce(
            image,
            block_size=block_size,
            func=np.median,
        )

    # ---------- Torch backend ----------

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:

        if not self._warned:
            warnings.warn(
                "MedianPooling is not differentiable and is expensive on the "
                "Torch backend. Avoid using it inside trainable models.",
                UserWarning,
                stacklevel=2,
            )
            self._warned = True

        image = self._crop_center(image)
        px, py, pz = self._get_pool_size(image)

        is_3d = image.ndim >= 3 and pz > 1

        if not is_3d:
            # 2D case (with optional channels)
            extra = image.shape[2:]
            C = int(np.prod(extra)) if extra else 1

            x = image.reshape(1, C, image.shape[0], image.shape[1])

            # unfold: (B, C, H', W', px, py)
            x_u = (
                x.unfold(2, px, px)
                 .unfold(3, py, py)
            )

            x_u = x_u.contiguous().view(
                1, C,
                x_u.shape[2],
                x_u.shape[3],
                -1,
            )

            pooled = x_u.median(dim=-1).values

        else:
            # 3D case (with optional channels)
            extra = image.shape[3:]
            C = int(np.prod(extra)) if extra else 1

            x = image.reshape(
                1, C,
                image.shape[0],
                image.shape[1],
                image.shape[2],
            )

            # unfold: (B, C, Z', Y', X', pz, px, py)
            x_u = (
                x.unfold(2, pz, pz)
                 .unfold(3, px, px)
                 .unfold(4, py, py)
            )

            x_u = x_u.contiguous().view(
                1, C,
                x_u.shape[2],
                x_u.shape[3],
                x_u.shape[4],
                -1,
            )

            pooled = x_u.median(dim=-1).values

        return pooled.reshape(pooled.shape[2:] + extra)


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
    >>> input_image = torch.rand(16, 16)         # channels-last
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


    # ---------- NumPy backend (OpenCV) ----------

    def _get_numpy(
        self,
        image: np.ndarray,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> np.ndarray:

        target_w, target_h = dsize

        # Prefer OpenCV if available
        if OPENCV_AVAILABLE:
            import cv2
            return utils.safe_call(
                cv2.resize,
                positional_args=[image, (target_w, target_h)],
                **kwargs,
            )
        if not OPENCV_AVAILABLE and kwargs:
            warnings.warn("OpenCV not available: resize kwargs may be ignored.", UserWarning)

        # Fallback: skimage (always available in DT)
        from skimage.transform import resize as sk_resize

        if image.ndim == 2:
            out_shape = (target_h, target_w)
        else:
            out_shape = (target_h, target_w) + image.shape[2:]

        out = sk_resize(
            image,
            out_shape,
            preserve_range=True,
            anti_aliasing=True,
        )

        return out.astype(image.dtype, copy=False)

    # ---------- Torch backend ----------

    def _get_torch(
        self,
        image: torch.Tensor,
        dsize: tuple[int, int],
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        target_w, target_h = dsize

        original_ndim = image.ndim
        has_channels = image.ndim >= 3 and image.shape[-1] <= 4

        # Convert to (N, C, H, W)
        if image.ndim == 2:
            x = image.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)

        elif image.ndim == 3 and has_channels:
            x = image.permute(2, 0, 1).unsqueeze(0)      # (1, C, H, W)

        elif image.ndim == 3:
            x = image.unsqueeze(1)                       # (Z, 1, H, W)

        elif image.ndim == 4 and has_channels:
            x = image.permute(0, 3, 1, 2)                # (Z, C, H, W)

        else:
            raise ValueError(
                f"Unsupported tensor shape {image.shape} for Resize."
            )

        # Resize spatial dimensions
        x = F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Restore original layout
        if original_ndim == 2:
            return x.squeeze(0).squeeze(0)

        if original_ndim == 3 and has_channels:
            return x.squeeze(0).permute(1, 2, 0)

        if original_ndim == 3:
            return x.squeeze(1)

        if original_ndim == 4:
            return x.permute(0, 2, 3, 1)

        raise RuntimeError("Unexpected shape restoration path.")


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
        super().__init__(**kwargs)

    def get(
        self: BlurCV2,
        image: np.ndarray,
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

        if apc.is_torch_array(image):
            raise TypeError(
                "BlurCV2 only supports NumPy arrays. "
                "Use GaussianBlur / AverageBlur for Torch."
            )

        import cv2

        filter_fn = getattr(cv2, self.filter) if isinstance(self.filter, str) else self.filter

        try:
            border_attr = self._MODE_TO_BORDER[self.mode]
        except KeyError as e:
            raise ValueError(f"Unsupported border mode '{self.mode}'") from e

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


def isotropic_dilation(
    mask: np.ndarray | torch.Tensor,
    radius: float,
    *,
    backend: Literal["numpy", "torch"],
    device=None,
    dtype=None,
) -> np.ndarray | torch.Tensor:
    """
    Binary dilation using an isotropic (NumPy) or box-shaped (Torch) kernel.

    Notes
    -----
    - NumPy backend uses a true Euclidean ball.
    - Torch backend uses a cubic structuring element (approximate).
    - Torch backend supports 3D masks only.
    - Operation is non-differentiable.

    """
    
    if radius <= 0:
        return mask

    if backend == "numpy":
        from skimage.morphology import isotropic_dilation
        return isotropic_dilation(mask, radius)

    # torch backend
    import torch

    r = int(np.ceil(radius))
    kernel = torch.ones(
        (1, 1, 2 * r + 1, 2 * r + 1, 2 * r + 1),
        device=device or mask.device,
        dtype=dtype or torch.float32,
    )

    x = mask.to(dtype=kernel.dtype)[None, None]
    y = torch.nn.functional.conv3d(
        x,
        kernel,
        padding=r,
    )

    return (y[0, 0] > 0)


def isotropic_erosion(
    mask: np.ndarray | torch.Tensor,
    radius: float,
    *,
    backend: Literal["numpy", "torch"],
    device=None,
    dtype=None,
) -> np.ndarray | torch.Tensor:
    """ 
    Binary erosion using an isotropic (NumPy) or box-shaped (Torch) kernel.
    
    Notes
    -----
    - NumPy backend uses a true Euclidean ball.
    - Torch backend uses a cubic structuring element (approximate).
    - Torch backend supports 3D masks only.
    - Operation is non-differentiable.

    """

    if radius <= 0:
        return mask

    if backend == "numpy":
        from skimage.morphology import isotropic_erosion
        return isotropic_erosion(mask, radius)

    import torch

    r = int(np.ceil(radius))
    kernel = torch.ones(
        (1, 1, 2 * r + 1, 2 * r + 1, 2 * r + 1),
        device=device or mask.device,
        dtype=dtype or torch.float32,
    )

    x = mask.to(dtype=kernel.dtype)[None, None]
    y = torch.nn.functional.conv3d(
        x,
        kernel,
        padding=r,
    )

    required = kernel.numel()
    return (y[0, 0] >= required)