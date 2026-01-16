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
from numpy.typing import NDArray #TODO TBE
from scipy import ndimage
import skimage
import skimage.measure

from deeptrack import utils, OPENCV_AVAILABLE, TORCH_AVAILABLE
from deeptrack.features import Feature
from deeptrack.image import Image, strip #TODO TBE
from deeptrack.types import PropertyLike
from deeptrack.backend import xp

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
    "MedianPooling",
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
        min: PropertyLike[float] = -np.inf,
        max: PropertyLike[float] = +np.inf,
        **kwargs: Any,
    ):
        """Initialize the clipping range.

        Parameters
        ----------
        min: float, optional
            Minimum allowed value. It defaults to `-np.inf`.
        max: float, optional
            Maximum allowed value. It defaults to `+np.inf`.
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

        if featurewise:
            # Normalize per feature (last axis)
            axis = tuple(range(image.ndim - 1))

            img_min = xp.min(image, axis=axis, keepdims=True)
            img_max = xp.max(image, axis=axis, keepdims=True)
        else:
            # Normalize globally
            img_min = xp.min(image)
            img_max = xp.max(image)

        ptp = img_max - img_min

        # Avoid division by zero
        image = (image - img_min) / ptp * (max - min) + min

        try:
            image[xp.isnan(image)] = 0
        except TypeError:
            pass

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

        if featurewise:
            # Normalize per feature (last axis)
            axis = tuple(range(image.ndim - 1))

            mean = xp.mean(image, axis=axis, keepdims=True)

            if apc.is_torch_array(image):
                std = torch.std(image, dim=axis, keepdim=True, unbiased=False)
            else:
                std = xp.std(image, axis=axis)
        else:
            # Normalize globally
            mean = xp.mean(image)

            if apc.is_torch_array(image):
                std = torch.std(image, unbiased=False)
            else:
                std = xp.std(image)

        image = (image - mean) / std

        try:
            image[xp.isnan(image)] = 0
        except TypeError:
            pass

        return image


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

    #TODO ___??___ Implement the `featurewise=False` option

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
        self: NormalizeQuantile,
        image: np.ndarray | torch.Tensor,
        quantiles: tuple[float, float],
        featurewise: bool,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
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

        if featurewise:
            # Per-feature normalization (last axis)
            axis = tuple(range(image.ndim - 1))

            if apc.is_torch_array(image):
                q = torch.tensor(
                    [q_low_val, q_high_val, 0.5],
                    device=image.device,
                    dtype=image.dtype,
                )
                q_low, q_high, median = torch.quantile(
                    image, q, dim=axis, keepdim=True
                )
            else:
                q_low, q_high, median = xp.quantile(
                    image, (q_low_val, q_high_val, 0.5),
                    axis=axis,
                    keepdims=True,
                )
        else:
            # Global normalization
            if apc.is_torch_array(image):
                q = torch.tensor(
                    [q_low_val, q_high_val, 0.5],
                    device=image.device,
                    dtype=image.dtype,
                )
                q_low, q_high, median = torch.quantile(
                    image, q, dim=None, keepdim=False
                )
            else:
                q_low, q_high, median = xp.quantile(
                    image, (q_low_val, q_high_val, 0.5)
                )

        image = (image - median) / (q_high - q_low) * 2.0

        try:
            image[xp.isnan(image)] = 0
        except TypeError:
            pass

        return image



#TODO ***CM*** revise typing, docstring, unit test
class Blur(Feature):
    """Apply a blurring filter to an image.

    This class acts as a backend-dispatching blur operator. Subclasses must
    implement backend-specific logic via `_get_numpy` and optionally
    `_get_torch`.

    Notes
    -----
    - NumPy execution is always supported.
    - Torch execution is only supported if `_get_torch` is implemented.
    - Generic `filter_function`-based blurs are NumPy-only by design.

    """

    def __init__(
        self,
        filter_function: Callable | None = None,
        mode: PropertyLike[str] = "reflect",
        **kwargs: Any,
    ):
        """Initialize the blur feature.

        Parameters
        ----------
        filter_function : Callable or None
            NumPy-based blurring function. Must accept the input image as a
            keyword argument named `input`. If `None`, the subclass must
            implement `_get_numpy`.
        mode : str
            Border mode for NumPy-based filters.
        **kwargs : Any
            Additional keyword arguments passed to Feature.
        """
        self.filter = filter_function
        self.mode = mode
        super().__init__(**kwargs)

    def __call__(
        self,
        image: np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(image, np.ndarray):
            return self._get_numpy(image, **kwargs)

        if TORCH_AVAILABLE and isinstance(image, torch.Tensor):
            return self._get_torch(image, **kwargs)

        raise TypeError(
            "Blur only supports numpy.ndarray or torch.Tensor inputs."
        )

    def _get_numpy(
        self,
        image: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.filter is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement a NumPy backend."
            )

        # Avoid passing conflicting keywords
        kwargs = dict(kwargs)
        kwargs.pop("input", None)

        return utils.safe_call(
            self.filter,
            input=image,
            mode=self.mode,
            **kwargs,
        )

    def _get_torch(
        self,
        image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise TypeError(
            f"{self.__class__.__name__} does not support torch.Tensor inputs. "
            "Use a Torch-enabled blur (e.g. AverageBlur or a V2 blur class)."
        )



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
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
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

        super().__init__(None, ksize=ksize, **kwargs)

    def _kernel_shape(self, shape: tuple[int, ...], ksize: int) -> tuple[int, ...]:
        if shape[-1] < ksize:
            return (ksize,) * (len(shape) - 1) + (1,)
        return (ksize,) * len(shape)

    def _get_numpy(
        self, input: np.ndarray, ksize: tuple[int, ...], **kwargs: Any
    ) -> np.ndarray:
        return ndimage.uniform_filter(
            input,
            size=ksize,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
            origin=kwargs.get("origin", 0),
            axes=tuple(range(0, len(ksize))),
        )

    def _get_torch(
        self, input: torch.Tensor, ksize: tuple[int, ...], **kwargs: Any
    ) -> torch.Tensor:

        last_dim_is_channel = len(ksize) < input.ndim
        if last_dim_is_channel:
            input = input.movedim(-1, 0)
        else:
            input = input.unsqueeze(0)

        # add batch dimension
        input = input.unsqueeze(0)

        # dynamic padding
        pad = []
        for k in reversed(ksize):
            p = k // 2
            pad.extend([p, p])
        pad = tuple(pad)

        input = F.pad(
            input,
            pad,
            mode=kwargs.get("mode", "reflect"),
            value=kwargs.get("cval", 0),
        )

        if input.ndim == 3:
            x = F.avg_pool1d(input, kernel_size=ksize, stride=1)
        elif input.ndim == 4:
            x = F.avg_pool2d(input, kernel_size=ksize, stride=1)
        elif input.ndim == 5:
            x = F.avg_pool3d(input, kernel_size=ksize, stride=1)
        else:
            raise NotImplementedError(
                f"Input dimension {input.ndim - 2} not supported for torch backend"
            )

        # restore layout
        x = x.squeeze(0)
        if last_dim_is_channel:
            x = x.movedim(0, -1)
        else:
            x = x.squeeze(0)

        return x

    def get(
        self: AverageBlur,
        input: np.ndarray | torch.Tensor,
        ksize: int,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Applies the average blurring filter to the input image.

        This method applies the average blurring filter to the input image.

        Parameters
        ----------
        input: np.ndarray
            The input image to blur.
        ksize: int
            Kernel size for the pooling operation.
        **kwargs: dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The blurred image.

        """

        k = self._kernel_shape(input.shape, ksize)

        if self.backend == "numpy":
            return self._get_numpy(input, k, **kwargs)
        elif self.backend == "torch":
            return self._get_torch(input, k, **kwargs)
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported")


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

    def _get_numpy(
        self,
        input: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return ndimage.gaussian_filter(
            input,
            sigma=self.sigma,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

    def _gaussian_kernel_1d(
        self,
        sigma: float,
        device,
        dtype,
    ) -> torch.Tensor:
        radius = int(np.ceil(3 * sigma))
        x = torch.arange(
            -radius, radius + 1,
            device=device,
            dtype=dtype,
        )
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def _get_torch(
        self,
        input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        sigma = self.sigma
        kernel_1d = self._gaussian_kernel_1d(
            sigma,
            device=input.device,
            dtype=input.dtype,
        )

        last_dim_is_channel = input.ndim >= 3
        if last_dim_is_channel:
            input = input.movedim(-1, 0)  # C, ...
        else:
            input = input.unsqueeze(0)    # 1, ...

        # add batch dimension
        input = input.unsqueeze(0)        # 1, C, ...

        spatial_dims = input.ndim - 2
        C = input.shape[1]

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

            input = F.pad(
                input,
                pad,
                mode=kwargs.get("mode", "reflect"),
            )

            if spatial_dims == 1:
                input = F.conv1d(input, k, groups=C)
            elif spatial_dims == 2:
                input = F.conv2d(input, k, groups=C)
            elif spatial_dims == 3:
                input = F.conv3d(input, k, groups=C)
            else:
                raise NotImplementedError(
                    f"{spatial_dims}D Gaussian blur not supported"
                )

        # restore layout
        input = input.squeeze(0)
        if last_dim_is_channel:
            input = input.movedim(0, -1)
        else:
            input = input.squeeze(0)

        return input


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

    def _get_numpy(
        self,
        input: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return ndimage.median_filter(
            input,
            size=self.ksize,
            mode=kwargs.get("mode", "reflect"),
            cval=kwargs.get("cval", 0),
        )

    def _get_torch(
        self,
        input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        k = self.ksize
        if k % 2 == 0:
            raise ValueError("MedianBlur requires an odd kernel size.")

        last_dim_is_channel = input.ndim >= 3
        if last_dim_is_channel:
            input = input.movedim(-1, 0)   # C, ...
        else:
            input = input.unsqueeze(0)     # 1, ...

        # add batch dimension
        input = input.unsqueeze(0)         # 1, C, ...

        spatial_dims = input.ndim - 2
        pad = k // 2

        # padding
        pad_tuple = []
        for _ in range(spatial_dims):
            pad_tuple.extend([pad, pad])
        pad_tuple = tuple(reversed(pad_tuple))

        input = F.pad(
            input,
            pad_tuple,
            mode=kwargs.get("mode", "reflect"),
        )

        # unfold spatial dimensions
        if spatial_dims == 1:
            x = input.unfold(2, k, 1)
        elif spatial_dims == 2:
            x = (
                input
                .unfold(2, k, 1)
                .unfold(3, k, 1)
            )
        elif spatial_dims == 3:
            x = (
                input
                .unfold(2, k, 1)
                .unfold(3, k, 1)
                .unfold(4, k, 1)
            )
        else:
            raise NotImplementedError(
                f"{spatial_dims}D median blur not supported"
            )

        # flatten neighborhood and take median
        x = x.contiguous().view(*x.shape[:-spatial_dims], -1)
        x = x.median(dim=-1).values

        # restore layout
        x = x.squeeze(0)
        if last_dim_is_channel:
            x = x.movedim(0, -1)
        else:
            x = x.squeeze(0)

        return x

#TODO ***CM*** revise typing, docstring, unit test
class Pool:
    """
    DeepTrack v2 replacement for Pool.

    Generic, center-preserving block pooling with NumPy and Torch backends.
    Public API matches v1: a single integer ksize.

    Pool size semantics:
    - 2D input  -> (ksize, ksize, 1)
    - 3D input  -> (ksize, ksize, ksize)
    """

    _TORCH_REDUCERS_2D: Dict[Callable, Callable] = {
        np.mean: lambda x, k, s: F.avg_pool2d(x, k, s),
        np.sum:  lambda x, k, s: F.avg_pool2d(x, k, s) * (k[0] * k[1]),
        np.max:  lambda x, k, s: F.max_pool2d(x, k, s),
        np.min:  lambda x, k, s: -F.max_pool2d(-x, k, s),
    }

    _TORCH_REDUCERS_3D: Dict[Callable, Callable] = {
        np.mean: lambda x, k, s: F.avg_pool3d(x, k, s),
        np.sum:  lambda x, k, s: F.avg_pool3d(x, k, s) * (k[0] * k[1] * k[2]),
        np.max:  lambda x, k, s: F.max_pool3d(x, k, s),
        np.min:  lambda x, k, s: -F.max_pool3d(-x, k, s),
    }

    def __init__(
        self,
        pooling_function: Callable,
        ksize: int = 2,
    ):
        if pooling_function not in (
            np.mean, np.sum, np.min, np.max, np.median
        ):
            raise ValueError(
                "Unsupported pooling_function. "
                "Use one of: np.mean, np.sum, np.min, np.max, np.median."
            )

        if not isinstance(ksize, int) or ksize < 1:
            raise ValueError("ksize must be a positive integer.")

        self.pooling_function = pooling_function
        self.ksize = int(ksize)

    def _get_pool_size(self, array) -> Tuple[int, int, int]:
        """
        Determine pooling kernel size based on semantic dimensionality.

        - 2D images: (Nx, Ny) or (Nx, Ny, C)  -> pool in x,y only
        - 3D volumes: (Nx, Ny, Nz) or (Nx, Ny, Nz, C) -> pool in x,y,z
        - Never pool over channels
        """
        k = self.ksize

        # 2D image
        if array.ndim == 2:
            return k, k, 1

        # 3D array: could be (x, y, z) or (x, y, c)
        if array.ndim == 3:
            # Heuristic: small last dim → channels
            if array.shape[-1] <= 4:
                return k, k, 1
            return k, k, k

        # 4D array: (x, y, z, c)
        if array.ndim == 4:
            return k, k, k

        raise ValueError(
            f"Unsupported array shape {array.shape} for pooling."
        )

    def _crop_center(self, array):
        px, py, pz = self._get_pool_size(array)

        # 2D (or effectively 2D)
        if array.ndim < 3 or pz == 1:
            H, W = array.shape[:2]
            crop_h = (H // px) * px
            crop_w = (W // py) * py
            off_h = (H - crop_h) // 2
            off_w = (W - crop_w) // 2
            return array[
                off_h : off_h + crop_h,
                off_w : off_w + crop_w,
                ...
            ]

        # 3D
        Z, H, W = array.shape[:3]
        crop_z = (Z // pz) * pz
        crop_h = (H // px) * px
        crop_w = (W // py) * py
        off_z = (Z - crop_z) // 2
        off_h = (H - crop_h) // 2
        off_w = (W - crop_w) // 2
        return array[
            off_z : off_z + crop_z,
            off_h : off_h + crop_h,
            off_w : off_w + crop_w,
            ...
        ]

    def _pool_numpy(self, array: np.ndarray) -> np.ndarray:
        array = self._crop_center(array)
        px, py, pz = self._get_pool_size(array)

        if array.ndim < 3 or pz == 1:
            pool_shape = (px, py) + (1,) * (array.ndim - 2)
        else:
            pool_shape = (pz, px, py) + (1,) * (array.ndim - 3)

        return skimage.measure.block_reduce(
            array,
            block_size=pool_shape,
            func=self.pooling_function,
        )

    def _pool_torch(self, array: torch.Tensor) -> torch.Tensor:
        array = self._crop_center(array)
        px, py, pz = self._get_pool_size(array)

        is_3d = array.ndim >= 3 and pz > 1

        if not is_3d:
            extra = array.shape[2:]
            C = int(np.prod(extra)) if extra else 1
            x = array.reshape(1, C, array.shape[0], array.shape[1])
            kernel = (px, py)
            stride = (px, py)
            reducers = self._TORCH_REDUCERS_2D
        else:
            extra = array.shape[3:]
            C = int(np.prod(extra)) if extra else 1
            x = array.reshape(
                1, C, array.shape[0], array.shape[1], array.shape[2]
            )
            kernel = (pz, px, py)
            stride = (pz, px, py)
            reducers = self._TORCH_REDUCERS_3D

        # Median: explicit unfolding
        if self.pooling_function is np.median:
            if is_3d:
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
            else:
                x_u = x.unfold(2, px, px).unfold(3, py, py)
                x_u = x_u.contiguous().view(
                    1, C,
                    x_u.shape[2],
                    x_u.shape[3],
                    -1,
                )
                pooled = x_u.median(dim=-1).values
        else:
            reducer = reducers[self.pooling_function]
            pooled = reducer(x, kernel, stride)

        return pooled.reshape(pooled.shape[2:] + extra)

    def __call__(self, array):
        if isinstance(array, np.ndarray):
            return self._pool_numpy(array)

        if TORCH_AVAILABLE and isinstance(array, torch.Tensor):
            return self._pool_torch(array)

        raise TypeError(
            "Pool only supports np.ndarray or torch.Tensor inputs."
        )


class AveragePooling(Pool):
    def __init__(self, ksize: int = 2):
        super().__init__(np.mean, ksize)


class SumPooling(Pool):
    def __init__(self, ksize: int = 2):
        super().__init__(np.sum, ksize)


class MinPooling(Pool):
    def __init__(self, ksize: int = 2):
        super().__init__(np.min, ksize)


class MaxPooling(Pool):
    def __init__(self, ksize: int = 2):
        super().__init__(np.max, ksize)


class MedianPooling(Pool):
    def __init__(self, ksize: int = 2):
        super().__init__(np.median, ksize)



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
            The target size. Format depends on backend: `(width, height)` for
            NumPy, `(height, width)` for PyTorch. Default is (256, 256).
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

        target_w, target_h = dsize

        # Torch backend
        if apc.is_torch_array(image):
            import torch.nn.functional as F

            original_ndim = image.ndim
            has_channels = (
                image.ndim >= 3 and image.shape[-1] <= 4
            )

            # Bring to (N, C, H, W)
            if image.ndim == 2:
                # (H, W) -> (1, 1, H, W)
                x = image.unsqueeze(0).unsqueeze(0)

            elif image.ndim == 3 and has_channels:
                # (H, W, C) -> (1, C, H, W)
                x = image.permute(2, 0, 1).unsqueeze(0)

            elif image.ndim == 3:
                # (Z, H, W) -> treat Z as batch
                x = image.unsqueeze(1)

            elif image.ndim == 4 and has_channels:
                # (Z, H, W, C) -> (Z, C, H, W)
                x = image.permute(0, 3, 1, 2)

            else:
                raise ValueError(
                    f"Unsupported tensor shape {image.shape} for Resize."
                )

            # Resize spatial dimensions
            resized = F.interpolate(
                x,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

            # Restore original layout
            if original_ndim == 2:
                return resized.squeeze(0).squeeze(0)

            if original_ndim == 3 and has_channels:
                return resized.squeeze(0).permute(1, 2, 0)

            if original_ndim == 3:
                return resized.squeeze(1)

            if original_ndim == 4:
                return resized.permute(0, 2, 3, 1)

            raise RuntimeError("Unexpected shape restoration path.")

        # NumPy / OpenCV backend
        else:
            import cv2

            # OpenCV expects (width, height)
            return utils.safe_call(
                cv2.resize,
                positional_args=[image, (target_w, target_h)],
                **kwargs,
            )


if OPENCV_AVAILABLE:
    _map_mode_to_cv2_borderType = {
        "reflect": cv2.BORDER_REFLECT,
        "wrap": cv2.BORDER_WRAP,
        "constant": cv2.BORDER_CONSTANT,
        "mirror": cv2.BORDER_REFLECT_101,
        "nearest": cv2.BORDER_REPLICATE,
    }


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
    `get(image: np.ndarray | Image, **kwargs: Any) --> np.ndarray`
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

    def __new__(
        cls: type,
        *args: tuple,
        **kwargs: Any,
    ):
        """Ensures that OpenCV (cv2) is available before instantiating the
        class.

        Overrides the default object creation process to check that the `cv2`
        module is available before creating the class. If OpenCV is not
        installed, it raises an ImportError with instructions for installation.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the class constructor.
        **kwargs : dict
            Keyword arguments passed to the class constructor.

        Returns
        -------
        BlurCV2
            An instance of the BlurCV2 feature class.

        Raises
        ------
        ImportError
            If the OpenCV (`cv2`) module is not available in the current
            environment.

        """

        print(cls.__name__)

        if not OPENCV_AVAILABLE:
            raise ImportError(
                "OpenCV not installed on device. Since OpenCV is an optional "
                f"dependency of DeepTrack2. To use {cls.__name__}, "
                "you need to install it manually."
            )

        return super().__new__(cls)

    def __init__(
        self: BlurCV2,
        filter_function: Callable,
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

        self.filter = filter_function
        borderType = _map_mode_to_cv2_borderType[mode]
        super().__init__(borderType=borderType, **kwargs)

    def get(
        self: BlurCV2,
        image: np.ndarray | Image,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies the blurring filter to the input image.

        This method applies the blurring filter to the input image.

        Parameters
        ----------
        image: np.ndarray | Image
            The input image to blur. Can be a NumPy array or DeepTrack Image.
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
                "For Torch tensors, use Blur or GaussianBlur instead."
            )

        kwargs.pop("name", None)
        result = self.filter(src=image, **kwargs)
        return result


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
            cv2.bilateralFilter,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            **kwargs,
        )


def isotropic_dilation(
    mask,
    radius: float,
    *,
    backend: str,
    device=None,
    dtype=None,
):
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
    mask,
    radius: float,
    *,
    backend: str,
    device=None,
    dtype=None,
):
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