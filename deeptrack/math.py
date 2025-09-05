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

from typing import Any, Callable, TYPE_CHECKING

import array_api_compat as apc
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
import skimage
import skimage.measure

from deeptrack import utils, OPENCV_AVAILABLE, TORCH_AVAILABLE
from deeptrack.features import Feature
from deeptrack.image import Image, strip
from deeptrack.types import ArrayLike, PropertyLike
from deeptrack.backend import xp

if TORCH_AVAILABLE:
    import torch

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
        images: list[NDArray[Any] | torch.Tensor | Image],
        axis: int | tuple[int],
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:
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
        image: NDArray[Any] | torch.Tensor | Image,
        min: float,
        max: float,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:
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
        Whether to normalize each feature independently. It default to `True`,
        which is the only behavior currently implemented.

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

    #TODO ___??___ Implement the `featurewise=False` option

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
        image: ArrayLike,
        min: float,
        max: float,
        **kwargs: Any,
    ) -> ArrayLike:
        """Normalize the input to fall between `min` and `max`.

        Parameters
        ----------
        image: array
            Input image to normalize.
        min: float
            Lower bound of the output range.
        max: float
            Upper bound of the output range.

        Returns
        -------
        array
            Min-max normalized image.

        """

        ptp = xp.max(image) - xp.min(image)
        image = image / ptp * (max - min)
        image = image - xp.min(image) + min

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

    #TODO ___??___ Implement the `featurewise=False` option

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
        image: NDArray[Any] | torch.Tensor | Image,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:
        """Normalizes the input image to have mean 0 and standard deviation 1.

        This method normalizes the input image to have mean 0 and standard
        deviation 1.

        Parameters
        ----------
        image: array
            The input image to normalize.

        Returns
        -------
        array
            The normalized image.

        """

        if apc.is_torch_array(image):
            # By default, torch.std() is unbiased, i.e., divides by N-1
            return (
                (image - torch.mean(image)) / torch.std(image, unbiased=False)
            )

        return (image - xp.mean(image)) / xp.std(image)


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
        image: NDArray[Any] | torch.Tensor | Image,
        quantiles: tuple[float, float] = None,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor | Image:
        """Normalize the input image based on the specified quantiles.

        This method normalizes the input image based on the specified
        quantiles.

        Parameters
        ----------
        image: array
            The input image to normalize.
        quantiles: tuple[float, float]
            Quantile range to calculate scaling factor.

        Returns
        -------
        array
            The normalized image.

        """

        if apc.is_torch_array(image):
            q_tensor = torch.tensor(
                [*quantiles, 0.5],
                device=image.device,
                dtype=image.dtype,
            )
            q_low, q_high, median = torch.quantile(
                image, q_tensor, dim=None, keepdim=False,
            )
        else:  # NumPy
            q_low, q_high, median = xp.quantile(image, (*quantiles, 0.5))

        return (image - median) / (q_high - q_low) * 2.0


#TODO ***JH*** revise Blur - torch, typing, docstring, unit test
class Blur(Feature):
    """Apply a blurring filter to an image.

    This class applies a blurring filter to an image. The filter function
    must be a function that takes an input image and returns a blurred
    image.

    Parameters
    ----------
    filter_function: Callable
        The blurring function to apply. This function must accept the input
        image as a keyword argument named `input`. If using OpenCV functions
        (e.g., `cv2.GaussianBlur`), use `BlurCV2` instead.
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
    >>> from scipy.ndimage import convolve

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a Gaussian kernel for blurring:
    >>> gaussian_kernel = np.array([
    ...     [1,  4,  6,  4, 1],
    ...     [4, 16, 24, 16, 4],
    ...     [6, 24, 36, 24, 6],
    ...     [4, 16, 24, 16, 4],
    ...     [1,  4,  6,  4, 1]
    ... ], dtype=float)
    >>> gaussian_kernel /= np.sum(gaussian_kernel)


    Define a blur function using the Gaussian kernel:
    >>> def gaussian_blur(input, **kwargs):
    ...     return convolve(input, gaussian_kernel, mode='reflect')

    Define a blur feature using the Gaussian blur function:
    >>> blur = dt.Blur(filter_function=gaussian_blur)
    >>> output_image = blur(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.
    The filter_function must accept the input image as a keyword argument named
    input. This is required because it is called via utils.safe_call. If you
    are using functions that do not support input=... (such as OpenCV filters
    like cv2.GaussianBlur), consider using BlurCV2 instead.

    """

    def __init__(
        self: Blur,
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
        super().__init__(borderType=mode, **kwargs)

    def get(self: Blur, image: np.ndarray | Image, **kwargs: Any) -> np.ndarray:
        """Applies the blurring filter to the input image.

        This method applies the blurring filter to the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to blur.
        **kwargs: dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The blurred image.

        """

        kwargs.pop("input", False)
        return utils.safe_call(self.filter, input=image, **kwargs)


#TODO ***JH*** revise AverageBlur - torch, typing, docstring, unit test
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
    `get(image: np.ndarray | Image, ksize: int, **kwargs: Any) --> np.ndarray`
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

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.

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
    ) -> np.ndarray:
        F = xp.nn.functional

        last_dim_is_channel = len(ksize) < input.ndim
        if last_dim_is_channel:
            # permute to first dim
            input = input.movedim(-1, 0)
        else:
            input = input.unsqueeze(0)

        # add batch dimension
        input = input.unsqueeze(0)

        # pad input
        input = F.pad(
            input,
            (ksize[0] // 2, ksize[0] // 2, ksize[1] // 2, ksize[1] // 2),
            mode=kwargs.get("mode", "reflect"),
            value=kwargs.get("cval", 0),
        )
        if input.ndim == 3:
            x = F.avg_pool1d(
                input,
                kernel_size=ksize,
                stride=1,
                padding=0,
                ceil_mode=False,
                count_include_pad=False,
            )
        elif input.ndim == 4:
            x = F.avg_pool2d(
                input,
                kernel_size=ksize,
                stride=1,
                padding=0,
                ceil_mode=False,
                count_include_pad=False,
            )
        elif input.ndim == 5:
            x = F.avg_pool3d(
                input,
                kernel_size=ksize,
                stride=1,
                padding=0,
                ceil_mode=False,
                count_include_pad=False,
            )
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
        input: ArrayLike,
        ksize: int,
        **kwargs: Any,
    ) -> np.ndarray:
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


#TODO ***JH*** revise GaussianBlur - torch, typing, docstring, unit test
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

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.

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

        super().__init__(ndimage.gaussian_filter, sigma=sigma, **kwargs)


#TODO ***JH*** revise MedianBlur - torch, typing, docstring, unit test
class MedianBlur(Blur):
    """Applies a median blur.

    This class replaces each pixel of the input image with the median value of
    its neighborhood. The `ksize` parameter determines the size of the
    neighborhood used to calculate the median filter. The median filter is
    useful for reducing noise while preserving edges. It is particularly
    effective for removing salt-and-pepper noise from images.

    Parameters
    ----------
    ksize: int
        Kernel size.
    **kwargs: dict
        Additional parameters sent to the blurring function.

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

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.

    """

    def __init__(
        self: MedianBlur,
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
        """Initialize the parameters for median blurring.

        This constructor initializes the parameters for median blurring.

        Parameters
        ----------
        ksize: int
            Kernel size.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(ndimage.median_filter, size=ksize, **kwargs)


#TODO ***AL*** revise Pool - torch, typing, docstring, unit test
class Pool(Feature):
    """Downsamples the image by applying a function to local regions of the
    image.

    This class reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the specified pooling
    function to each block. The result is a downsampled image where each pixel
    value represents the result of the pooling function applied to the
    corresponding block.

    Parameters
    ----------
    pooling_function: function
        A function that is applied to each local region of the image.
        DOES NOT NEED TO BE WRAPPED IN ANOTHER FUNCTION.
        The `pooling_function` must accept the input image as a keyword argument
        named `input`, as it is called via `utils.safe_call`.
        Examples include `np.mean`, `np.max`, `np.min`, etc.
    ksize: int
        Size of the pooling kernel.
    **kwargs: Any
        Additional parameters sent to the pooling function.

    Methods
    -------
    `get(image: np.ndarray | Image, ksize: int, **kwargs: Any) --> np.ndarray`
        Applies the pooling function to the input image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a pooling feature:
    >>> pooling_feature = dt.Pool(pooling_function=np.mean, ksize=4)
    >>> output_image = pooling_feature.get(input_image, ksize=4)
    >>> print(output_image.shape)
    (8, 8)

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.
    The filter_function must accept the input image as a keyword argument named
    input. This is required because it is called via utils.safe_call. If you
    are using functions that do not support input=... (such as OpenCV filters
    like cv2.GaussianBlur), consider using BlurCV2 instead.

    """

    def __init__(
        self: Pool,
        pooling_function: Callable,
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
        """Initialize the parameters for pooling input features.

        This constructor initializes the parameters for pooling input
        features.

        Parameters
        ----------
        pooling_function: Callable
            The pooling function to apply.
        ksize: int
            Size of the pooling kernel.
        **kwargs: Any
            Additional keyword arguments.

        """

        self.pooling = pooling_function
        super().__init__(ksize=ksize, **kwargs)

    def get(
        self: Pool,
        image: np.ndarray | Image,
        ksize: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies the pooling function to the input image.

        This method applies the pooling function to the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to pool.
        ksize: int
            Size of the pooling kernel.
        **kwargs: dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The pooled image.

        """

        kwargs.pop("func", False)
        kwargs.pop("image", False)
        kwargs.pop("block_size", False)
        return utils.safe_call(
            skimage.measure.block_reduce,
            image=image,
            func=self.pooling,
            block_size=ksize,
            **kwargs,
        )


#TODO ***AL*** revise AveragePooling - torch, typing, docstring, unit test
class AveragePooling(Pool):
    """Apply average pooling to an image.

    This class reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the average function to
    each block. The result is a downsampled image where each pixel value
    represents the average value within the corresponding block of the
    original image.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    **kwargs: dict
        Additional parameters sent to the pooling function.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define an average pooling feature:
    >>> average_pooling = dt.AveragePooling(ksize=4)
    >>> output_image = average_pooling(input_image)
    >>> print(output_image.shape)
    (8, 8)

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.

    """

    def __init__(
        self: Pool,
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
        """Initialize the parameters for average pooling.

        This constructor initializes the parameters for average pooling.

        Parameters
        ----------
        ksize: int
            Size of the pooling kernel.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(np.mean, ksize=ksize, **kwargs)


class MaxPooling(Pool):
    """Apply max-pooling to images.

    `MaxPooling` reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the `max` function
    to each block. The result is a downsampled image where each pixel value
    represents the maximum value within the corresponding block of the
    original image. This is useful for reducing the size of an image while
    retaining the most significant features.

    If the backend is NumPy, the downsampling is performed using
    `skimage.measure.block_reduce`.
    If the backend is PyTorch, the downsampling
    is performed using `torch.nn.functional.max_pool2d`.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    **kwargs: Any
        Additional parameters sent to the pooling function.

    Examples
    --------
    >>> import deeptrack as dt
    Create an input image:
    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(32, 32)

    Define and use a max pooling feature:

    >>> max_pooling = dt.MaxPooling(ksize=8)
    >>> output_image = max_pooling(input_image)
    >>> output_image.shape
    (4, 4)

    """

    def __init__(
        self: MaxPooling,
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
        """Initialize the parameters for max-pooling.

        This constructor initializes the parameters for max-pooling.

        Parameters
        ----------
        ksize: int
            Size of the pooling kernel.
        **kwargs: Any
            Additional keyword arguments.

        """
        
        super().__init__(np.max, ksize=ksize, **kwargs)

    
    def get(
        self: MaxPooling,
        image: NDArray[Any] | torch.Tensor,
        ksize: int=3,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor:
        """Max pooling of input.

        Checks the current backend and chooses the appropriate function to pool
        the input image, either `._get_torch()` or `._get_numpy()`.

        Parameters
        ----------
        image: array or tensor
            Input array or tensor be pooled.
        ksize: int
            Kernel size of the pooling operation.

        Returns
        -------
        array or tensor
            The pooled input as `NDArray` or `torch.Tensor` depending on
            the backend.

        """
        if self.get_backend() == "numpy":
            return self._get_numpy(image, ksize, **kwargs)
        elif self.get_backend() == "torch":
            return self._get_torch(image, ksize, **kwargs)
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported")


    def _get_numpy(
        self: MaxPooling,
        image: NDArray[Any],
        ksize: int=3,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Max pooling pooling with the NumPy backend enabled.

        Returns the result of the input array passed to the scikit
        image `block_reduce()` function with `np.max()` as the pooling function.

        Parameters
        ----------
        image: NDArray
            Input array to be pooled.
        ksize: int
            Kernel size of the pooling operation.

        Returns
        -------
        NDArray
            The pooled image as a `NDArray`.
            
        """
        return utils.safe_call(
            skimage.measure.block_reduce,
            image=image,
            func=np.max,
            block_size=ksize,
            **kwargs,
        )

    def _get_torch(
        self: MaxPooling,
        image: torch.Tensor,
        ksize: int=3,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Perform max pooling with the PyTorch backend enabled.


        Returns the result of the tensor passed to a PyTorch max
        pooling layer.

        Parameters
        ----------
        image: torch.Tensor
            Input tensor to be pooled.
        ksize: int
            Kernel size of the pooling operation.

        Returns
        -------
        torch.Tensor
            The pooled image as a `torch.Tensor`.

        """

        # If input tensor is 2D
        if len(image.shape) == 2:
            # Add batch dimension for max pooling.
            expanded_image = image.unsqueeze(0)

            pooled_image = torch.nn.functional.max_pool2d(
                expanded_image, kernel_size=ksize,
            )
            # Remove the expanded dim.
            return pooled_image.squeeze(0)

        return torch.nn.functional.max_pool2d(
            image,
            kernel_size=ksize,
        )


class MinPooling(Pool):
    """Apply min-pooling to images.

    `MinPooling` reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the `min` function to
    each block. The result is a downsampled image where each pixel value
    represents the minimum value within the corresponding block of the original
    image.

    If the backend is NumPy, the downsampling is performed using 
    `skimage.measure.block_reduce`.

    If the backend is PyTorch, the downsampling is performed using the inverse
    of `torch.nn.functional.max_pool2d` by changing the sign of the input.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    **kwargs: Any
        Additional parameters sent to the pooling function.

    Examples
    --------
    >>> import deeptrack as dt

    Create an input image:
    >>> import numpy as np
    >>>
    >>> input_image = np.random.rand(32, 32)

    Define and use a min-pooling feature:
    >>> min_pooling = dt.MinPooling(ksize=4)
    >>> output_image = min_pooling(input_image)
    >>> output_image.shape
    (8, 8)

    """

    def __init__(
        self: MinPooling,
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
        """Initialize the parameters for min-pooling.

        This constructor initializes the parameters for min-pooling and checks
        whether to use the NumPy or PyTorch implementation, defaults to NumPy.

        Parameters
        ----------
        ksize: int
            Size of the pooling kernel.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(np.min, ksize=ksize, **kwargs)

    def get(
        self: MinPooling,
        image: NDArray[Any] | torch.Tensor,
        ksize: int=3,
        **kwargs: Any,
    ) -> NDArray[Any] | torch.Tensor:
        """Min pooling of input.

        Checks the current backend and chooses the appropriate function to pool
        the input image, either `._get_torch()` or `._get_numpy()`.

        Parameters
        ----------
        image: array or tensor
            Input array or tensor to be pooled.
        ksize: int
            Kernel size of the pooling operation.

        Returns
        -------
        array or tensor
            The pooled image as `NDArray` or `torch.Tensor` depending on the
            backend.

        """

        if self.get_backend() == "numpy":
            return self._get_numpy(image, ksize, **kwargs)

        if self.get_backend() == "torch":
            return self._get_torch(image, ksize, **kwargs)

        raise NotImplementedError(f"Backend {self.backend} not supported")

    def _get_numpy(
        self: MinPooling,
        image: NDArray[Any],
        ksize: int=3,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Min-pooling with the NumPy backend.

        Returns the result of the input array passed to the scikit
        `image block_reduce()` function with `np.min()` as the pooling
        function.

        Parameters
        ----------
        image: NDArray
            Input image to be pooled.
        ksize: int
            Kernel size of the pooling operation.

        Returns
        -------
        NDArray
            The pooled image as a `NDArray`.

        """

        return utils.safe_call(
            skimage.measure.block_reduce,
            image=image,
            func=np.min,
            block_size=ksize,
            **kwargs,
        )

    def _get_torch(
        self: MinPooling,
        image: torch.Tensor,
        ksize: int=3,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Min-pooling with the PyTorch backend.

        As PyTorch does not have a min-pooling layer, the equivalent operation
        is to first multiply the input tensor with `-1`, then perform
        max-pooling, and finally multiply the max pooled tensor with `-1`.

        Parameters
        ----------
        image: torch.Tensor
            Input tensor to be pooled.
        ksize: int
            Kernel size of the pooling operation.

        Returns
        -------
        torch.Tensor
            The pooled image as a `torch.Tensor`.

        """

        # If input tensor is 2D
        if len(image.shape) == 2:
            # Add batch dimension for min-pooling
            expanded_image = image.unsqueeze(0)

            pooled_image = - torch.nn.functional.max_pool2d(
                expanded_image * (-1),
                kernel_size=ksize,
            )

            # Remove the expanded dim
            return pooled_image.squeeze(0)

        return -torch.nn.functional.max_pool2d(
            image * (-1),
            kernel_size=ksize,
        )


#TODO ***AL*** revise MedianPooling - torch, typing, docstring, unit test
class MedianPooling(Pool):
    """Apply median pooling to images.

    This class reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the median function to
    each block. The result is a downsampled image where each pixel value
    represents the median value within the corresponding block of the
    original image. This is useful for reducing the size of an image while
    retaining the most significant features.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    **kwargs: Any
        Additional parameters sent to the pooling function.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a median pooling feature:
    >>> median_pooling = dt.MedianPooling(ksize=3)
    >>> output_image = median_pooling(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Visualize the input and output images:
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.imshow(input_image, cmap='gray')
    >>> plt.subplot(1, 2, 2)
    >>> plt.imshow(output_image, cmap='gray')
    >>> plt.show()

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.

    """

    def __init__(
        self: MedianPooling,
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
        """Initialize the parameters for median pooling.

        This constructor initializes the parameters for median pooling.

        Parameters
        ----------
        ksize: int
            Size of the pooling kernel.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(np.median, ksize=ksize, **kwargs)


#TODO ***MG*** revise Resize - torch, typing, docstring, unit test
class Resize(Feature):
    """Resize an image to a specified size.

    This class is a wrapper around cv2.resize and resizes an image to a
    specified size. The `dsize` parameter specifies the desired output size of
    the image.
    Note that the order of the axes is different in cv2 and numpy. In cv2, the
    first axis is the vertical axis, while in numpy it is the horizontal axis.
    This is reflected in the default values of the arguments.

    Parameters
    ----------
    dsize: tuple
        Size to resize to.
    **kwargs: Any
        Additional parameters sent to the resizing function.

    """

    def __init__(
        self: Resize,
        dsize: PropertyLike[tuple] = (256, 256),
        **kwargs: Any,
    ):
        """Initialize the parameters for resizing input features.

        This constructor initializes the parameters for resizing input
        features.

        Parameters
        ----------
        dsize: tuple
            Size to resize to.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(dsize=dsize, **kwargs)

    def get(self: Resize, image: np.ndarray, dsize: tuple, **kwargs: Any) -> np.ndarray:
        """Resize the input image to the specified size.

        This method resizes the input image to the specified size.

        Parameters
        ----------
        image: np.ndarray
            The input image to resize.
        dsize: tuple
            Desired output size of the image.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The resized image.

        """

        import cv2
        from deeptrack import config

        if self._wrap_array_with_image:
            image = strip(image)

        return utils.safe_call(cv2.resize, positional_args=[image, dsize], **kwargs)


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
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.

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
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.

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
