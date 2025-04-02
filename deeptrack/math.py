"""Mathematical operations and structures.

This module provides classes and utilities to perform common mathematical 
operations and transformations on images, including clipping, normalization, 
blurring, and pooling. These are implemented as subclasses of `Feature` for 
seamless integration with the feature-based design of the library.

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

- `MaxPooling`: Apply max pooling to the image.

- `MinPooling`: Apply min pooling to the image.

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

from __future__ import annotations
from typing import Callable, Any

import numpy as np
import scipy.ndimage as ndimage
import skimage
import skimage.measure

from deeptrack import utils
from deeptrack.features import Feature
from deeptrack.image import Image, strip
from deeptrack.types import PropertyLike


class Average(Feature):
    """Average of input images.

    This class computes the average of input images along the specified axis. 
    If `features` is not None, it instead resolves all features in the list and 
    averages the result.

    Parameters
    ----------
    axis: int or tuple of ints
        Axis along which to average
    features: list of features, optional

    Attributes
    ----------
    __distributed__: bool
        Determines whether `.get(image, **kwargs)` is applied to each element 
        of the input list independently (`__distributed__ = True`) or to the 
        list as a whole (`__distributed__ = False`).
     
    Methods
    -------
    `get(images: np.ndarray | Image | list[Image], axis: int, **kwargs: Any) --> np.ndarray`
        Computes the average of the input images along the specified axis.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    
    Create two input images:
    >>> input_image1 = np.random.rand(10, 30, 20)
    >>> input_image2 = np.random.rand(10, 30, 20)

    Define a simple pipeline with the average feature:
    >>> average = dt.Average(axis=1)
    >>> output_image = average([input_image1, input_image2])
    >>> print(output_image)
    (2, 30, 20)

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If 
    `store_properties` is set to `True`, the returned array will be 
    automatically wrapped in an `Image` object. This behavior is handled 
    internally and does not affect the return type of the `get()` method.

    """

    __distributed__ = False

    def __init__(
        self: Average,
        features: PropertyLike[list[Feature] | None] = None,
        axis: PropertyLike[int] = 0,
        **kwargs: Any
    ):
        """Initialize the parameters for averaging input features. 
        
        This constructor initializes the parameters for averaging input 
        features.

        Parameters
        ----------
        features: list of Feature or None, optional
            List of features to be resolved and averaged. Defaults to None.
        axis: int or tuple[int]
            Axis along which to compute the average. Defaults to 0.
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
        images: np.ndarray | Image | list[Image],
        axis: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Computes the average of input images along the specified axis.

        This method computes the average of the input images along the 
        specified axis.

        Parameters
        ----------
        images: np.ndarray
            The input images to average.
        axis: int
            The axis along which to average.

        Returns
        -------
        np.ndarray
            The average of the input images along the specified axis.

        """
        if self.features is not None:
            images = [feature.resolve() for feature in self.features]
        result = Image(np.mean(images, axis=axis))

        for image in images:
            result.merge_properties_from(image)

        return result


class Clip(Feature):
    """Clip the input within a minimum and a maximum value.

    This class clips the input values within a specified minimum and maximum
    range.

    Parameters
    ----------
    min: float
        Clip the input to be larger than this value.
    max: float
        Clip the input to be smaller than this value.

    Methods
    -------
    `get(image: np.ndarray | Image, min: float, max: float, **kwargs: Any) --> np.ndarray`
        Clips the input image within the specified minimum and maximum values.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.array([[10, 4], [4, -10]])
    
    Define a clipper feature:
    >>> clipper = dt.Clip(=0, max=5)
    >>> output_image = clipper(input_image)
    >>> print(output_image)
    [[5 4]
     [4 0]]

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If 
    `store_properties` is set to `True`, the returned array will be 
    automatically wrapped in an `Image` object. This behavior is handled 
    internally and does not affect the return type of the `get()` method.

    """

    def __init__(
        self: Clip,
        min: PropertyLike[float] = -np.inf,
        max: PropertyLike[float] = +np.inf,
        **kwargs: Any,
    ):
        """Initialize the parameters for clipping input features.

        This constructor initializes the parameters for clipping input features.

        Parameters
        ----------
        min: float
            Clip the input to be larger than this value.
        max: float
            Clip the input to be smaller than this value.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(min=min, max=max, **kwargs)

    def get(
        self: Clip, 
        image: np.ndarray | Image, 
        min: float = None, 
        max: float = None, 
        **kwargs: Any,
    ) -> np.ndarray:
        """Clips the input image within the specified minimum and maximum values.

        This method clips the input image within the specified minimum and
        maximum values.

        Parameters
        ----------
        image: np.ndarray
            The input image to clip.
        min: float
            Clip the input to be larger than this value.
        max: float
            Clip the input to be smaller than this value.

        Returns
        -------
        np.ndarray
            The clipped image.

        """

        return np.clip(image, min, max)


class NormalizeMinMax(Feature):
    """Image normalization.

    Transforms the input to be between a minimum and a maximum value using
    a linear transformation.

    Parameters
    ----------
    min: float
        The minimum of the transformation.
    max: float
        The maximum of the transformation.
    featurewise: bool
        Whether to normalize each feature independently.

    Methods
    -------
    `get(image: np.ndarray | Image, min: float, max: float, **kwargs: Any) --> np.ndarray`
        Normalizes the input image to be between the specified minimum and 
        maximum values.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.array([[10, 4], [4, -10]])

    Define a min-max normalizer:
    >>> normalizer = dt.NormalizeMinMax(min=-5, max=5)
    >>> output_image = normalizer(input_image)
    >>> print(output_image)
    [[ 5.  2.]
     [ 2. -5.]]

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If 
    `store_properties` is set to `True`, the returned array will be 
    automatically wrapped in an `Image` object. This behavior is handled 
    internally and does not affect the return type of the `get()` method.

    """

    def __init__(
        self: NormalizeMinMax,
        min: PropertyLike[float] = 0,
        max: PropertyLike[float] = 1,
        featurewise: bool = True,
        **kwargs: Any,
    ):
        """Initialize the parameters for min-max normalization.

        This constructor initializes the parameters for min-max normalization.

        Parameters
        ----------
        min: float
            The minimum of the transformation.
        max: float
            The maximum of the transformation.
        featurewise: bool
            Whether to normalize each feature independently.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(min=min, max=max, featurewise=featurewise, **kwargs)

    def get(
        self: NormalizeMinMax, 
        image: np.ndarray | Image, 
        min: float = None, 
        max: float = None, 
        **kwargs: Any,
    ) -> np.ndarray:
        """Normalizes the input image to be between the specified minimum and 
        maximum values.

        This method normalizes the input image to be between the specified
        minimum and maximum values.

        Parameters
        ----------
        image: np.ndarray
            The input image to normalize.
        min: float
            The minimum of the transformation.
        max: float
            The maximum of the transformation.

        Returns
        -------
        np.ndarray
            The normalized image.

        """

        image = image / np.ptp(image) * (max - min)
        image = image - np.min(image) + min
        try:
            image[np.isnan(image)] = 0
        except TypeError:
            pass
        return image


class NormalizeStandard(Feature):
    """Image normalization (standardization).

    Normalize (standardize) the image to have sigma 1 and mean 0.

    Parameters
    ----------
    featurewise: bool
        Whether to normalize each feature independently

    Methods
    -------
    `get(image: np.ndarray | Image, **kwargs: Any) --> np.ndarray`
        Normalizes (standardizes) the input image to have mean 0 and standard 
        deviation 1.

    Returns
    -------
    np.ndarray
    The normalized image as a NumPy array. If `_wrap_array_with_image` or 
    `store_properties` is set to `True`, the result is returned as an `Image` 
    instead.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.array([[1, 2], [3, 4]], dtype=float)
    
    >>> standardizer = dt.NormalizeStandard()
    >>> output_image = standardizer(input_image)
    >>> print(output_image)
    [[-1.34164079 -0.4472136]
     [ 0.4472136   1.34164079]]

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If 
    `store_properties` is set to `True`, the returned array will be 
    automatically wrapped in an `Image` object. This behavior is handled 
    internally and does not affect the return type of the `get()` method.

    """

    def __init__(
        self:NormalizeStandard,
        featurewise: bool = True,
        **kwargs: Any,
    ):
        """Initialize the parameters for standardization.

        This constructor initializes the parameters for standardization.

        Parameters
        ----------
        featurewise: bool
            Whether to normalize each feature independently.
        **kwargs: Any
            Additional keyword arguments.

        """
        super().__init__(featurewise=featurewise, **kwargs)

    def get(
        self: NormalizeStandard,
        image: np.ndarray | Image, 
        **kwargs: Any,
    ) -> np.ndarray:
        """Normalizes the input image to have mean 0 and standard deviation 1.

        This method normalizes the input image to have mean 0 and standard
        deviation 1.

        Parameters
        ----------
        image: np.ndarray
            The input image to normalize.

        Returns
        -------
        np.ndarray
            The normalized image.

        """

        return (image - np.mean(image)) / np.std(image)


class NormalizeQuantile(Feature):
    """Image normalization.

    Center the image to the median, and divide by the difference between the
    quantiles defined by `q_max` and `q_min`.

    Parameters
    ----------
    quantiles: tuple (q_min, q_max), 0.0 < q_min < q_max < 1.0
       Quantile range to calculate scaling factor
    featurewise: bool
        Whether to normalize each feature independently

    Methods
    -------
    `get(image: np.ndarray | Image, quantiles: tuple[float, float], **kwargs: Any) --> np.ndarray`
        Normalizes the input image based on the specified quantiles.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.array([[10, 4], [4, -10]])

    Define a quantile normalizer:
    >>> normalizer = dt.NormalizeQuantile(quantiles=(0.25, 0.75))
    >>> output_image = normalizer(input_image)
    >>> print(output_image)
    [[ 1.2  0. ]
     [ 0.  -2.8]]

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If 
    `store_properties` is set to `True`, the returned array will be 
    automatically wrapped in an `Image` object. This behavior is handled 
    internally and does not affect the return type of the `get()` method.

    """

    def __init__(
        self: NormalizeQuantile, 
        quantiles: tuple[float, float] = (0.25, 0.75),
        featurewise: bool = True,
        **kwargs: Any,
    ):
        """Initialize the parameters for quantile normalization.

        This constructor initializes the parameters for quantile normalization.

        Parameters
        ----------
        quantiles: tuple[float, float]
            Quantile range to calculate scaling factor.
        featurewise: bool
            Whether to normalize each feature independently.
        **kwargs: Any
            Additional keyword arguments.

        """

        super().__init__(
            self,
            quantiles=quantiles,
            featurewise=featurewise,
            **kwargs)

    def get(
        self: NormalizeQuantile,
        image: np.ndarray | Image,
        quantiles: tuple[float, float] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Normalizes the input image based on the specified quantiles.

        This method normalizes the input image based on the specified 
        quantiles.

        Parameters
        ----------
        image: np.ndarray
            The input image to normalize.
        quantiles: tuple[float, float]
            Quantile range to calculate scaling factor.

        Returns
        -------
        np.ndarray
            The normalized image.

        """

        if quantiles is None:
            quantiles = self.quantiles
        q_low, q_high, median = np.quantile(image, (*quantiles, 0.5))
        return (image - median) / (q_high - q_low)


class Blur(Feature):
    """Apply a blurring filter to an image.

    This class applies a blurring filter to an image. The filter function
    must be a function that takes an input image and returns a blurred
    image.

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

    def get(
        self: Blur, 
        image: np.ndarray | Image, 
        **kwargs: Any
    ) -> np.ndarray:
        """Applies the blurring filter to the input image.

        This method applies the blurring filter to the input image.

        Parameters
        ----------
        image: np.ndarray
            The input image to blur.
        kwargs: dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The blurred image.

        """

        kwargs.pop("input", False)
        return utils.safe_call(self.filter, input=image, **kwargs)


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

    def get(
        self: AverageBlur,
        input: np.ndarray | Image,
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
        kwargs: dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The blurred image.
             
        """

        if input.shape[-1] < ksize:
            ksize = (ksize,) * (input.ndim - 1) + (1,)
        else:
            ksize = ((ksize,) * input.ndim,)

        weights = np.ones(ksize) / np.prod(ksize)

        return utils.safe_call(
            ndimage.convolve,
            input=input,
            weights=weights,
            **kwargs,
            )


class GaussianBlur(Blur):
    """Applies a Gaussian blur to images using Gaussian kernels for
    image augmentation.

    This class blurs images by convolving them with a Gaussian filter, which
    smooths the image and reduces high-frequency details. The level of blurring
    is controlled by the standard deviation (`sigma`) of the Gaussian kernel.

    Parameters
    ----------
    sigma: float
        Standard deviation of the Gaussian kernel.

    """

    def __init__(self, sigma: PropertyLike[float] = 2, **kwargs):
        super().__init__(ndimage.gaussian_filter, sigma=sigma, **kwargs)


class MedianBlur(Blur):
    """Applies a median blur to images by replacing each pixel with the median
    of its neighborhood.

    Parameters
    ----------
    ksize: int
        Kernel size.
    kwargs: dict
        Additional parameters sent to the blurring function.

    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(ndimage.median_filter, k=ksize, **kwargs)


class Pool(Feature):
    """Downsamples the image by applying a function to local regions of the
    image.

    This class reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the specified pooling
    function to each block.

    Parameters
    ----------
    pooling_function: function
        A function that is applied to each local region of the image.
        DOES NOT NEED TO BE WRAPPED IN A ANOTHER FUNCTION.
        Must support the axis argument. 
        Examples include np.mean, np.max, np.min, etc.
    ksize: int
        Size of the pooling kernel.
    kwargs: Any
        Additional parameters sent to the pooling function.
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
    )-> np.ndarray:
        """Applies the pooling function to the input image.
        
        This method applies the pooling function to the input image.
        
        Parameters
        ----------
        image: np.ndarray
            The input image to pool.
        ksize: int
            Size of the pooling kernel.
        kwargs: dict[str, Any]
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
            **kwargs
        )


class AveragePooling(Pool):
    """Apply average pooling to an images.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    kwargs: dict
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
        super().__init__(np.mean, ksize=ksize, **kwargs)


class MaxPooling(Pool):
    """Apply max pooling to images.

    This class reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the max function to
    each block. The result is a downsampled image where each pixel value
    represents the maximum value within the corresponding block of the
    original image.
    This is useful for reducing the size of an image while retaining the
    most significant features.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    cval: number
        Value to pad edges with if necessary. Default 0.
    func_kwargs: dict
        Additional parameters sent to the pooling function.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a max pooling feature:
    >>> max_pooling = dt.MaxPooling(ksize=8)
    >>> output_image = max_pooling(input_image)
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
        self: MaxPooling, 
        ksize: PropertyLike[int] = 3,
        **kwargs: Any,
    ):
        """Initialize the parameters for max pooling.

        This constructor initializes the parameters for max pooling.
        
        Parameters
        ----------
        ksize: int
            Size of the pooling kernel.
        **kwargs: Any
            Additional keyword arguments.
        
        """

        super().__init__(np.max, ksize=ksize, **kwargs)


class MinPooling(Pool):
    """Apply min pooling to images.

    This class reduces the resolution of an image by dividing it into
    non-overlapping blocks of size `ksize` and applying the min function to
    each block. The result is a downsampled image where each pixel value
    represents the minimum value within the corresponding block of the
    original image.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    kwargs: dict
        Additional parameters sent to the pooling function.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    Create an input image:
    >>> input_image = np.random.rand(32, 32)

    Define a min pooling feature:
    >>> min_pooling = dt.MinPooling(ksize=3)
    >>> output_image = min_pooling(input_image)
    >>> print(output_image.shape)
    (32, 32)

    Notes
    -----
    Calling this feature returns a `np.ndarray` by default. If
    `store_properties` is set to `True`, the returned array will be
    automatically wrapped in an `Image` object. This behavior is handled
    internally and does not affect the return type of the `get()` method.


    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(np.min, ksize=ksize, **kwargs)


class MedianPooling(Pool):
    """Apply median pooling to images.

    Parameters
    ----------
    ksize: int
        Size of the pooling kernel.
    cval: number
        Value to pad edges with if necessary. Default 0.
    func_kwargs: dict
        Additional parameters sent to the pooling function.
    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(np.median, ksize=ksize, **kwargs)


class Resize(Feature):
    """Resize an image to a specified size.
    
    This is a wrapper around cv2.resize and takes the same arguments.
    Note that the order of the axes is different in cv2 and numpy. In cv2, the
    first axis is the vertical axis, while in numpy it is the horizontal axis.
    This is reflected in the default values of the arguments.

    Parameters
    ----------
    size: tuple
        Size to resize to.
    """

    def __init__(self, dsize: PropertyLike[tuple] = (256, 256), **kwargs):
        super().__init__(dsize=dsize, **kwargs)

    def get(self, image, dsize, **kwargs):
        import cv2
        from deeptrack import config

        if self._wrap_array_with_image:
            image = strip(image)

        return utils.safe_call(
            cv2.resize, positional_args=[image, dsize], **kwargs
        )


try:
    import cv2

    IMPORTED_CV2 = True

    _map_mode_to_cv2_borderType = {
        "reflect": cv2.BORDER_REFLECT,
        "wrap": cv2.BORDER_WRAP,
        "constant": cv2.BORDER_CONSTANT,
        "mirror": cv2.BORDER_REFLECT_101,
        "nearest": cv2.BORDER_REPLICATE,
    }
except ImportError:
    IMPORTED_CV2 = False


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

        if not IMPORTED_CV2:
            raise ImportError(
                "opencv not installed on device, it is an optional "
                "dependency of deeptrack. To use this feature, you "
                "need to install it manually."
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
    )  -> np.ndarray:
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
    (32, 32)]

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

