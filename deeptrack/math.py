"""Mathematical operations and structures.

This module provides classes and utilities to perform common mathematical 
operations and transformations on images, including clipping, normalization, 
blurring, and pooling. These are implemented as subclasses of `Feature` for 
seamless integration with the  feature-based design of the library.

Classes
-------
Clip
    Clip the input values within a specified minimum and maximum range.
NormalizeMinMax
    Perform min-max normalization on images.
"""

from typing import Callable, List

import numpy as np
import scipy.ndimage as ndimage
import skimage
import skimage.measure

from . import utils
from .features import Feature
from .image import Image, strip
from .types import PropertyLike


class Average(Feature):
    """Average of input images

    If `features` is not None, it instead resolves all features
    in the list and averages the result.

    Parameters
    ----------
    axis : int or tuple of ints
        Axis along which to average
    features : list of features, optional
    """

    __distributed__ = False

    def __init__(
        self,
        features=PropertyLike[List[Feature] or None],
        axis: PropertyLike[int] = 0,
        **kwargs
    ):

        super().__init__(axis=axis, **kwargs)
        if features is not None:
            self.features = [self.add_feature(feature) for feature in features]

    def get(self, images, axis, **kwargs):
        if self.features is not None:
            images = [feature.resolve() for feature in self.features]
        result = Image(np.mean(images, axis=axis))

        for image in images:
            result.merge_properties_from(image)

        return result


class Clip(Feature):
    """Clip the input within a minimum and a maximum value.

    Parameters
    ----------
    min : float
        Clip the input to be larger than this value.
    max : float
        Clip the input to be smaller than this value.
    """

    def __init__(
        self,
        min: PropertyLike[float] = -np.inf,
        max: PropertyLike[float] = +np.inf,
        **kwargs
    ):
        super().__init__(min=min, max=max, **kwargs)

    def get(self, image, min=None, max=None, **kwargs):
        return np.clip(image, min, max)


class NormalizeMinMax(Feature):
    """Image normalization.

    Transforms the input to be between a minimum and a maximum value using a linear transformation.

    Parameters
    ----------
    min : float
        The minimum of the transformation.
    max : float
        The maximum of the transformation.
    featurewise : bool
        Whether to normalize each feature independently
    """

    def __init__(
        self,
        min: PropertyLike[float] = 0,
        max: PropertyLike[float] = 1,
        featurewise=True,
        **kwargs
    ):
        super().__init__(min=min, max=max, featurewise=featurewise, **kwargs)

    def get(self, image, min, max, **kwargs):
        image = image / np.ptp(image) * (max - min)
        image = image - np.min(image) + min
        try:
            image[np.isnan(image)] = 0
        except TypeError:
            pass
        return image


class NormalizeStandard(Feature):
    """Image normalization.

    Normalize the image to have sigma 1 and mean 0.

    Parameters
    ----------
    featurewise : bool
        Whether to normalize each feature independently
    """

    def __init__(self, featurewise=True, **kwargs):
        super().__init__(featurewise=featurewise, **kwargs)

    def get(self, image, **kwargs):

        return (image - np.mean(image)) / np.std(image)


class NormalizeQuantile(Feature):
    """Image normalization.

    Center the image to the median, and divide by the difference between the quantiles
    defined by `q_max` and `q_min`

    Parameters
    ----------
    quantiles : tuple (q_min, q_max), 0.0 < q_min < q_max < 1.0
       Quantile range to calculate scaling factor
    featurewise : bool
        Whether to normalize each feature independently
    """

    def __init__(self, quantiles=(0.25, 0.75), featurewise=True, **kwargs):
        super().__init__(self, quantiles=quantiles, featurewise=featurewise, **kwargs)

    def get(self, image, quantiles, **kwargs):
        q_low, q_high, median = np.quantile(image, (*quantiles, 0.5))
        return (image - median) / (q_high - q_low)


class Blur(Feature):
    def __init__(
        self, filter_function: Callable, mode: PropertyLike[str] = "reflect", **kwargs
    ):
        self.filter = filter_function
        super().__init__(borderType=mode, **kwargs)

    def get(self, image, **kwargs):
        kwargs.pop("input", False)
        return utils.safe_call(self.filter, input=image, **kwargs)


class AverageBlur(Blur):
    """Blur an image by computing simple means over neighbourhoods.

    Performs a (N-1)D convolution if the last dimension is smaller than the kernel size.

    Parameters
    ----------
    ksize : int
        Kernel size to use.
    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(None, ksize=ksize, **kwargs)

    def get(self, input, ksize, **kwargs):

        if input.shape[-1] < ksize:
            ksize = (ksize,) * (input.ndim - 1) + (1,)
        else:
            ksize = ((ksize,) * input.ndim,)

        weights = np.ones(ksize) / np.prod(ksize)

        return utils.safe_call(ndimage.convolve, input=input, weights=weights, **kwargs)


class GaussianBlur(Blur):
    """Augmenter to blur images using gaussian kernels.

    Parameters
    ----------
    sigma : number
        Standard deviation of the gaussian kernel.

    """

    def __init__(self, sigma: PropertyLike[float] = 2, **kwargs):
        super().__init__(ndimage.gaussian_filter, sigma=sigma, **kwargs)


class MedianBlur(Blur):
    """Blur an image by computing median values over neighbourhoods.

    Parameters
    ----------
    ksize :
        Kernel size.

    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(ndimage.median_filter, k=ksize, **kwargs)


# POOLING


class Pool(Feature):
    """Downsamples the image by applying a function to local regions of the image.

    Parameters
    ----------
    pooling_function : function
        A function that is applied to each local region of the image.
        DOES NOT NEED TO BE WRAPPED IN A ANOTHER FUNCTIOn.
        Must implement the axis argument. Examples include
        np.mean, np.max, np.min, etc.
    ksize : int
        Size of the pooling kernel.
    cval : number
        Value to pad edges with if necessary.
    func_kwargs : dict
        Additional parameters sent to the pooling function.
    """

    def __init__(
        self, pooling_function: Callable, ksize: PropertyLike[int] = 3, **kwargs
    ):
        self.pooling = pooling_function
        super().__init__(ksize=ksize, **kwargs)

    def get(self, image, ksize, **kwargs):
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
    """Apply average pooling to images.

    Parameters
    ----------
    ksize : int
        Size of the pooling kernel.
    cval : number
        Value to pad edges with if necessary. Default 0.
    func_kwargs : dict
        Additional parameters sent to the pooling function.
    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(np.mean, ksize=ksize, **kwargs)


class MaxPooling(Pool):
    """Apply max pooling to images.

    Parameters
    ----------
    ksize : int
        Size of the pooling kernel.
    cval : number
        Value to pad edges with if necessary. Default 0.
    func_kwargs : dict
        Additional parameters sent to the pooling function.
    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(np.max, ksize=ksize, **kwargs)


class MinPooling(Pool):
    """Apply min pooling to images.

    Parameters
    ----------
    ksize : int
        Size of the pooling kernel.
    cval : number
        Value to pad edges with if necessary. Default 0.
    func_kwargs : dict
        Additional parameters sent to the pooling function.
    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(np.min, ksize=ksize, **kwargs)


class MedianPooling(Pool):
    """Apply median pooling to images.

    Parameters
    ----------
    ksize : int
        Size of the pooling kernel.
    cval : number
        Value to pad edges with if necessary. Default 0.
    func_kwargs : dict
        Additional parameters sent to the pooling function.
    """

    def __init__(self, ksize: PropertyLike[int] = 3, **kwargs):
        super().__init__(np.median, ksize=ksize, **kwargs)


class Resize(Feature):
    """Resize an image. This is a wrapper around cv2.resize and takes the same arguments.
    Note that the order of the axes is different in cv2 and numpy. In cv2, the first axis is the
    vertical axis, while in numpy it is the horizontal axis. This is reflected in the default
    values of the arguments.

    Parameters
    ----------
    size : tuple
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


# OPENCV2 blur

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
    def __new__(cls, *args, **kwargs):
        if not IMPORTED_CV2:
            raise ImportError(
                "opencv not installed on device, it is an optional dependency of deeptrack. To use this feature, you need to install it manually."
            )

        return super(BlurCV2, cls).__new__(*args, **kwargs)

    def __init__(
        self, filter_function: Callable, mode: PropertyLike[str] = "refelct", **kwargs
    ):
        self.filter = filter_function
        borderType = _map_mode_to_cv2_borderType[mode]
        super().__init__(borderType=borderType, **kwargs)

    def get(self, image, **kwargs):
        kwargs.pop("src", False)
        kwargs.pop("dst", False)
        utils.safe_call(self.filter, src=image, dst=image, **kwargs)
        return image


class BilateralBlur(Blur):
    """Blur an image using a bilateral filter.

    Bilateral filters blur homogenous areas while trying to
    preserve edges.


    Parameters
    ----------
    d : int
        Diameter of each pixel neighborhood with value range.

    sigma_color : number
        Filter sigma in the color space with value range. A
        large value of the parameter means that farther colors within the
        pixel neighborhood (see `sigma_space`) will be mixed together,
        resulting in larger areas of semi-equal color.

    sigma_space : number
        Filter sigma in the coordinate space with value range. A
        large value of the parameter means that farther pixels will influence
        each other as long as their colors are close enough (see
        `sigma_color`).

    """

    def __init__(
        self,
        d: PropertyLike[int] = 3,
        sigma_color: PropertyLike[float] = 50,
        sigma_space: PropertyLike[float] = 50,
        **kwargs
    ):
        super().__init__(
            cv2.bilateralFilter,
            d=d,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            **kwargs
        )
