""" Mathematical oprations and structures

Classses
--------
Clip
    Clip the input within a minimum and a maximum value.
NormalizeMinMax
    Min-max image normalization.
"""

from deeptrack.features import Feature
from deeptrack.image import Image
import numpy as np


class Add(Feature):
    """Adds a value to the input.

    Parameters
    ----------
    value : number
        The value to add
    """

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image + value


class Subtract(Feature):
    """Subtracts a value from the input.

    Parameters
    ----------
    value : number
        The value to subtract
    """

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image - value


class Multiply(Feature):
    """Multiplies the input with a value.

    Parameters
    ----------
    value : number
        The value to multiply with.
    """

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image * value


class Divide(Feature):
    """Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    """

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image / value


class Power(Feature):
    """Raises the input to a power.

    Parameters
    ----------
    value : number
        The power to raise with.
    """

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image ** value


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

    def __init__(self, features=None, axis=0, **kwargs):
        super().__init__(axis=axis, features=features, **kwargs)

    def get(self, images, axis, features, **kwargs):
        if features is not None:
            images = [feature.resolve() for feature in features]
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

    def __init__(self, min=-np.inf, max=+np.inf, **kwargs):
        super().__init__(min=min, max=max, **kwargs)

    def get(self, image, min=None, max=None, **kwargs):
        np.clip(image, min, max, image)
        return image


class NormalizeMinMax(Feature):
    """Image normalization.

    Transforms the input to be between a minimum and a maximum value using a linear transformation.

    Parameters
    ----------
    min : float
        The minimum of the transformation.
    max : float
        The maximum of the transformation.
    """

    def __init__(self, min=0, max=1, **kwargs):
        super().__init__(min=min, max=max, **kwargs)

    def get(self, image, min, max, **kwargs):
        image = image / (np.max(image) - np.min(image)) * (max - min)
        image = image - np.min(image) + min
        image[np.isnan(image)] = 0
        return image


import deeptrack.utils as utils
import skimage

import scipy.ndimage as ndimage


class Blur(Feature):
    def __init__(self, filter_function, mode="reflect", **kwargs):
        self.filter = filter_function
        super().__init__(borderType=borderType, **kwargs)

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

    def __init__(self, ksize=3, **kwargs):
        super().__init__(None, ksize=ksize, **kwargs)

    def get(self, input, ksize, **kwargs):

        if input.shape[-1] < ksize:
            ksize = (ksize,) * (input.ndim - 1) + (1,)
        else:
            ksize = ((ksize,) * input.ndim,)

        weights = np.ones(ksize) / np.prod(ksize)

        return safe_call(ndimage, input=input, weights=weights, **kwargs)


class GaussianBlur(Blur):
    """Augmenter to blur images using gaussian kernels.

    Parameters
    ----------
    sigma : number
        Standard deviation of the gaussian kernel.

    """

    def __init__(self, sigma=2, **kwargs):
        super().__init__(ndimage.gaussian_filter, sigma=sigma, **kwargs)


class MedianBlur(Blur):
    """Blur an image by computing median values over neighbourhoods.

    Parameters
    ----------
    ksize :
        Kernel size.

    """

    def __init__(self, ksize=3, **kwargs):
        super().__init__(ndimage.median_filter, k=k, **kwargs)


## POOLING

import skimage.measure


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

    def __init__(self, pooling_function, ksize=3, **kwargs):
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

    def __init__(self, ksize=3, **kwargs):
        super().__init__(np.mean, ksize=ksize, **kwargs)


### OPENCV2 blur

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

    def __init__(self, filter_function, mode="refelct", **kwargs):
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

    def __init__(self, d=3, sigma_color=50, sigma_space=50, **kwargs):
        super().__init__(
            cv2.bilateralFilter,
            d=d,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            **kwargs
        )