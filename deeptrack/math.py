''' Mathematical oprations and structures

Classses
--------
Clip
    Clip the input within a minimum and a maximum value.
NormalizeMinMax
    Min-max image normalization.
'''

from deeptrack.features import Feature
from deeptrack.image import Image
import numpy as np


class Add(Feature):
    '''Adds a value to the input.

    Parameters
    ----------
    value : number
        The value to add
    '''

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image + value

class Subtract(Feature):
    '''Subtracts a value from the input.

    Parameters
    ----------
    value : number
        The value to subtract
    '''

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image - value

class Multiply(Feature):
    '''Multiplies the input with a value.

    Parameters
    ----------
    value : number
        The value to multiply with.
    '''

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image * value

class Divide(Feature):
    '''Divides the input with a value.

    Parameters
    ----------
    value : number
        The value to divide with.
    '''

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image / value

class Power(Feature):
    '''Raises the input to a power.

    Parameters
    ----------
    value : number
        The power to raise with.
    '''

    def __init__(self, value=0, **kwargs):
        super().__init__(value=value, **kwargs)

    def get(self, image, value, **kwargs):
        return image ** value

class Average(Feature):
    ''' Average of input images

    If `features` is not None, it instead resolves all features
    in the list and averages the result.

    Parameters
    ----------
    axis : int or tuple of ints
        Axis along which to average
    features : list of features, optional
    '''
    
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
    '''Clip the input within a minimum and a maximum value.

    Parameters
    ----------
    min : float
        Clip the input to be larger than this value.
    max : float
        Clip the input to be smaller than this value.
    '''

    def __init__(self, min=-np.inf, max=+np.inf, **kwargs):
        super().__init__(min=min, max=max, **kwargs)



    def get(self, image, min=None, max=None, **kwargs):
        np.clip(image, min, max, image)
        return image 


    
class NormalizeMinMax(Feature):
    '''Image normalization.
    
    Transforms the input to be between a minimum and a maximum value using a linear transformation.

    Parameters
    ----------
    min : float
        The minimum of the transformation.
    max : float
        The maximum of the transformation.
    '''

    def __init__(self, min=0, max=1, **kwargs):
        super().__init__(min=min, max=max, **kwargs)



    def get(self, image, min, max, **kwargs):
        image = image / (np.max(image) - np.min(image)) * (max - min)
        image = image - np.min(image) + min 
        image[np.isnan(image)] = 0
        return image


from deeptrack.augmentations import ImgAug
import imgaug.augmenters as iaa

## IMGAUG BLUR
# Please see https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/blur.py
# for source implementation

class AverageBlur(ImgAug):
	'''Blur an image by computing simple means over neighbourhoods.

    The padding behaviour around the image borders is cv2's
    ``BORDER_REFLECT_101``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: no (4)
        * ``int64``: no (5)
        * ``float16``: yes; tested (6)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no
        * ``bool``: yes; tested (7)

        - (1) rejected by ``cv2.blur()``
        - (2) loss of resolution in ``cv2.blur()`` (result is ``int32``)
        - (3) ``int8`` is mapped internally to ``int16``, ``int8`` itself
              leads to cv2 error "Unsupported combination of source format
              (=1), and buffer format (=4) in function 'getRowSumFilter'" in
              ``cv2``
        - (4) results too inaccurate
        - (5) loss of resolution in ``cv2.blur()`` (result is ``int32``)
        - (6) ``float16`` is mapped internally to ``float32``
        - (7) ``bool`` is mapped internally to ``float32``

    Parameters
    ----------
    k : int or tuple of int or tuple of tuple of int or imgaug.parameters.StochasticParameter or tuple of StochasticParameter, optional
        Kernel size to use.

            * If a single ``int``, then that value will be used for the height
              and width of the kernel.
            * If a tuple of two ``int`` s ``(a, b)``, then the kernel size will
              be sampled from the interval ``[a..b]``.
            * If a tuple of two tuples of ``int`` s ``((a, b), (c, d))``,
              then per image a random kernel height will be sampled from the
              interval ``[a..b]`` and a random kernel width will be sampled
              from the interval ``[c..d]``.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing
              the kernel size for the n-th image.
            * If a tuple ``(a, b)``, where either ``a`` or ``b`` is a tuple,
              then ``a`` and ``b`` will be treated according to the rules
              above. This leads to different values for height and width of
              the kernel.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AverageBlur(k=5)

    Blur all images using a kernel size of ``5x5``.

    >>> aug = iaa.AverageBlur(k=(2, 5))

    Blur images using a varying kernel size, which is sampled (per image)
    uniformly from the interval ``[2..5]``.

    >>> aug = iaa.AverageBlur(k=((5, 7), (1, 3)))

    Blur images using a varying kernel size, which's height is sampled
    (per image) uniformly from the interval ``[5..7]`` and which's width is
    sampled (per image) uniformly from ``[1..3]``.

    '''
	def __init__(self, k=(1, 7), **kwargs):
		self.augmenter=iaa.AverageBlur
		super().__init__(k=k, **kwargs)


class BilateralBlur(ImgAug):
	'''Blur/Denoise an image using a bilateral filter.

    Bilateral filters blur homogenous and textured areas, while trying to
    preserve edges.

    See
    http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
    for more information regarding the parameters.

    **Supported dtypes**:

        * ``uint8``: yes; not tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    d : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Diameter of each pixel neighborhood with value range ``[1 .. inf)``.
        High values for `d` lead to significantly worse performance. Values
        equal or less than ``10`` seem to be good. Use ``<5`` for real-time
        applications.

            * If a single ``int``, then that value will be used for the
              diameter.
            * If a tuple of two ``int`` s ``(a, b)``, then the diameter will
              be a value sampled from the interval ``[a..b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing
              the diameter for the n-th image. Expected to be discrete.

    sigma_color : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Filter sigma in the color space with value range ``[1, inf)``. A
        large value of the parameter means that farther colors within the
        pixel neighborhood (see `sigma_space`) will be mixed together,
        resulting in larger areas of semi-equal color.

            * If a single ``int``, then that value will be used for the
              diameter.
            * If a tuple of two ``int`` s ``(a, b)``, then the diameter will
              be a value sampled from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing
              the diameter for the n-th image. Expected to be discrete.

    sigma_space : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Filter sigma in the coordinate space with value range ``[1, inf)``. A
        large value of the parameter means that farther pixels will influence
        each other as long as their colors are close enough (see
        `sigma_color`).

            * If a single ``int``, then that value will be used for the
              diameter.
            * If a tuple of two ``int`` s ``(a, b)``, then the diameter will
              be a value sampled from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing
              the diameter for the n-th image. Expected to be discrete.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.BilateralBlur(
    >>>     d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))

    Blur all images using a bilateral filter with a `max distance` sampled
    uniformly from the interval ``[3, 10]`` and wide ranges for `sigma_color`
    and `sigma_space`.

    '''
	def __init__(self, d=(1, 9), sigma_color=(10, 250), sigma_space=(10, 250), **kwargs):
		self.augmenter=iaa.BilateralBlur
		super().__init__(d=d, sigma_color=sigma_color, sigma_space=sigma_space, **kwargs)



class GaussianBlur(ImgAug):
	'''Augmenter to blur images using gaussian kernels.

    **Supported dtypes**:

    See ``~imgaug.augmenters.blur.blur_gaussian_(backend="auto")``.

    Parameters
    ----------
    sigma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the gaussian kernel.
        Values in the range ``0.0`` (no blur) to ``3.0`` (strong blur) are
        common.

            * If a single ``float``, that value will always be used as the
              standard deviation.
            * If a tuple ``(a, b)``, then a random value from the interval
              ``[a, b]`` will be picked per image.
            * If a list, then a random value will be sampled per image from
              that list.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.GaussianBlur(sigma=1.5)

    Blur all images using a gaussian kernel with a standard deviation of
    ``1.5``.

    >>> aug = iaa.GaussianBlur(sigma=(0.0, 3.0))

    Blur images using a gaussian kernel with a random standard deviation
    sampled uniformly (per image) from the interval ``[0.0, 3.0]``.

    '''
	def __init__(self, sigma=(0.0, 3.0), **kwargs):
		self.augmenter=iaa.GaussianBlur
		super().__init__(sigma=sigma, **kwargs)



class MeanShiftBlur(ImgAug):
	'''Apply a pyramidic mean shift filter to each image.

    See also :func:`blur_mean_shift_` for details.

    This augmenter expects input images of shape ``(H,W)`` or ``(H,W,1)``
    or ``(H,W,3)``.

    .. note::

        This augmenter is quite slow.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.blur.blur_mean_shift_`.

    Parameters
    ----------
    spatial_radius : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Spatial radius for pixels that are assumed to be similar.

            * If ``number``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly
              sampled per image from the interval ``[a, b)``.
            * If ``list``: A random value will be sampled from that ``list``
              per image.
            * If ``StochasticParameter``: The parameter will be queried once
              per batch for ``(N,)`` values with ``N`` denoting the number of
              images.

    color_radius : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Color radius for pixels that are assumed to be similar.

            * If ``number``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly
              sampled per image from the interval ``[a, b)``.
            * If ``list``: A random value will be sampled from that ``list``
              per image.
            * If ``StochasticParameter``: The parameter will be queried once
              per batch for ``(N,)`` values with ``N`` denoting the number of
              images.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MeanShiftBlur()

    Create a mean shift blur augmenter.

    '''
	def __init__(self, spatial_radius=(5.0, 40.0), color_radius=(5.0, 40.0), **kwargs):
		self.augmenter=iaa.MeanShiftBlur
		super().__init__(spatial_radius=spatial_radius, color_radius=color_radius, **kwargs)



class MedianBlur(ImgAug):
	'''Blur an image by computing median values over neighbourhoods.

    Median blurring can be used to remove small dirt from images.
    At larger kernel sizes, its effects have some similarity with Superpixels.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    k : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Kernel size.

            * If a single ``int``, then that value will be used for the
              height and width of the kernel. Must be an odd value.
            * If a tuple of two ints ``(a, b)``, then the kernel size will be
              an odd value sampled from the interval ``[a..b]``. ``a`` and
              ``b`` must both be odd values.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing
              the kernel size for the nth image. Expected to be discrete. If
              a sampled value is not odd, then that value will be increased
              by ``1``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MedianBlur(k=5)

    Blur all images using a kernel size of ``5x5``.

    >>> aug = iaa.MedianBlur(k=(3, 7))

    Blur images using varying kernel sizes, which are sampled uniformly from
    the interval ``[3..7]``. Only odd values will be sampled, i.e. ``3``
    or ``5`` or ``7``.

    '''
	def __init__(self, k=(1, 7), **kwargs):
		self.augmenter=iaa.MedianBlur
		super().__init__(k=k, **kwargs)



class MotionBlur(ImgAug):
	'''Blur images in a way that fakes camera or object movements.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.convolutional.Convolve`.

    Parameters
    ----------
    k : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Kernel size to use.

            * If a single ``int``, then that value will be used for the height
              and width of the kernel.
            * If a tuple of two ``int`` s ``(a, b)``, then the kernel size
              will be sampled from the interval ``[a..b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing
              the kernel size for the n-th image.

    angle : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Angle of the motion blur in degrees (clockwise, relative to top center
        direction).

            * If a number, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be uniformly sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    direction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Forward/backward direction of the motion blur. Lower values towards
        ``-1.0`` will point the motion blur towards the back (with angle
        provided via `angle`). Higher values towards ``1.0`` will point the
        motion blur forward. A value of ``0.0`` leads to a uniformly (but
        still angled) motion blur.

            * If a number, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be uniformly sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use when rotating the kernel according to
        `angle`.
        See :func:`~imgaug.augmenters.geometric.Affine.__init__`.
        Recommended to be ``0`` or ``1``, with ``0`` being faster, but less
        continuous/smooth as `angle` is changed, particularly around multiple
        of ``45`` degrees.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MotionBlur(k=15)

    Apply motion blur with a kernel size of ``15x15`` pixels to images.

    >>> aug = iaa.MotionBlur(k=15, angle=[-45, 45])

    Apply motion blur with a kernel size of ``15x15`` pixels and a blur angle
    of either ``-45`` or ``45`` degrees (randomly picked per image).

    '''
	def __init__(self, k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0), order=1, **kwargs):
		self.augmenter=iaa.MotionBlur
		super().__init__(k=k, angle=angle, direction=direction, order=order, **kwargs)










## IMGAUG POOLING
# Please see https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/pooling.py
# for source implementation


class AveragePooling(ImgAug):
	'''
    Apply average pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by averaging the
    pixel values within these windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    Note that this augmenter is very similar to ``AverageBlur``.
    ``AverageBlur`` applies averaging within windows of given kernel size
    *without* striding, while ``AveragePooling`` applies striding corresponding
    to the kernel size, with optional upscaling afterwards. The upscaling
    is configured to create "pixelated"/"blocky" images by default.

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.avg_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AveragePooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.AveragePooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.AveragePooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.AveragePooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.AveragePooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    '''
	def __init__(self, kernel_size=(1, 5), keep_size=True, **kwargs):
		self.augmenter=iaa.AveragePooling
		super().__init__(kernel_size=kernel_size, keep_size=keep_size, **kwargs)


class MaxPooling(ImgAug):
	'''
    Apply max pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by taking the
    maximum pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The maximum within each pixel window is always taken channelwise..

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.max_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MaxPooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.MaxPooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.MaxPooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.MaxPooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.MaxPooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    '''
	def __init__(self, kernel_size=(1, 5), keep_size=True, **kwargs):
		self.augmenter=iaa.MaxPooling
		super().__init__(kernel_size=kernel_size, keep_size=keep_size, **kwargs)


class MedianPooling(ImgAug):
	'''
    Apply median pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by taking the
    median pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The median within each pixel window is always taken channelwise.

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.median_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MedianPooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.MedianPooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.MedianPooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.MedianPooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.MedianPooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    '''
	def __init__(self, kernel_size=(1, 5), keep_size=True, **kwargs):
		self.augmenter=iaa.MedianPooling
		super().__init__(kernel_size=kernel_size, keep_size=keep_size, **kwargs)


class MinPooling(ImgAug):
	'''
    Apply minimum pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by taking the
    minimum pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The minimum within each pixel window is always taken channelwise.

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.min_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MinPooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.MinPooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.MinPooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.MinPooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.MinPooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    '''
	def __init__(self, kernel_size=(1, 5), keep_size=True, **kwargs):
		self.augmenter=iaa.MinPooling
		super().__init__(kernel_size=kernel_size, keep_size=keep_size, **kwargs)


