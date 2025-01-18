"""Classes to augment images.

This module provides the `augmentations` DeepTrack2 classes
that manipulates an image object with various transformations.

When used in a training pipeline, these augmentations synthetically
increase the volume of training data for machine learning models.

Key Features
------------
- **General Augmentations**

    Basic image augmentation methods to perform transformations
    such as flipping an image horizontally (left-right), vertically (up-down),
    or along the diagonal.

- **Advanced Augmentations**

    For users who require more control over the image transformations.
    Advanced augmentation methods allow operations like translation, scaling,
    rotation, and shearing. These operations can be customized with
    user-specified parameters (e.g., degrees of rotation, scaling factors),
    giving flexibility in how the images are transformed.

- **Caching**

    To avoid redundant computations, the `Reuse class` offers
    caching functionality to save the outputs of feature outputs to be reused
    later, saving time and computational resources.

- **Cropping**

    Enables different methods to crop an image. Region-specific cropping, 
    cropping based on multiples of height/width of and image, and crop to 
    remove empty space at edges of an image.

- **Padding**

    Padding operations allow you to extend the shape of an image by
    adding extra pixels around its edges, which is essential for ensuring
    that the shape of the image stay consistent.

Module Structure
----------------

- `Augmentation`: Base class for augmentations.

- `Reuse`: Stores and reuses feature outputs.

- `FlipLR`: Flips image left to right.

- `FlipUD`: Flips image up to down.

- `FlipDiagonal`: Flips image along the diagonal.

- `Affine`: Translation, scaling, rotation, shearing.

- 'ElasticTransformation': Transform using a displacement field.

- `Crop`: Crop regions of an image.

- `CropToMultiplesOf`: Crops image until height/width is multiple of a value.

- `CropTight`: Crops to remove empty space at start and end of a 3D array.

- `Pad`: Pads image.

- `PadMultiplesOf`: Pad images until height/width is a multiple of a value.

Examples
--------
Flip an image of a particle up-down then flips left-right:

    >>> import deeptrack as dt

    >>> particle = dt.PointParticle()
    >>> optics = dt.Fluorescence()
    >>> image = dt.Value(optics(particle))\ 
    ...     >> dt.FlipUD(p=1.0) >> dt.FlipLR(p=1.0)
    image.plot()


Reuse the output of a pipeline twice, augmented randomly by FlipLR.

    >>> import deeptrack as dt
    
    >>> particle = dt.PointParticle()
    >>> optics = dt.Fluorescence()
    >>> pipeline = dt.Reuse(pipeline, uses=2) >> dt.FlipLR()    
    >>> image = optics(particle) >> pipeline
    >>> image.plot()


"""

import warnings
import random
from typing import Callable, List, Union, Tuple, Dict

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from . import utils
from .features import Feature
from .image import Image
from .types import ArrayLike, PropertyLike


class Augmentation(Feature):
    """Base abstract augmentation class.

    This class provides the template for the other augmentation
    classes to inherit from, and is primarily used to handle the
    input cases of either `Image` objects or `List[Image]` objects
    via the `_image_wrapped_process_and_get` and `_no_wrap_process_and_get`
    methods respectively.

    Parameters
    ----------
    time_consistent: boolean
       Whether to augment all images in a sequence equally.
       
    """

    def __init__(
        self,
        time_consistent: bool = False,
        **kwargs
    ) -> None:
        super().__init__(time_consistent=time_consistent, **kwargs)

    def _image_wrapped_process_and_get (
        self,
        image_list: List[Image],
        time_consistent: PropertyLike[bool],
        **kwargs
    ) -> List[List]:
        """Augments a list of images and returns a wrapped output.
        
        This function handles input to ensure compatibility with nested
        image lists and wraps the output into a new `Image` object.
        
        For non-wrapping, see the `_no_wrap_process_and_get` method.
        
        """
        
        if not isinstance(image_list, list):
            wrap_depth = 2
            image_list_of_lists = [[image_list]]
        elif len(image_list) == 0 or not isinstance(image_list[0], list):
            wrap_depth = 1
            image_list_of_lists = [image_list]
        else:
            wrap_depth = 0
            image_list_of_lists = image_list

        new_list_of_lists = []
        for image_list in image_list_of_lists:

            if time_consistent:
                self.seed()

            augmented_list = []
            for image in image_list:
                self.seed()
                augmented_image = Image(self.get(image, **kwargs))
                augmented_image.merge_properties_from(image)
                self.update_properties(augmented_image, **kwargs)
                augmented_list.append(augmented_image)

            new_list_of_lists.append(augmented_list)

        for _ in range(wrap_depth):
            new_list_of_lists = new_list_of_lists[0]

        return new_list_of_lists
    
    def _no_wrap_process_and_get(
        self,
        image_list: List[Image],
        time_consistent: PropertyLike[bool],
        **kwargs
    ) -> List[List]:
        """Augments a list of images and returns the raw output.
        
        This function handles input to ensure compatibility with nested
        image lists and does not wrap the output into a new `Image` object.
        
        For wrapping, see the `_image_wrapped_process_and_get` method.
        
        """
        if not isinstance(image_list, list):
            wrap_depth = 2
            image_list_of_lists = [[image_list]]
        elif len(image_list) == 0 or not isinstance(image_list[0], list):
            wrap_depth = 1
            image_list_of_lists = [image_list]
        else:
            wrap_depth = 0
            image_list_of_lists = image_list

        new_list_of_lists = []
        for image_list in image_list_of_lists:

            if time_consistent:
                self.seed()

            augmented_list = []
            for image in image_list:
                self.seed()
                augmented_image = self.get(image, **kwargs)
                augmented_list.append(augmented_image)

            new_list_of_lists.append(augmented_list)

        for _ in range(wrap_depth):
            new_list_of_lists = new_list_of_lists[0]

        return new_list_of_lists


    def update_properties(self, *args, **kwargs):
        pass
    """Abstract method to update the properties of the image.

    Currently not in use.
    
    """    
        


class Reuse(Feature):
    """Acts like cache.

    `Reuse` stores the output of a feature and reuses it for subsequent calls,
    even if it is updated. This is can be used after a time-consuming feature
    to augment the output of the feature without recalculating it.

    Parameters
    ----------
    feature: Feature
       The feature to reuse.
       
    uses: int
       Number of each stored image uses before evaluating `feature`.
       Note that the actual total number of uses is `uses * storage`.
       Should be constant.
       
    storage: int
       Number of instances of the output of `feature` to cache.
       Should be constant.

    """

    __distributed__ = False

    def __init__(
        self,
        feature: Feature,
        uses: int = 2,
        storage:int = 1,
        **kwargs
    ):
        super().__init__(uses=uses, storage=storage, **kwargs)
        self.feature = self.add_feature(feature)
        self.counter = 0
        self.cache = []

    def get(
        self,
        image: Image,
        uses: PropertyLike[int],
        storage: PropertyLike[int],
        **kwargs
    ) -> List[Image]:
        """Abstract method which performs the `Reuse` augmentation.

        """

        self.cache = self.cache[-storage:]
        output = None

        if len(self.cache) < storage or self.counter % (uses * storage) == 0:
            output = self.feature(image)
            self.cache.append(output)
        else:
            output = random.choice(self.cache)

        self.counter += 1

        if not isinstance(output, list):
            output = [output]

        if not self._wrap_array_with_image:
            return output
        
        outputs = []
        for image in output:
            image_copy = Image(image)
            # shallow copy properties before output
            image_copy.properties = [prop.copy() for prop in image.properties]
            outputs.append(image_copy)

        return outputs


class FlipLR(Augmentation):
    """Flips images left-right.

    Updates all properties called "position" to flip the second index in the 
    image.

    Parameters
    ----------
    p: float
       Probability of flipping the image, 
       leaving as default (0.5 ) is sufficient most of the time.

    augment: bool
       Whether to perform the augmentation.
       
    """

    def __init__(
        self,
        p: PropertyLike[float] = 0.5,
        augment: PropertyLike[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(
            p=p,
            augment=(
                lambda p: np.random.rand() < p
            ) if augment is None else augment,
            **kwargs,
        )

    def get(
        self,
        image: Image,
        augment: PropertyLike[bool],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `FlipLR` augmentation.

        """
        
        if augment:
            image = image[:, ::-1]
        return image

    def update_properties(
        self,
        image: Image,
        augment: PropertyLike[bool],
        **kwargs
    ) -> None:
        """Abstract method to update the properties of the image.
    
        """            
        if augment:
            for prop in image.properties:
                if "position" in prop:
                    position = np.array(prop["position"])
                    position[..., 1] = image.shape[1] - position[..., 1] - 1
                    prop["position"] = position


class FlipUD(Augmentation):
    """Flips images up-down.

    Updates all properties called "position" to flip the first index
    in the image.

    Parameters
    ----------
    p: float
       Probability of flipping the image,
       leaving as default (0.5) is sufficient most of the time.

    augment: bool
       Whether to perform the augmentation.
       
    """

    def __init__(
        self,
        p: float = 0.5,
        augment: PropertyLike[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(
            p=p,
            augment=(
                lambda p: np.random.rand() < p
            ) if augment is None else augment,
            **kwargs,
        )

    def get(
        self,
        image: Image,
        augment: PropertyLike[bool],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `FlipUD` augmentation.

        """
        
        if augment:
            image = image[::-1]
        return image

    def update_properties(
        self,
        image: Image,
        augment: PropertyLike[bool],
        **kwargs
    ) -> None:
        """Abstract method to update the properties of the image.
    
        """    
        if augment:
            for prop in image.properties:
                if "position" in prop:
                    position = np.array(prop["position"])
                    position[..., 0] = image.shape[0] - position[..., 0] - 1
                    prop["position"] = position


class FlipDiagonal(Augmentation):
    """Flips images along the main diagonal.

    Updates all properties called "position" by swapping
    the first and second index.

    Parameters
    ----------
    p: float
       Probability of flipping the image,
       leaving as default (0.5) is sufficient most of the time.

    augment: bool
       Whether to perform the augmentation.
       
    """

    def __init__(
        self,
        p: PropertyLike[float] = 0.5,
        augment: PropertyLike[bool] = None,
        **kwargs
    ):
        super().__init__(
            p=p,
            augment=(
                lambda p: np.random.rand() < p
                ) if augment is None else augment,
            **kwargs,
        )

    def get(
        self,
        image: Image,
        augment: PropertyLike[bool],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `FlipDiagonal` augmentation.

        """        
        if augment:
            image = np.transpose(image, axes=(1, 0, *range(2, image.ndim)))
        return image

    def update_properties(
        self,
        image: Image,
        augment: PropertyLike[bool],
        **kwargs
    ) -> None:
        """Abstract method to update the properties of the image.
    
        """            
        if augment:
            for prop in image.properties:
                if "position" in prop:
                    position = np.array(prop["position"])
                    t = np.array(position[..., 0])
                    position[..., 0] = position[..., 1]
                    position[..., 1] = t
                    prop["position"] = position


class Affine(Augmentation):
    """Augmenter to apply affine transformations to images.

    Affine transformations include:

        - `Translation`
        - `Scaling`
        - `Rotation`
        - `Shearing`

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.

    Parameters
    ----------
    scale: float or tuple of float or list of float or dict
        Scaling factor to use, where ``1.0`` denotes "no change" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.
        If two values are provided (using tuple, list, or dict),
        the two first dimensions of the input are scaled individually.

    translate: float or tuple of float or list of float or dict
        Translation in pixels.

    translate_px: float or tuple of float or list of float or dict
        DEPRECATED, use translate.

    rotate: float
        Rotation in radians, i.e. Rotation happens around the *center* of the
        image.

    shear: float
        Shear in radians. Values in the range (-pi/4, pi/4) are common.

    order: int
        Interpolation order to use. Same meaning as in ``skimage``:

            * ``0``: ``Nearest-neighbor``
            * ``1``: ``Bi-linear`` (default)
            * ``2``: ``Bi-quadratic`` (not recommended by skimage)
            * ``3``: ``Bi-cubic``
            * ``4``: ``Bi-quartic``
            * ``5``: ``Bi-quintic``
    """

    def __init__(
        self,
        scale: PropertyLike[float] = 1,
        translate: PropertyLike[Union[float,  None]] = None,
        translate_px: PropertyLike[float] = 0.0,
        rotate: PropertyLike[float] = 0.0,
        shear: PropertyLike[float] = 0.0,
        order: PropertyLike[int] = 1,
        cval: PropertyLike[float] = 0.0,
        mode: PropertyLike[str] = "reflect",
        **kwargs
    ) -> None:
        
        if translate is None:
            translate = translate_px
        super().__init__(
            scale=scale,
            translate=translate,
            translate_px=translate,
            rotate=rotate,
            shear=shear,
            order=order,
            cval=cval,
            mode=mode,
            **kwargs,
        )

    def _process_properties(
        self,
        properties: Dict
    ) -> Dict:
        
        properties = super()._process_properties(properties)
        # Make translate tuple.
        translate = properties["translate"]
        if isinstance(translate, (float, int)):
            translate = (translate, translate)
        if isinstance(translate, dict):
            translate = (translate["x"], translate["y"])
        properties["translate"] = translate

        # Make scale tuple.
        scale = properties["scale"]
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        if isinstance(scale, dict):
            scale = (scale["x"], scale["y"])
        properties["scale"] = scale

        return properties

    def get(
        self,
        image: Image,
        scale: PropertyLike[float],
        translate: PropertyLike[float],
        rotate: PropertyLike[float],
        shear: PropertyLike[float],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `Affine` augmentation.
        
        Affine transformations include:
        - `Translation`
        - `Scaling`
        - `Rotation`
        - `Shearing`

        """    
        assert (
            image.ndim == 2 or image.ndim == 3
        ), "Affine only supports 2-dimensional or 3-dimension inputs, got {0}"\
        .format(image.ndim)

        dx, dy = translate
        fx, fy = scale

        cr = np.cos(rotate)
        sr = np.sin(rotate)

        k = np.tan(shear)

        scale_map = np.array([[1 / fx, 0], [0, 1 / fy]])
        rotation_map = np.array([[cr, sr], [-sr, cr]])
        shear_map = np.array([[1, 0], [-k, 1]])

        mapping = scale_map @ rotation_map @ shear_map

        shape = image.shape
        center = np.array(shape[:2]) / 2

        d = center - np.dot(mapping, center) - np.array([dy, dx])

        # Clean up kwargs.
        kwargs.pop("input", False)
        kwargs.pop("matrix", False)
        kwargs.pop("offset", False)
        kwargs.pop("output", False)

        # Call affine_transform.
        if image.ndim == 2:
            new_image = utils.safe_call(
                ndimage.affine_transform,
                input=image,
                matrix=mapping,
                offset=d,
                **kwargs,
            )

            new_image = Image(new_image)
            new_image.merge_properties_from(image)
            image = new_image

        elif image.ndim == 3:
            for z in range(shape[-1]):
                image[:, :, z] = utils.safe_call(
                    ndimage.affine_transform,
                    input=image[:, :, z],
                    matrix=mapping,
                    offset=d,
                    **kwargs,
                )

        # Map positions.
        if hasattr(image, "properties"):
            inverse_mapping = np.linalg.inv(mapping)
            for prop in image.properties:
                if "position" in prop:
                    position = np.array(prop["position"])

                    inverted = (
                        np.dot(
                            inverse_mapping,
                            (position[..., :2] - center + np.array([dy, dx]))[
                                ..., np.newaxis
                            ],
                        )
                        .squeeze()
                        .transpose()
                    ) + center

                    position[..., :2] = inverted

                    prop["position"] = position

        return image


class ElasticTransformation(Augmentation):
    """Transform images using displacement fields.

    The augmenter creates a random distortion field using `alpha` and `sigma`,
    which define the strength and smoothness of the field respectively.
    These are used to transform the input locally.

    Note:
        This augmentation does not currently update the position property
        of the image, meaning that it is not recommended to use it if
        the data label is derived from the position properties of the
        resulting image.

    For a detailed explanation, see:

        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003.


    Parameters
    ----------
    alpha: float
        Strength of the distortion field.
        Common values are in the range (10, 100)

    sigma: float
        Standard deviation of the gaussian kernel used to smooth the distortion
        fields. Common values are in the range (1, 10)

    ignore_last_dim: bool
        Whether to skip creating a distortion field for the last dimension.
        This is often desired if the last dimension is a channel dimension
        (such as a color image.) In that case, the three channels are
        transformed identically and do not "bleed" into eachother.

    order: int
        Interpolation order to use. Takes integers from 0 to 5

            * 0: ``Nearest-neighbor``
            * 1: ``Bi-linear`` (default)
            * 2: ``Bi-quadratic`` (not recommended by skimage)
            * 3: ``Bi-cubic``
            * 4: ``Bi-quartic``
            * 5: ``Bi-quintic``

    cval: float
        The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to ``constant``.

    mode: str
        Parameter that defines newly created pixels.
        May take the same values as in :func:`scipy.ndimage.map_coordinates`,
        i.e. ``constant``, ``nearest``, ``reflect`` or ``wrap``.

    """

    def __init__(
        self,
        alpha: PropertyLike[float] = 20,
        sigma: PropertyLike[float] = 2,
        ignore_last_dim: PropertyLike[bool] = True,
        order: PropertyLike[int] = 3,
        cval: PropertyLike[float] = 0,
        mode: PropertyLike[str] = "constant",
        **kwargs
    ) -> None:
        super().__init__(
            alpha=alpha,
            sigma=sigma,
            ignore_last_dim=ignore_last_dim,
            order=order,
            cval=cval,
            mode=mode,
            **kwargs,
        )

    def get(
        self,
        image: Image,
        sigma: PropertyLike[float],
        alpha: PropertyLike[float],
        ignore_last_dim: PropertyLike[bool],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `ElasticTransformation` augmentation.

        """    
        shape = image.shape

        if ignore_last_dim:
            shape = shape[:-1]

        deltas = []
        ranges = []
        coordinates = []

        for dim in shape:
            deltas.append(
                gaussian_filter(
                    (np.random.rand(*shape) * 2 - 1),
                    sigma,
                    mode="constant",
                    cval=0,
                )
                * alpha
            )

            ranges.append(np.arange(dim))

        grids = list(np.meshgrid(*ranges))

        for grid, delta in zip(grids, deltas):
            dDim = np.transpose(
                grid, axes=(1, 0
                ) + tuple(range(2, grid.ndim))) + delta
            coordinates.append(np.reshape(dDim, (-1, 1)))

        if ignore_last_dim:
            for z in range(image.shape[-1]):
                image[..., z] = utils.safe_call(
                    map_coordinates,
                    input=image[..., z],
                    coordinates=coordinates,
                    **kwargs,
                ).reshape(shape)
        else:
            image = utils.safe_call(
                map_coordinates, input=image, coordinates=coordinates, **kwargs
            ).reshape(shape)

        # TODO: implement interpolated coordinate mapping for property positions
        # for prop in image:
        #     if "position" in prop:

        return image


class Crop(Augmentation):
    """Crops a regions of an image.

    Parameters
    ----------
    feature: Feature or list of Features
        Feature(s) to augment.

    crop: int or tuple of int or list of int or Callable[Image]->tuple of ints
        Number of pixels to remove or retain (depending in `crop_mode`)
        If a tuple or list, it is assumed to be per axis.
        Can also be a function that returns any of the other types.

    crop_mode: str {"retain", "remove"}
        How the `crop` argument is interpreted. If "remove", then
        `crop` denotes the amount to crop from the edges. If "retain",
        `crop` denotes the size of the output.

    corner: tuple of ints or Callable[Image]->tuple of ints or "random"
        Top left corner of the cropped region. Can be a tuple of ints,
        a function that returns a tuple of ints or the string random.
        If corner is placed so that the cropping cannot be performed,
        the modulo of the corner with the allowed region is used.

    """

    def __init__(
        self,
        *args,
        crop: Union[List[int], Tuple[int], int] = (64, 64),
        crop_mode: PropertyLike[str] = "retain",
        corner: PropertyLike[str] = "random",
        **kwargs
    ) -> None:
        super().__init__(
            *args,
            crop=crop,
            crop_mode=crop_mode,
            corner=corner,
            **kwargs,
        )

    def get(
        self,
        image: Image,
        corner: PropertyLike[str],
        crop: Union[int, List[int], Tuple[int]],
        crop_mode: PropertyLike[str],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `Crop` augmentation.

        """    
        # Get crop argument.
        if callable(crop):
            crop = crop(image)
        if isinstance(crop, int):
            crop = (crop,) * image.ndim

        crop = [c if c is not None else image.shape[i]\
        for i, c in enumerate(crop)]

        # Get amount to crop from image.
        if crop_mode == "retain":
            crop_amount = np.array(image.shape) - np.array(crop)
        elif crop_mode == "remove":
            crop_amount = np.array(crop)
        else:
            raise ValueError("Unrecognized crop_mode {0}".format(crop_mode))

        # Contain within image.
        crop_amount = np.amax(
            (np.array(crop_amount), [0] * image.ndim),
            axis=0
        )
        crop_amount = np.amin((np.array(image.shape) - 1, crop_amount), axis=0)

        # Get corner of crop.
        if isinstance(corner, str) and corner == "random":

            # Ensure seed is consistent
            slice_start = [np.random.randint(m + 1) for m in crop_amount]
        elif callable(corner):
            slice_start = corner(image)
        else:
            slice_start = corner

        # Ensure compatible with image.
        slice_start = [c % (m + 1) for c, m in zip(slice_start, crop_amount)]
        slice_end = [
            a - c + s for a, s, c in zip(image.shape, slice_start, crop_amount)
        ]

        slices = tuple(
            [
                slice(slice_start_i, slice_end_i)
                for slice_start_i, slice_end_i in zip(slice_start, slice_end)
            ]
        )

        cropped_image = image[slices]

        # Update positions.
        if hasattr(image, "properties"):
            cropped_image.properties =\
            [dict(prop) for prop in image.properties]
            for prop in cropped_image.properties:
                if "position" in prop:
                    position = np.array(prop["position"])
                    try:
                        position[..., 0:2] -= np.array(slice_start)[0:2]
                        prop["position"] = position
                    except IndexError:
                        pass

        return cropped_image


class CropToMultiplesOf(Crop):
    """Crop images down until their height/width is a multiple of a value.

    Parameters
    ----------
    multiple: int or tuple of (int or None)
        Images will be cropped down until their width is a multiple of
        this value. If a tuple, it is assumed to be a multiple per axis.
        A value of None or -1 indicates to skip that axis.

    """

    def __init__(
        self,
        multiple: Union[int, Tuple[int], Tuple[None]] = 1,
        corner: PropertyLike[str] = "random",
        **kwargs
    ) -> None:
        
        kwargs.pop("crop", False)
        kwargs.pop("crop_mode", False)

        def image_to_crop(
            image: Image
        ) -> Image:
            
            shape = image.shape
            multiple = self.multiple()

            if not isinstance(multiple, (list, tuple, np.ndarray)):
                multiple = (multiple,) * image.ndim
            new_shape = list(shape)
            idx = 0
            for dim, mul in zip(shape, multiple):
                if mul is not None and mul != -1:
                    new_shape[idx] = int((dim // mul) * mul)
                idx += 1

            return new_shape

        super().__init__(
            multiple=multiple,
            corner=corner,
            crop=lambda: image_to_crop,
            crop_mode="retain",
            **kwargs,
        )


class CropTight(Feature):
    """Crops input array to remove empty space.

    Removes indices from the start and end of the array,
    where all values are below eps.
    Currently only works for 3D arrays.

    Parameters
    ----------
    eps: float
        The threshold for considering a pixel to be empty,
        by default 1e-10.

    """
    def __init__(
        self,
        eps: PropertyLike[float] = 1e-10,
        **kwargs
    ) -> None:
        super().__init__(eps=eps, **kwargs)

    def get(
        self,
        image: Image,
        eps: PropertyLike[float],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `CropTight` augmentation.
        
        `CropTight` removes indices from the start and end of the array,
        where all values are below eps.

        """          
        image = np.asarray(image)
        image = image[..., np.any(image > eps, axis=(0, 1))]
        image = image[np.any(image > eps, axis=(1, 2)), ...]
        image = image[:, np.any(image > eps, axis=(0, 2)), :]

        return image


class Pad(Augmentation):
    """Pads the image.

    Arguments match this of numpy.pad, save for pad_width, which is called px,
    and is defined as (left, right, up, down, before_axis_3, after_axis_3, ...)

    Parameters
    ----------
    px: int or list of int
        Ammount to pad in each direction.

    """

    def __init__(
        self,
        px: Union[List[int], Tuple[int]] = (0, 0, 0, 0),
        mode: PropertyLike[str] = "constant",
        cval: PropertyLike[float] = 0,
        **kwargs
    ) -> None:
        super().__init__(px=px, mode=mode, cval=cval, **kwargs)

    def get(
        self,
        image: Image,
        px: PropertyLike[int],
        **kwargs
    ) -> Image:
        """Abstract method which performs the `Pad` augmentation.

        """    
        padding = []
        if callable(px):
            px = px(image)
        elif isinstance(px, int):
            padding = [(px, px)] * image.ndim

        for idx in range(0, len(px), 2):
            padding.append((px[idx], px[idx + 1]))

        while len(padding) < image.ndim:
            padding.append((0, 0))

        return utils.safe_call(
            np.pad,
            positional_args=(image, padding),
            **kwargs,
            )
 

    def _image_wrap_process_and_get(
        self,
        images: List[Image],
        **kwargs
    ) -> List[Image]:
        """Simple method which wraps an `Image` in a `List`.
        
        """
        results = [self.get(image, **kwargs) for image in images]

        # for idx, result in enumerate(results):
        #    if isinstance(result, tuple):
        #    results[idx] = Image(result[0]).merge_properties_from(images[idx])
        #    else:
        #    Image(results[idx]).merge_properties_from(images[idx])
        return results


class PadToMultiplesOf(Pad):
    """Pad images until their height/width is a multiple of a value.

    Parameters
    ----------

    multiple: int or tuple of (int or None)
        Images will be padded until their width is a multiple of
        this value. If a tuple, it is assumed to be a multiple per axis.
        A value of None or -1 indicates to skip that axis.

    """

    def __init__(
        self,
        multiple: PropertyLike[Union[int, Tuple[int], Tuple[None]]] = 1,
        **kwargs
    ) -> None:
        
        def amount_to_pad(
            image: Image
        ) -> List[int]:
            """Method to calculate number of pixels.
        
            Calculates the number of pixels needed to pad an image 
            for its height/width to be a multiple of a value.
        
            """
            shape = image.shape
            multiple = self.multiple()

            if not isinstance(multiple, (list, tuple, np.ndarray)):
                multiple = (multiple,) * image.ndim
            new_shape = [0] * (image.ndim * 2)
            idx = 0
            for dim, mul in zip(shape, multiple):
                if mul is not None and mul != -1:
                    to_add = -dim % mul
                    to_add_first = to_add // 2
                    to_add_after = to_add - to_add_first
                    new_shape[idx * 2] = to_add_first
                    new_shape[idx * 2 + 1] = to_add_after

                idx += 1

            return new_shape

        super().__init__(multiple=multiple, px=lambda: amount_to_pad, **kwargs)

# TODO: add resizing by rescaling
