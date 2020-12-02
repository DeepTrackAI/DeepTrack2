""" Features that augment images

Augmentations are features that can resolve more than one image without
calling `.resolve()` on the parent feature. Specifically, they create
`updates_per_reload` images, while calling their parent feature
`load_size` times.

Classes
-------
Augmentation
    Base abstract augmentation class.
PreLoad
    Simple storage with no augmentation.
FlipLR
    Flips images left-right.
FlipUD
    Flips images up-down.
FlipDiagonal
    Flips images diagonally.
"""

from .features import Feature
from .image import Image
from . import utils

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from typing import Callable
import warnings


class Augmentation(Feature):
    """Base abstract augmentation class.

    Augmentations are features that can resolve more than one image without
    calling `.resolve()` on the parent feature. Specifically, they create
    `updates_per_reload` images, while calling their parent feature
    `load_size` times. They achieve this by resolving `load_size` results
    from the parent feature at once, and randomly drawing one of these
    results as input to the method `.get()`. A new input is chosen
    every time `.update()` is called. Once `.update()` has been called
    `updated_per_reload` times, a new batch of `load_size` results are
    resolved from the parent feature.

    The method `.get()` of implementations of this class may accept the
    property `number_of_updates` as an argument. This number represents
    the number of times the `.update()` method has been called since the
    last time the parent feature was resolved.

    Parameters
    ----------
    feature : Feature, optional, deprecated
        DEPRECATED. The parent feature. If not passed, it acts like any other feature.
    load_size : int
        Number of results to resolve from the parent feature.
    updates_per_reload : int
        Number of times `.update()` is called before resolving new results
        from the parent feature.
    update_properties : Callable or None
        Function called on the output of the method `.get()`. Overrides
        the default behaviour, allowing full control over how to update
        the properties of the output to account for the augmentation.
    """

    __distributed__ = False

    def __init__(
        self,
        feature: Feature = None,
        load_size: int = 1,
        updates_per_reload: int = 1,
        update_properties: Callable or None = None,
        **kwargs
    ):

        if load_size != 1:
            warnings.warn(
                "Using an augmentation with a load size other than one is no longer supported",
                DeprecationWarning,
            )

        self.feature = feature

        def get_number_of_updates(updates_per_reload=1):
            # Updates the number of updates. The very first update is not counted.
            if not hasattr(self.properties["number_of_updates"], "_current_value"):
                return 0
            return (
                self.properties["number_of_updates"].current_value + 1
            ) % updates_per_reload

        def tally():
            idx = 0
            while True:
                yield idx
                idx += 1

        if not update_properties:
            update_properties = self.update_properties

        super().__init__(
            load_size=load_size,
            update_tally=tally(),
            updates_per_reload=updates_per_reload,
            index=kwargs.pop("index", False)
            or (lambda load_size: np.random.randint(load_size)),
            number_of_updates=get_number_of_updates,
            update_properties=lambda: update_properties,
            **kwargs
        )

    def _process_and_get(self, *args, update_properties=None, **kwargs):

        # Loads a result from storage
        if self.feature and (
            not hasattr(self, "cache")
            or kwargs["update_tally"] - self.last_update >= kwargs["updates_per_reload"]
        ):

            if isinstance(self.feature, list):
                self.cache = [feature.resolve() for feature in self.feature]
            else:
                self.cache = self.feature.resolve()
            self.last_update = kwargs["update_tally"]

        if not self.feature:
            image_list_of_lists = args[0]
        else:
            image_list_of_lists = self.cache

        if not isinstance(image_list_of_lists, list):
            image_list_of_lists = [image_list_of_lists]

        new_list_of_lists = []
        # Calls get

        np.random.seed(kwargs["hash_key"][0])

        for image_list in image_list_of_lists:
            if isinstance(self.feature, list):
                # If multiple features, ensure consistent rng
                np.random.seed(kwargs["hash_key"][0])

            if isinstance(image_list, list):
                new_list_of_lists.append(
                    [
                        [
                            Image(
                                self.get(Image(image), **kwargs)
                            ).merge_properties_from(image)
                            for image in image_list
                        ]
                    ]
                )
            else:
                # DANGEROUS
                # if not isinstance(image_list, Image):
                image_list = Image(image_list)

                output = self.get(image_list, **kwargs)

                if not isinstance(output, Image):
                    output = Image(output)
                new_list_of_lists.append(output.merge_properties_from(image_list))

        if update_properties:
            if not isinstance(new_list_of_lists, list):
                new_list_of_lists = [new_list_of_lists]
            for image_list in new_list_of_lists:
                if not isinstance(image_list, list):
                    image_list = [image_list]
                for image in image_list:
                    image.properties = [dict(prop) for prop in image.properties]
                    update_properties(image, **kwargs)

        return new_list_of_lists

    def _update(self, **kwargs):
        super()._update(**kwargs)
        if self.feature and not self.number_of_updates.current_value:
            if isinstance(self.feature, Feature):
                self.feature._update(**kwargs)
            elif isinstance(self.feature, list):
                [feature._update(**kwargs) for feature in self.feature]

    def update_properties(*args, **kwargs):
        pass


class PreLoad(Augmentation):
    """Simple storage with no augmentation.

    Parameters
    ----------
    feature : Feature
        The parent feature.
    load_size : int
        Number of results to resolve from the parent feature.
    updates_per_reload : int
        Number of times `.update()` is called before resolving new results
        from the parent feature.
    index : int
        The index of the image to grab from storage. By default a random image.
    update_properties : Callable or None
        Function called on the output of the method `.get()`. Overrides
        the default behaviour, allowing full control over how to update
        the properties of the output to account for the augmentation.

    """

    def get(self, image, **kwargs):
        return image


class FlipLR(Augmentation):
    """Flips images left-right.

    Updates all properties called "position" to flip the second index.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_size=1, updates_per_reload=2, **kwargs)

    def get(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2:
            image = np.fliplr(image)
        return image

    def update_properties(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2:
            for prop in image.properties:
                if "position" in prop:
                    position = prop["position"]
                    new_position = (
                        position[0],
                        image.shape[1] - position[1] - 1,
                        *position[2:],
                    )
                    prop["position"] = new_position


class FlipUD(Augmentation):
    """Flips images up-down.

    Updates all properties called "position" by flipping the first index.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_size=1, updates_per_reload=2, **kwargs)

    def get(self, image, number_of_updates=0, **kwargs):
        if number_of_updates % 2:
            image = np.flipud(image)
        return image

    def update_properties(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2:
            for prop in image.properties:
                if "position" in prop:
                    position = prop["position"]
                    new_position = (
                        image.shape[0] - position[0] - 1,
                        *position[1:],
                    )
                    prop["position"] = new_position


class FlipDiagonal(Augmentation):
    """Flips images along the main diagonal.

    Updates all properties called "position" by swapping the first and second index.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_size=1, updates_per_reload=2, **kwargs)

    def get(self, image, number_of_updates, axes=(1, 0, 2), **kwargs):
        if number_of_updates % 2:
            image = np.transpose(image, axes=axes)
        return image

    def update_properties(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2:
            for prop in image.properties:
                if "position" in prop:
                    position = prop["position"]
                    new_position = (position[1], position[0], *position[2:])
                    prop["position"] = new_position


class Affine(Augmentation):
    """
    Augmenter to apply affine transformations to images.

    Affine transformations involve:

        - Translation
        - Scaling
        - Rotation
        - Shearing

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.


    Parameters
    ----------
    scale : number or tuple of number or list of number or dict {"x": number, "y": number}
        Scaling factor to use, where ``1.0`` denotes "no change" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.
        If two values are provided (using tuple, list, or dict), the two first dimensions of the input are scaled individually.

    translate : number or tuple of number or list of number or dict {"x": number, "y": number}
        Translation in pixels.

    translate_px : number or tuple of number or list of number or dict {"x": number, "y": number}
        DEPRECATED, use translate.

    rotate : number
        Rotation in radians, i.e. Rotation happens around the *center* of the
        image.

    shear : number
        Shear in radians. Values in the range (-pi/4, pi/4) are common


    order : int
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
        scale=1,
        translate=None,
        translate_px=0,
        rotate=0,
        shear=0,
        order=1,
        cval=0,
        mode="reflect",
        **kwargs
    ):
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
            **kwargs
        )

    def _process_properties(self, properties):

        properties = super()._process_properties(properties)

        # Make translate tuple
        translate = properties["translate"]
        if isinstance(translate, (float, int)):
            translate = (translate, translate)
        if isinstance(translate, dict):
            translate = (translate["x"], translate["y"])
        properties["translate"] = translate

        # Make scale tuple
        scale = properties["scale"]
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        if isinstance(scale, dict):
            scale = (scale["x"], scale["y"])
        properties["scale"] = scale

        return properties

    def get(self, image, scale, translate, rotate, shear, **kwargs):

        assert (
            image.ndim == 2 or image.ndim == 3
        ), "Affine only supports 2-dimensional or 3-dimension inputs, got {0}".format(
            image.ndim
        )

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

        # Clean up kwargs
        kwargs.pop("input", False)
        kwargs.pop("matrix", False)
        kwargs.pop("offset", False)
        kwargs.pop("output", False)

        # Call affine_transform
        if image.ndim == 2:
            new_image = utils.safe_call(
                ndimage.affine_transform,
                input=image,
                matrix=mapping,
                offset=d,
                **kwargs
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
                    **kwargs
                )

        # Map positions
        inverse_mapping = np.linalg.inv(mapping)
        for prop in image.properties:
            if "position" in prop:
                position = np.array(prop["position"])
                prop["position"] = np.array(
                    (
                        *(
                            (
                                inverse_mapping
                                @ (position[:2] - center + np.array([dy, dx]))
                                + center
                            )
                        ),
                        *position[3:],
                    )
                )

        return image


class ElasticTransformation(Augmentation):
    """Transform images by moving pixels locally around using displacement fields.

    The augmenter creates a random distortion field using `alpha` and `sigma`, which define
    the strength and smoothness of the field respectively. These are used to transform the
    input locally.

    .. Note:
        This augmentation does not currently update the position property of the image,
        meaning that it is not recommended to use it if the network label is
        derived from the position properties of the resulting image.

    For a detailed explanation, see ::

        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003


    Parameters
    ----------
    alpha : number
        Strength of the distortion field. Common values are in the range (10, 100)

    sigma : number
        Standard deviation of the gaussian kernel used to smooth the distortion
        fields. Common values are in the range (1, 10)

    ignore_last_dim : bool
        Whether to skip creating a distortion field for the last dimension.
        This is often desired if the last dimension is a channel dimension (such as
        a color image.) In that case, the three channels are transformed identically
        and do not `bleed` into eachother.


    order : int
        Interpolation order to use. Takes integers from 0 to 5

            * ``0``: ``Nearest-neighbor``
            * ``1``: ``Bi-linear`` (default)
            * ``2``: ``Bi-quadratic`` (not recommended by skimage)
            * ``3``: ``Bi-cubic``
            * ``4``: ``Bi-quartic``
            * ``5``: ``Bi-quintic``


    cval : number
        The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to ``constant``.

    mode : str
        Parameter that defines newly created pixels.
        May take the same values as in :func:`scipy.ndimage.map_coordinates`,
        i.e. ``constant``, ``nearest``, ``reflect`` or ``wrap``.

    """

    def __init__(
        self,
        alpha=20,
        sigma=2,
        ignore_last_dim=True,
        order=3,
        cval=0,
        mode="constant",
        **kwargs
    ):
        super().__init__(
            alpha=alpha,
            sigma=sigma,
            ignore_last_dim=ignore_last_dim,
            order=order,
            cval=cval,
            mode=mode,
            **kwargs
        )

    def get(self, image, sigma, alpha, ignore_last_dim, **kwargs):

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
            dDim = np.transpose(grid, axes=(1, 0) + tuple(range(2, grid.ndim))) + delta
            coordinates.append(np.reshape(dDim, (-1, 1)))

        if ignore_last_dim:
            for z in range(image.shape[-1]):
                image[..., z] = utils.safe_call(
                    map_coordinates,
                    input=image[..., z],
                    coordinates=coordinates,
                    **kwargs
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
    feature : feature or list of features
        Feature(s) to augment.
    crop : int or tuple of int or list of int or Callable[Image]->tuple of ints
        Number of pixels to remove or retain (depending in `crop_mode`)
        If a tuple or list, it is assumed to be per axis.
        Can also be a function that returns any of the other types.
    crop_mode : str {"remove", "retain"}
        How the `crop` argument is interpreted. If "remove", then
        crop
    corner : tuple of ints or Callable[Image]->tuple of ints or "random"
        Top left corner of the cropped region. Can be a tuple of ints,
        a function that returns a tuple of ints or the string random.
        If corner is placed so that the cropping cannot be performed,
        the modulo of the corner with the allowed region is used.

    """

    def __init__(
        self, *args, crop=(64, 64), crop_mode="retain", corner="random", **kwargs
    ):
        super().__init__(*args, crop=crop, crop_mode=crop_mode, corner=corner, **kwargs)

    def get(self, image, corner, crop, crop_mode, **kwargs):

        # Get crop argument
        if callable(crop):
            crop = crop(image)
        if isinstance(crop, int):
            crop = (crop,) * image.ndim

        crop = [c if c is not None else image.shape[i] for i, c in enumerate(crop)]

        # Get amount to crop from image
        if crop_mode == "retain":
            crop_amount = np.array(image.shape) - np.array(crop)
        elif crop_mode == "remove":
            crop_amount = np.array(crop)
        else:
            raise ValueError("Unrecognized crop_mode {0}".format(crop_mode))

        # Contain within image
        crop_amount = np.amax((np.array(crop_amount), [0] * image.ndim), axis=0)
        crop_amount = np.amin((np.array(image.shape) - 1, crop_amount), axis=0)
        # Get corner of crop
        if isinstance(corner, str) and corner == "random":
            # Ensure seed is consistent
            slice_start = [np.random.randint(m + 1) for m in crop_amount]
        elif callable(corner):
            slice_start = corner(image)
        else:
            slice_start = corner

        # Ensure compatible with image
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

        # Update positions
        cropped_image.properties = [dict(prop) for prop in image.properties]
        for prop in cropped_image.properties:
            if "position" in prop:
                position = np.array(prop["position"])
                try:
                    position[0:2] -= np.array(slice_start)[0:2]
                    prop["position"] = position
                except IndexError:
                    pass

        return cropped_image


class CropToMultiplesOf(Crop):
    """Crop images down until their height/width is a multiple of a value.

    Parameters
    ----------
    multiple : int or tuple of (int or None)
        Images will be cropped down until their width is a multiple of
        this value. If a tuple, it is assumed to be a multiple per axis.
        A value of None or -1 indicates to skip that axis.

    """

    def __init__(self, multiple=1, corner="random", **kwargs):
        kwargs.pop("crop", False)
        kwargs.pop("crop_mode", False)

        def image_to_crop(image):
            shape = image.shape
            multiple = self.multiple.current_value

            if not isinstance(multiple, (list, tuple, np.ndarray)):
                multiple = (multiple,) * image.ndim
            new_shape = list(shape)
            idx = 0
            for dim, mul in zip(shape, multiple):
                if mul is not None and mul is not -1:
                    new_shape[idx] = int((dim // mul) * mul)
                idx += 1

            return new_shape

        super().__init__(
            multiple=multiple,
            corner=corner,
            crop=lambda: image_to_crop,
            crop_mode="retain",
            **kwargs
        )


class Pad(Augmentation):
    """Pads the image.

    Arguments match this of numpy.pad, save for pad_width, which is called px,
    and is defined as (left, right, up, down, before_axis_3, after_axis_3, ...)

    Parameters
    ----------
    px : int or list of int
        amount to pad in each direction

    """

    def __init__(self, px=(0, 0, 0, 0), mode="constant", cval=0, **kwargs):
        super().__init__(px=px, mode=mode, cval=cval, **kwargs)

    def get(self, image, px, **kwargs):

        padding = []
        if callable(px):
            px = px(image)
        elif isinstance(px, int):
            padding = [(px, px)] * image.ndom

        for idx in range(0, len(px), 2):
            padding.append((px[idx], px[idx + 1]))

        while len(padding) < image.ndim:
            padding.append((0, 0))

        return (
            utils.safe_call(np.pad, positional_args=(image, padding), **kwargs),
            padding,
        )

    def _process_and_get(self, images, **kwargs):
        results = [self.get(image, **kwargs) for image in images]
        for idx, result in enumerate(results):
            if isinstance(result, tuple):
                shape = result[0].shape
                padding = result[1]
                de_pad = tuple(
                    slice(p[0], shape[dim] - p[1]) for dim, p in enumerate(padding)
                )
                results[idx] = (
                    Image(result[0]).merge_properties_from(images[idx]),
                    {"undo_padding": de_pad},
                )
            else:
                Image(results[idx]).merge_properties_from(images[idx])
        return results


class PadToMultiplesOf(Pad):
    """Pad images until their height/width is a multiple of a value.

    Parameters
    ----------
    multiple : int or tuple of (int or None)
        Images will be padded until their width is a multiple of
        this value. If a tuple, it is assumed to be a multiple per axis.
        A value of None or -1 indicates to skip that axis.

    """

    def __init__(self, multiple=1, **kwargs):
        def amount_to_pad(image):
            shape = image.shape
            multiple = self.multiple.current_value

            if not isinstance(multiple, (list, tuple, np.ndarray)):
                multiple = (multiple,) * image.ndim
            new_shape = [0] * (image.ndim * 2)
            idx = 0
            for dim, mul in zip(shape, multiple):
                if mul is not None and mul is not -1:
                    to_add = -dim % mul
                    to_add_first = to_add // 2
                    to_add_after = to_add - to_add_first
                    new_shape[idx * 2] = to_add_first
                    new_shape[idx * 2 + 1] = to_add_after

                idx += 1

            return new_shape

        super().__init__(multiple=multiple, px=lambda: amount_to_pad, **kwargs)


# TODO: add resizing by rescaling
