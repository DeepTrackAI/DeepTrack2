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

from deeptrack.features import Feature
from deeptrack.image import Image
import numpy as np
from typing import Callable
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
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

        if feature is not None:
            warnings.warn(
                "Calling an augmentation with a feature is deprecated in a future release. Instead, just use the + operator.",
                DeprecationWarning,
            )

        if load_size is not 1:
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
                new_list_of_lists.append(
                    Image(self.get(Image(image_list), **kwargs)).merge_properties_from(
                        image_list
                    )
                )

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


class Crop(Augmentation):
    """Crops a regions of an image.

    Parameters
    ----------
    feature : feature or list of features
        Feature(s) to augment.
    corner : tuple of ints or Callable[Image] or "random"
        Top left corner of the cropped region. Can be a tuple of ints,

    """

    def __init__(self, *args, corner="random", crop_size=(64, 64), **kwargs):
        super().__init__(*args, corner=corner, crop_size=crop_size, **kwargs)

    def get(self, image, corner, crop_size, **kwargs):
        if corner == "random":
            # Ensure seed is consistent
            slice_start = np.random.randint(
                [0] * len(crop_size),
                np.array(image.shape[: len(crop_size)]) - crop_size,
            )

        elif callable(corner):
            slice_start = corner(image)

        else:
            slice_start = corner

        slices = tuple(
            [
                slice(slice_start_i, slice_start_i + crop_size_i)
                for slice_start_i, crop_size_i in zip(slice_start, crop_size)
            ]
        )

        cropped_image = image[slices]

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
                    new_position = (image.shape[0] - position[0] - 1, *position[1:])
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


from deeptrack.utils import get_kwarg_names
import imgaug.augmenters as iaa
import imgaug.imgaug as ia
import warnings


class ImgAug(Augmentation):
    """Interfaces imagaug augmentations."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def get(self, image, **kwargs):
        argument_names = get_kwarg_names(self.augmenter)
        class_options = {}
        for key in argument_names:
            if key in kwargs:
                class_options[key] = kwargs[key]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ia.seed(entropy=np.random.randint(2 ** 31 - 1))
            return self.augmenter(**class_options)(image=image)


## IMGAUG GEOMETRIC
# Please see https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/geometric.py
# for source implementation


class Affine(ImgAug):
    """
    Augmenter to apply affine transformations to images.

    This is mostly a wrapper around the corresponding classes and functions
    in OpenCV and skimage.

    Affine transformations involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a
    defined content, e.g. if the image is translated to the left, pixels
    are created on the right.
    A method has to be defined to deal with these pixel values. The
    parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.

    .. note::

        While this augmenter supports segmentation maps and heatmaps that
        have a different size than the corresponding image, it is strongly
        recommended to use the same aspect ratios. E.g. for an image of
        shape ``(200, 100, 3)``, good segmap/heatmap array shapes also follow
        a ``2:1`` ratio and ideally are ``(200, 100, C)``, ``(100, 50, C)`` or
        ``(50, 25, C)``. Otherwise, transformations involving rotations or
        shearing will produce unaligned outputs.
        For performance reasons, there is no explicit validation of whether
        the aspect ratios are similar.

    **Supported dtypes**:

    if (backend="skimage", order in [0, 1]):

        * ``uint8``: yes; tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested  (1)
        * ``int64``: no (2)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (2)
        * ``bool``: yes; tested

        - (1) scikit-image converts internally to float64, which might
              affect the accuracy of large integers. In tests this seemed
              to not be an issue.
        - (2) results too inaccurate

    if (backend="skimage", order in [3, 4]):

        * ``uint8``: yes; tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested  (1)
        * ``int64``: no (2)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: limited; tested (3)
        * ``float128``: no (2)
        * ``bool``: yes; tested

        - (1) scikit-image converts internally to float64, which might
              affect the accuracy of large integers. In tests this seemed
              to not be an issue.
        - (2) results too inaccurate
        - (3) ``NaN`` around minimum and maximum of float64 value range

    if (backend="skimage", order=5]):

            * ``uint8``: yes; tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested  (1)
            * ``int64``: no (2)
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: limited; not tested (3)
            * ``float128``: no (2)
            * ``bool``: yes; tested

            - (1) scikit-image converts internally to ``float64``, which
                  might affect the accuracy of large integers. In tests
                  this seemed to not be an issue.
            - (2) results too inaccurate
            - (3) ``NaN`` around minimum and maximum of float64 value range

    if (backend="cv2", order=0):

        * ``uint8``: yes; tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: no (2)
        * ``float16``: yes; tested (3)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (3)

        - (1) rejected by cv2
        - (2) changed to ``int32`` by cv2
        - (3) mapped internally to ``float32``

    if (backend="cv2", order=1):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (4)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (4)

        - (1) rejected by cv2
        - (2) causes cv2 error: ``cv2.error: OpenCV(3.4.4)
              (...)imgwarp.cpp:1805: error:
              (-215:Assertion failed) ifunc != 0 in function 'remap'``
        - (3) mapped internally to ``int16``
        - (4) mapped internally to ``float32``

    if (backend="cv2", order=3):

        * ``uint8``: yes; tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (4)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (4)

        - (1) rejected by cv2
        - (2) causes cv2 error: ``cv2.error: OpenCV(3.4.4)
              (...)imgwarp.cpp:1805: error:
              (-215:Assertion failed) ifunc != 0 in function 'remap'``
        - (3) mapped internally to ``int16``
        - (4) mapped internally to ``float32``


    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Scaling factor to use, where ``1.0`` denotes "no change" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.

            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That value will be
              used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_percent : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Translation as a fraction of the image height/width (x-translation,
        y-translation), where ``0`` denotes "no change" and ``0.5`` denotes
        "half of the axis size".

            * If ``None`` then equivalent to ``0.0`` unless `translate_px` has
              a value other than ``None``.
            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That sampled fraction
              value will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_px : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Translation in pixels.

            * If ``None`` then equivalent to ``0`` unless `translate_percent`
              has a value other than ``None``.
            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``. That number
              will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Rotation in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``. Rotation happens around the *center* of the
        image, not the top left corner as in some other frameworks.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and used as the rotation
              value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the rotation value per image.

    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Shear in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``, with reasonable values being in the range
        of ``[-45, 45]``.

            * If a number, then that value will be used for all images as
              the shear on the x-axis (no shear on the y-axis will be done).
            * If a tuple ``(a, b)``, then two value will be uniformly sampled
              per image from the interval ``[a, b]`` and be used as the
              x- and y-shear value.
            * If a list, then two random values will be sampled from that list
              per image, denoting x- and y-shear.
            * If a ``StochasticParameter``, then this parameter will be used
              to sample the x- and y-shear values per image.
            * If a dictionary, then similar to `translate_percent`, i.e. one
              ``x`` key and/or one ``y`` key are expected, denoting the
              shearing on the x- and y-axis respectively. The allowed datatypes
              are again ``number``, ``tuple`` ``(a, b)``, ``list`` or
              ``StochasticParameter``.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Same meaning as in ``skimage``:

            * ``0``: ``Nearest-neighbor``
            * ``1``: ``Bi-linear`` (default)
            * ``2``: ``Bi-quadratic`` (not recommended by skimage)
            * ``3``: ``Bi-cubic``
            * ``4``: ``Bi-quartic``
            * ``5``: ``Bi-quintic``

        Method ``0`` and ``1`` are fast, ``3`` is a bit slower, ``4`` and
        ``5`` are very slow. If the backend is ``cv2``, the mapping to
        OpenCV's interpolation modes is as follows:

            * ``0`` -> ``cv2.INTER_NEAREST``
            * ``1`` -> ``cv2.INTER_LINEAR``
            * ``2`` -> ``cv2.INTER_CUBIC``
            * ``3`` -> ``cv2.INTER_CUBIC``
            * ``4`` -> ``cv2.INTER_CUBIC``

        As datatypes this parameter accepts:

            * If a single ``int``, then that order will be used for all images.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL``, then equivalant to list ``[0, 1, 3, 4, 5]``
              in case of ``backend=skimage`` and otherwise ``[0, 1, 3]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the order value to use.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        (E.g. translating by 1px to the right will create a new 1px-wide
        column of pixels on the left of the image).  The value is only used
        when `mode=constant`. The expected value range is ``[0, 255]`` for
        ``uint8`` images. It may be a float value.

            * If this is a single number, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then three values (for three image
              channels) will be uniformly sampled per image from the
              interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL`` then equivalent to tuple ``(0, 255)`.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Method to use when filling in newly created pixels.
        Same meaning as in ``skimage`` (and :func:`numpy.pad`):

            * ``constant``: Pads with a constant value
            * ``edge``: Pads with the edge values of array
            * ``symmetric``: Pads with the reflection of the vector mirrored
              along the edge of the array.
            * ``reflect``: Pads with the reflection of the vector mirrored on
              the first and last values of the vector along each axis.
            * ``wrap``: Pads with the wrap of the vector along the axis.
              The first values are used to pad the end and the end values
              are used to pad the beginning.

        If ``cv2`` is chosen as the backend the mapping is as follows:

            * ``constant`` -> ``cv2.BORDER_CONSTANT``
            * ``edge`` -> ``cv2.BORDER_REPLICATE``
            * ``symmetric`` -> ``cv2.BORDER_REFLECT``
            * ``reflect`` -> ``cv2.BORDER_REFLECT_101``
            * ``wrap`` -> ``cv2.BORDER_WRAP``

        The datatype of the parameter may be:

            * If a single string, then that mode will be used for all images.
            * If a list of strings, then a random mode will be picked
              from that list per image.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    fit_output : bool, optional
        Whether to modify the affine transformation so that the whole output
        image is always contained in the image plane (``True``) or accept
        parts of the image being outside the image plane (``False``).
        This can be thought of as first applying the affine transformation
        and then applying a second transformation to "zoom in" on the new
        image so that it fits the image plane,
        This is useful to avoid corners of the image being outside of the image
        plane after applying rotations. It will however negate translation
        and scaling.
        Note also that activating this may lead to image sizes differing from
        the input image sizes. To avoid this, wrap ``Affine`` in
        :class:`~imgaug.augmenters.size.KeepSizeByResize`,
        e.g. ``KeepSizeByResize(Affine(...))``.

    backend : str, optional
        Framework to use as a backend. Valid values are ``auto``, ``skimage``
        (scikit-image's warp) and ``cv2`` (OpenCV's warp).
        If ``auto`` is used, the augmenter will automatically try
        to use ``cv2`` whenever possible (order must be in ``[0, 1, 3]``). It
        will silently fall back to skimage if order/dtype is not supported by
        cv2. cv2 is generally faster than skimage. It also supports RGB cvals,
        while skimage will resort to intensity cvals (i.e. 3x the same value
        as RGB). If ``cv2`` is chosen and order is ``2`` or ``4``, it will
        automatically fall back to order ``3``.

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
    >>> aug = iaa.Affine(scale=2.0)

    Zoom in on all images by a factor of ``2``.

    >>> aug = iaa.Affine(translate_px=16)

    Translate all images on the x- and y-axis by 16 pixels (towards the
    bottom right) and fill up any new pixels with zero (black values).

    >>> aug = iaa.Affine(translate_percent=0.1)

    Translate all images on the x- and y-axis by ``10`` percent of their
    width/height (towards the bottom right). The pixel values are computed
    per axis based on that axis' size. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(rotate=35)

    Rotate all images by ``35`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(shear=15)

    Shear all images by ``15`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(translate_px=(-16, 16))

    Translate all images on the x- and y-axis by a random value
    between ``-16`` and ``16`` pixels (to the bottom right) and fill up any new
    pixels with zero (black values). The translation value is sampled once
    per image and is the same for both axis.

    >>> aug = iaa.Affine(translate_px={"x": (-16, 16), "y": (-4, 4)})

    Translate all images on the x-axis by a random value
    between ``-16`` and ``16`` pixels (to the right) and on the y-axis by a
    random value between ``-4`` and ``4`` pixels to the bottom. The sampling
    happens independently per axis, so even if both intervals were identical,
    the sampled axis-wise values would likely be different.
    This also fills up any new pixels with zero (black values).

    >>> aug = iaa.Affine(scale=2.0, order=[0, 1])

    Same as in the above `scale` example, but uses (randomly) either
    nearest neighbour interpolation or linear interpolation. If `order` is
    not specified, ``order=1`` would be used by default.

    >>> aug = iaa.Affine(translate_px=16, cval=(0, 255))

    Same as in the `translate_px` example above, but newly created pixels
    are now filled with a random color (sampled once per image and the
    same for all newly created pixels within that image).

    >>> aug = iaa.Affine(translate_px=16, mode=["constant", "edge"])

    Similar to the previous example, but the newly created pixels are
    filled with black pixels in half of all images (mode ``constant`` with
    default `cval` being ``0``) and in the other half of all images using
    ``edge`` mode, which repeats the color of the spatially closest pixel
    of the corresponding image edge.

    >>> aug = iaa.Affine(shear={"y": (-45, 45)})

    Shear images only on the y-axis. Set `shear` to ``shear=(-45, 45)`` to
    shear randomly on both axes, using for each image the same sample for
    both the x- and y-axis. Use ``shear={"x": (-45, 45), "y": (-45, 45)}``
    to get independent samples per axis.

    """

    def __init__(
        self,
        scale=None,
        translate_percent=None,
        translate_px=None,
        rotate=None,
        shear=None,
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.Affine
        super().__init__(
            scale=scale,
            translate_percent=translate_percent,
            translate_px=translate_px,
            rotate=rotate,
            shear=shear,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class AffineCv2(ImgAug):
    """
    **Deprecated.** Augmenter to apply affine transformations to images using
    cv2 (i.e. opencv) backend.

    .. warning::

        This augmenter is deprecated since 0.4.0.
        Use ``Affine(..., backend='cv2')`` instead.

    Affine transformations
    involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a
    defined content, e.g. if the image is translated to the left, pixels
    are created on the right.
    A method has to be defined to deal with these pixel values. The
    parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.

    Deprecated since 0.4.0.

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
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Scaling factor to use, where ``1.0`` denotes "no change" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.

            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That value will be
              used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_percent : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Translation as a fraction of the image height/width (x-translation,
        y-translation), where ``0`` denotes "no change" and ``0.5`` denotes
        "half of the axis size".

            * If ``None`` then equivalent to ``0.0`` unless `translate_px` has
              a value other than ``None``.
            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That sampled fraction
              value will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Translation in pixels.

            * If ``None`` then equivalent to ``0`` unless `translate_percent`
              has a value other than ``None``.
            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``. That number
              will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Rotation in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``. Rotation happens around the *center* of the
        image, not the top left corner as in some other frameworks.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and used as the rotation
              value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the rotation value per image.

    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Shear in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and be used as the
              rotation value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used
              to sample the shear value per image.

    order : int or list of int or str or list of str or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Allowed are:

            * ``cv2.INTER_NEAREST`` (nearest-neighbor interpolation)
            * ``cv2.INTER_LINEAR`` (bilinear interpolation, used by default)
            * ``cv2.INTER_CUBIC`` (bicubic interpolation over ``4x4`` pixel
                neighborhood)
            * ``cv2.INTER_LANCZOS4``
            * string ``nearest`` (same as ``cv2.INTER_NEAREST``)
            * string ``linear`` (same as ``cv2.INTER_LINEAR``)
            * string ``cubic`` (same as ``cv2.INTER_CUBIC``)
            * string ``lanczos4`` (same as ``cv2.INTER_LANCZOS``)

        ``INTER_NEAREST`` (nearest neighbour interpolation) and
        ``INTER_NEAREST`` (linear interpolation) are the fastest.

            * If a single ``int``, then that order will be used for all images.
            * If a string, then it must be one of: ``nearest``, ``linear``,
              ``cubic``, ``lanczos4``.
            * If an iterable of ``int``/``str``, then for each image a random
              value will be sampled from that iterable (i.e. list of allowed
              order values).
            * If ``imgaug.ALL``, then equivalant to list ``[cv2.INTER_NEAREST,
              cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the order value to use.

    cval : number or tuple of number or list of number or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        (E.g. translating by 1px to the right will create a new 1px-wide
        column of pixels on the left of the image).  The value is only used
        when `mode=constant`. The expected value range is ``[0, 255]`` for
        ``uint8`` images. It may be a float value.

            * If this is a single number, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then three values (for three image
              channels) will be uniformly sampled per image from the
              interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL`` then equivalent to tuple ``(0, 255)`.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : int or str or list of str or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter,
           optional
        Method to use when filling in newly created pixels.
        Same meaning as in OpenCV's border mode. Let ``abcdefgh`` be an image's
        content and ``|`` be an image boundary after which new pixels are
        filled in, then the valid modes and their behaviour are the following:

            * ``cv2.BORDER_REPLICATE``: ``aaaaaa|abcdefgh|hhhhhhh``
            * ``cv2.BORDER_REFLECT``: ``fedcba|abcdefgh|hgfedcb``
            * ``cv2.BORDER_REFLECT_101``: ``gfedcb|abcdefgh|gfedcba``
            * ``cv2.BORDER_WRAP``: ``cdefgh|abcdefgh|abcdefg``
            * ``cv2.BORDER_CONSTANT``: ``iiiiii|abcdefgh|iiiiiii``,
               where ``i`` is the defined cval.
            * ``replicate``: Same as ``cv2.BORDER_REPLICATE``.
            * ``reflect``: Same as ``cv2.BORDER_REFLECT``.
            * ``reflect_101``: Same as ``cv2.BORDER_REFLECT_101``.
            * ``wrap``: Same as ``cv2.BORDER_WRAP``.
            * ``constant``: Same as ``cv2.BORDER_CONSTANT``.

        The datatype of the parameter may be:

            * If a single ``int``, then it must be one of the ``cv2.BORDER_*``
              constants.
            * If a single string, then it must be one of: ``replicate``,
              ``reflect``, ``reflect_101``, ``wrap``, ``constant``.
            * If a list of ``int``/``str``, then per image a random mode will
              be picked from that list.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

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
    >>> aug = iaa.AffineCv2(scale=2.0)

    Zoom in on all images by a factor of ``2``.

    >>> aug = iaa.AffineCv2(translate_px=16)

    Translate all images on the x- and y-axis by 16 pixels (towards the
    bottom right) and fill up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(translate_percent=0.1)

    Translate all images on the x- and y-axis by ``10`` percent of their
    width/height (towards the bottom right). The pixel values are computed
    per axis based on that axis' size. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(rotate=35)

    Rotate all images by ``35`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(shear=15)

    Shear all images by ``15`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(translate_px=(-16, 16))

    Translate all images on the x- and y-axis by a random value
    between ``-16`` and ``16`` pixels (to the bottom right) and fill up any new
    pixels with zero (black values). The translation value is sampled once
    per image and is the same for both axis.

    >>> aug = iaa.AffineCv2(translate_px={"x": (-16, 16), "y": (-4, 4)})

    Translate all images on the x-axis by a random value
    between ``-16`` and ``16`` pixels (to the right) and on the y-axis by a
    random value between ``-4`` and ``4`` pixels to the bottom. The sampling
    happens independently per axis, so even if both intervals were identical,
    the sampled axis-wise values would likely be different.
    This also fills up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(scale=2.0, order=[0, 1])

    Same as in the above `scale` example, but uses (randomly) either
    nearest neighbour interpolation or linear interpolation. If `order` is
    not specified, ``order=1`` would be used by default.

    >>> aug = iaa.AffineCv2(translate_px=16, cval=(0, 255))

    Same as in the `translate_px` example above, but newly created pixels
    are now filled with a random color (sampled once per image and the
    same for all newly created pixels within that image).

    >>> aug = iaa.AffineCv2(translate_px=16, mode=["constant", "replicate"])

    Similar to the previous example, but the newly created pixels are
    filled with black pixels in half of all images (mode ``constant`` with
    default `cval` being ``0``) and in the other half of all images using
    ``replicate`` mode, which repeats the color of the spatially closest pixel
    of the corresponding image edge.

    """

    def __init__(
        self,
        scale=1.0,
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=0.0,
        order=1,
        cval=0,
        mode=0,
        **kwargs
    ):
        self.augmenter = iaa.AffineCv2
        super().__init__(
            scale=scale,
            translate_percent=translate_percent,
            translate_px=translate_px,
            rotate=rotate,
            shear=shear,
            order=order,
            cval=cval,
            mode=mode,
            **kwargs
        )


class ElasticTransformation(ImgAug):
    """
    Transform images by moving pixels locally around using displacement fields.

    The augmenter has the parameters ``alpha`` and ``sigma``. ``alpha``
    controls the strength of the displacement: higher values mean that pixels
    are moved further. ``sigma`` controls the smoothness of the displacement:
    higher values lead to smoother patterns -- as if the image was below water
    -- while low values will cause indivdual pixels to be moved very
    differently from their neighbours, leading to noisy and pixelated images.

    A relation of 10:1 seems to be good for ``alpha`` and ``sigma``, e.g.
    ``alpha=10`` and ``sigma=1`` or ``alpha=50``, ``sigma=5``. For ``128x128``
    a setting of ``alpha=(0, 70.0)``, ``sigma=(4.0, 6.0)`` may be a good
    choice and will lead to a water-like effect.

    Code here was initially inspired by
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

    For a detailed explanation, see ::

        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003

    .. note::

        For coordinate-based inputs (keypoints, bounding boxes, polygons,
        ...), this augmenter still has to perform an image-based augmentation,
        which will make it significantly slower for such inputs than other
        augmenters. See :ref:`performance`.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested (1)
        * ``uint16``: yes; tested (1)
        * ``uint32``: yes; tested (2)
        * ``uint64``: limited; tested (3)
        * ``int8``: yes; tested (1) (4) (5)
        * ``int16``: yes; tested (4) (6)
        * ``int32``: yes; tested (4) (6)
        * ``int64``: limited; tested (3)
        * ``float16``: yes; tested (1)
        * ``float32``: yes; tested (1)
        * ``float64``: yes; tested (1)
        * ``float128``: no
        * ``bool``: yes; tested (1) (7)

        - (1) Always handled by ``cv2``.
        - (2) Always handled by ``scipy``.
        - (3) Only supported for ``order != 0``. Will fail for ``order=0``.
        - (4) Mapped internally to ``float64`` when ``order=1``.
        - (5) Mapped internally to ``int16`` when ``order>=2``.
        - (6) Handled by ``cv2`` when ``order=0`` or ``order=1``, otherwise by
              ``scipy``.
        - (7) Mapped internally to ``float32``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the distortion field. Higher values mean that pixels are
        moved further with respect to the distortion field's direction. Set
        this to around 10 times the value of `sigma` for visible effects.

            * If number, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    sigma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the gaussian kernel used to smooth the distortion
        fields. Higher values (for ``128x128`` images around 5.0) lead to more
        water-like effects, while lower values (for ``128x128`` images
        around ``1.0`` and lower) lead to more noisy, pixelated images. Set
        this to around 1/10th of `alpha` for visible effects.

            * If number, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    order : int or list of int or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Same meaning as in
        :func:`scipy.ndimage.map_coordinates` and may take any integer value
        in the range ``0`` to ``5``, where orders close to ``0`` are faster.

            * If a single int, then that order will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``imgaug.ALL``, then equivalant to list
              ``[0, 1, 2, 3, 4, 5]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the order value to use.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to ``constant``.
        For standard ``uint8`` images (value range ``0`` to ``255``), this
        value may also should also be in the range ``0`` to ``255``. It may
        be a ``float`` value, even for images with integer dtypes.

            * If this is a single number, then that value will be used
              (e.g. ``0`` results in black pixels).
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then a random value will be picked from that list per
              image.
            * If ``imgaug.ALL``, a value from the discrete range ``[0..255]``
              will be sampled per image.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Parameter that defines the handling of newly created pixels.
        May take the same values as in :func:`scipy.ndimage.map_coordinates`,
        i.e. ``constant``, ``nearest``, ``reflect`` or ``wrap``.

            * If a single string, then that mode will be used for all images.
            * If a list of strings, then per image a random mode will be picked
              from that list.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    polygon_recoverer : 'auto' or None or imgaug.augmentables.polygons._ConcavePolygonRecoverer, optional
        The class to use to repair invalid polygons.
        If ``"auto"``, a new instance of
        :class`imgaug.augmentables.polygons._ConcavePolygonRecoverer`
        will be created.
        If ``None``, no polygon recoverer will be used.
        If an object, then that object will be used and must provide a
        ``recover_from()`` method, similar to
        :class:`~imgaug.augmentables.polygons._ConcavePolygonRecoverer`.

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
    >>> aug = iaa.ElasticTransformation(alpha=50.0, sigma=5.0)

    Apply elastic transformations with a strength/alpha of ``50.0`` and
    smoothness of ``5.0`` to all images.

    >>> aug = iaa.ElasticTransformation(alpha=(0.0, 70.0), sigma=5.0)

    Apply elastic transformations with a strength/alpha that comes
    from the interval ``[0.0, 70.0]`` (randomly picked per image) and
    with a smoothness of ``5.0``.

    """

    def __init__(
        self,
        alpha=(0.0, 40.0),
        sigma=(4.0, 8.0),
        order=3,
        cval=0,
        mode="constant",
        polygon_recoverer="auto",
        **kwargs
    ):
        self.augmenter = iaa.ElasticTransformation
        super().__init__(
            alpha=alpha,
            sigma=sigma,
            order=order,
            cval=cval,
            mode=mode,
            polygon_recoverer=polygon_recoverer,
            **kwargs
        )


class Jigsaw(ImgAug):
    """Move cells within images similar to jigsaw patterns.

    .. note::

        This augmenter will by default pad images until their height is a
        multiple of `nb_rows`. Analogous for `nb_cols`.

    .. note::

        This augmenter will resize heatmaps and segmentation maps to the
        image size, then apply similar padding as for the corresponding images
        and resize back to the original map size. That also means that images
        may change in shape (due to padding), but heatmaps/segmaps will not
        change. For heatmaps/segmaps, this deviates from pad augmenters that
        will change images and heatmaps/segmaps in corresponding ways and then
        keep the heatmaps/segmaps at the new size.

    .. warning::

        This augmenter currently only supports augmentation of images,
        heatmaps, segmentation maps and keypoints. Other augmentables,
        i.e. bounding boxes, polygons and line strings, will result in errors.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.geometric.apply_jigsaw`.

    Parameters
    ----------
    nb_rows : int or list of int or tuple of int or imgaug.parameters.StochasticParameter, optional
        How many rows the jigsaw pattern should have.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    nb_cols : int or list of int or tuple of int or imgaug.parameters.StochasticParameter, optional
        How many cols the jigsaw pattern should have.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    max_steps : int or list of int or tuple of int or imgaug.parameters.StochasticParameter, optional
        How many steps each jigsaw cell may be moved.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    allow_pad : bool, optional
        Whether to allow automatically padding images until they are evenly
        divisible by ``nb_rows`` and ``nb_cols``.

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
    >>> aug = iaa.Jigsaw(nb_rows=10, nb_cols=10)

    Create a jigsaw augmenter that splits images into ``10x10`` cells
    and shifts them around by ``0`` to ``2`` steps (default setting).

    >>> aug = iaa.Jigsaw(nb_rows=(1, 4), nb_cols=(1, 4))

    Create a jigsaw augmenter that splits each image into ``1`` to ``4``
    cells along each axis.

    >>> aug = iaa.Jigsaw(nb_rows=10, nb_cols=10, max_steps=(1, 5))

    Create a jigsaw augmenter that moves the cells in each image by a random
    amount between ``1`` and ``5`` times (decided per image). Some images will
    be barely changed, some will be fairly distorted.

    """

    def __init__(
        self, nb_rows=(3, 10), nb_cols=(3, 10), max_steps=1, allow_pad=True, **kwargs
    ):
        self.augmenter = iaa.Jigsaw
        super().__init__(
            nb_rows=nb_rows,
            nb_cols=nb_cols,
            max_steps=max_steps,
            allow_pad=allow_pad,
            **kwargs
        )


class PerspectiveTransform(ImgAug):
    """
    Apply random four point perspective transformations to images.

    Each of the four points is placed on the image using a random distance from
    its respective corner. The distance is sampled from a normal distribution.
    As a result, most transformations don't change the image very much, while
    some "focus" on polygons far inside the image.

    The results of this augmenter have some similarity with ``Crop``.

    Code partially from
    http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    **Supported dtypes**:

    if (keep_size=False):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (4)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (4)

        - (1) rejected by opencv
        - (2) leads to opencv error: cv2.error: ``OpenCV(3.4.4)
              (...)imgwarp.cpp:1805: error: (-215:Assertion failed)
              ifunc != 0 in function 'remap'``.
        - (3) mapped internally to ``int16``.
        - (4) mapped intenally to ``float32``.

    if (keep_size=True):

        minimum of (
            ``imgaug.augmenters.geometric.PerspectiveTransform(keep_size=False)``,
            :func:`~imgaug.imgaug.imresize_many_images`
        )

    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the normal distributions. These are used to
        sample the random distances of the subimage's corners from the full
        image's corners. The sampled values reflect percentage values (with
        respect to image height/width). Recommended values are in the range
        ``0.0`` to ``0.1``.

            * If a single number, then that value will always be used as the
              scale.
            * If a tuple ``(a, b)`` of numbers, then a random value will be
              uniformly sampled per image from the interval ``(a, b)``.
            * If a list of values, a random value will be picked from the
              list per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    keep_size : bool, optional
        Whether to resize image's back to their original size after applying
        the perspective transform. If set to ``False``, the resulting images
        may end up having different shapes and will always be a list, never
        an array.

    cval : number or tuple of number or list of number or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value used to fill up pixels in the result image that
        didn't exist in the input image (e.g. when translating to the left,
        some new pixels are created at the right). Such a fill-up with a
        constant value only happens, when `mode` is ``constant``.
        The expected value range is ``[0, 255]`` for ``uint8`` images.
        It may be a float value.

            * If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then a random value is uniformly sampled
              per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL``, then equivalent to tuple ``(0, 255)``.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : int or str or list of str or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Parameter that defines the handling of newly created pixels.
        Same meaning as in OpenCV's border mode. Let ``abcdefgh`` be an image's
        content and ``|`` be an image boundary, then:

            * ``cv2.BORDER_REPLICATE``: ``aaaaaa|abcdefgh|hhhhhhh``
            * ``cv2.BORDER_CONSTANT``: ``iiiiii|abcdefgh|iiiiiii``, where
              ``i`` is the defined cval.
            * ``replicate``: Same as ``cv2.BORDER_REPLICATE``.
            * ``constant``: Same as ``cv2.BORDER_CONSTANT``.

        The datatype of the parameter may be:

            * If a single ``int``, then it must be one of ``cv2.BORDER_*``.
            * If a single string, then it must be one of: ``replicate``,
              ``reflect``, ``reflect_101``, ``wrap``, ``constant``.
            * If a list of ints/strings, then per image a random mode will be
              picked from that list.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked per image.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    fit_output : bool, optional
        If ``True``, the image plane size and position will be adjusted
        to still capture the whole image after perspective transformation.
        (Followed by image resizing if `keep_size` is set to ``True``.)
        Otherwise, parts of the transformed image may be outside of the image
        plane.
        This setting should not be set to ``True`` when using large `scale`
        values as it could lead to very large images.

        Added in 0.4.0.

    polygon_recoverer : 'auto' or None or imgaug.augmentables.polygons._ConcavePolygonRecoverer, optional
        The class to use to repair invalid polygons.
        If ``"auto"``, a new instance of
        :class`imgaug.augmentables.polygons._ConcavePolygonRecoverer`
        will be created.
        If ``None``, no polygon recoverer will be used.
        If an object, then that object will be used and must provide a
        ``recover_from()`` method, similar to
        :class:`~imgaug.augmentables.polygons._ConcavePolygonRecoverer`.

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
    >>> aug = iaa.PerspectiveTransform(scale=(0.01, 0.15))

    Apply perspective transformations using a random scale between ``0.01``
    and ``0.15`` per image, where the scale is roughly a measure of how far
    the perspective transformation's corner points may be distanced from the
    image's corner points. Higher scale values lead to stronger "zoom-in"
    effects (and thereby stronger distortions).

    >>> aug = iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=False)

    Same as in the previous example, but images are not resized back to
    the input image size after augmentation. This will lead to smaller
    output images.

    """

    def __init__(
        self,
        scale=(0.0, 0.06),
        cval=0,
        mode="constant",
        keep_size=True,
        fit_output=False,
        polygon_recoverer="auto",
        **kwargs
    ):
        self.augmenter = iaa.PerspectiveTransform
        super().__init__(
            scale=scale,
            cval=cval,
            mode=mode,
            keep_size=keep_size,
            fit_output=fit_output,
            polygon_recoverer=polygon_recoverer,
            **kwargs
        )


class PiecewiseAffine(ImgAug):
    """
    Apply affine transformations that differ between local neighbourhoods.

    This augmenter places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.
    This leads to local distortions.

    This is mostly a wrapper around scikit-image's ``PiecewiseAffine``.
    See also ``Affine`` for a similar technique.

    .. note::

        This augmenter is very slow. See :ref:`performance`.
        Try to use ``ElasticTransformation`` instead, which is at least 10x
        faster.

    .. note::

        For coordinate-based inputs (keypoints, bounding boxes, polygons,
        ...), this augmenter still has to perform an image-based augmentation,
        which will make it significantly slower for such inputs than other
        augmenters. See :ref:`performance`.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested (1)
        * ``uint32``: yes; tested (1) (2)
        * ``uint64``: no (3)
        * ``int8``: yes; tested (1)
        * ``int16``: yes; tested (1)
        * ``int32``: yes; tested (1) (2)
        * ``int64``: no (3)
        * ``float16``: yes; tested (1)
        * ``float32``: yes; tested (1)
        * ``float64``: yes; tested (1)
        * ``float128``: no (3)
        * ``bool``: yes; tested (1) (4)

        - (1) Only tested with `order` set to ``0``.
        - (2) scikit-image converts internally to ``float64``, which might
              introduce inaccuracies. Tests showed that these inaccuracies
              seemed to not be an issue.
        - (3) Results too inaccurate.
        - (4) Mapped internally to ``float64``.

    Parameters
    ----------
    scale : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        Each point on the regular grid is moved around via a normal
        distribution. This scale factor is equivalent to the normal
        distribution's sigma. Note that the jitter (how far each point is
        moved in which direction) is multiplied by the height/width of the
        image if ``absolute_scale=False`` (default), so this scale can be
        the same for different sized images.
        Recommended values are in the range ``0.01`` to ``0.05`` (weak to
        strong augmentations).

            * If a single ``float``, then that value will always be used as
              the scale.
            * If a tuple ``(a, b)`` of ``float`` s, then a random value will
              be uniformly sampled per image from the interval ``[a, b]``.
            * If a list, then a random value will be picked from that list
              per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    nb_rows : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of points that the regular grid should have.
        Must be at least ``2``. For large images, you might want to pick a
        higher value than ``4``. You might have to then adjust scale to lower
        values.

            * If a single ``int``, then that value will always be used as the
              number of rows.
            * If a tuple ``(a, b)``, then a value from the discrete interval
              ``[a..b]`` will be uniformly sampled per image.
            * If a list, then a random value will be picked from that list
              per image.
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    nb_cols : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        Number of columns. Analogous to `nb_rows`.

    order : int or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.geometric.Affine.__init__`.

    cval : int or float or tuple of float or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.geometric.Affine.__init__`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.geometric.Affine.__init__`.

    absolute_scale : bool, optional
        Take `scale` as an absolute value rather than a relative value.

    polygon_recoverer : 'auto' or None or imgaug.augmentables.polygons._ConcavePolygonRecoverer, optional
        The class to use to repair invalid polygons.
        If ``"auto"``, a new instance of
        :class`imgaug.augmentables.polygons._ConcavePolygonRecoverer`
        will be created.
        If ``None``, no polygon recoverer will be used.
        If an object, then that object will be used and must provide a
        ``recover_from()`` method, similar to
        :class:`~imgaug.augmentables.polygons._ConcavePolygonRecoverer`.

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
    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))

    Place a regular grid of points on each image and then randomly move each
    point around by ``1`` to ``5`` percent (with respect to the image
    height/width). Pixels between these points will be moved accordingly.

    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=8, nb_cols=8)

    Same as the previous example, but uses a denser grid of ``8x8`` points
    (default is ``4x4``). This can be useful for large images.

    """

    def __init__(
        self,
        scale=(0.0, 0.04),
        nb_rows=(2, 4),
        nb_cols=(2, 4),
        order=1,
        cval=0,
        mode="constant",
        absolute_scale=False,
        polygon_recoverer=None,
        **kwargs
    ):
        self.augmenter = iaa.PiecewiseAffine
        super().__init__(
            scale=scale,
            nb_rows=nb_rows,
            nb_cols=nb_cols,
            order=order,
            cval=cval,
            mode=mode,
            absolute_scale=absolute_scale,
            polygon_recoverer=polygon_recoverer,
            **kwargs
        )


class Rot90(ImgAug):
    """
    Rotate images clockwise by multiples of 90 degrees.

    This could also be achieved using ``Affine``, but ``Rot90`` is
    significantly more efficient.

    **Supported dtypes**:

    if (keep_size=False):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    if (keep_size=True):

        minimum of (
            ``imgaug.augmenters.geometric.Rot90(keep_size=False)``,
            :func:`~imgaug.imgaug.imresize_many_images`
        )

    Parameters
    ----------
    k : int or list of int or tuple of int or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        How often to rotate clockwise by 90 degrees.

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``imgaug.ALL``, then equivalant to list ``[0, 1, 2, 3]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    keep_size : bool, optional
        After rotation by an odd-valued `k` (e.g. 1 or 3), the resulting image
        may have a different height/width than the original image.
        If this parameter is set to ``True``, then the rotated
        image will be resized to the input image's size. Note that this might
        also cause the augmented image to look distorted.

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
    >>> aug = iaa.Rot90(1)

    Rotate all images by 90 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90([1, 3])

    Rotate all images by 90 or 270 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3))

    Rotate all images by 90, 180 or 270 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3), keep_size=False)

    Rotate all images by 90, 180 or 270 degrees.
    Does not resize to the original image size afterwards, i.e. each image's
    size may change.

    """

    def __init__(self, k=1, keep_size=True, **kwargs):
        self.augmenter = iaa.Rot90
        super().__init__(k=k, keep_size=keep_size, **kwargs)


class Rotate(ImgAug):
    """Apply affine rotation on the y-axis to input data.

    This is a wrapper around :class:`Affine`.
    It is the same as ``Affine(rotate=<value>)``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.geometric.Affine`.

    Parameters
    ----------
    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    fit_output : bool, optional
        See :class:`Affine`.

    backend : str, optional
        See :class:`Affine`.

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
    >>> aug = iaa.Rotate((-45, 45))

    Create an augmenter that rotates images by a random value between ``-45``
    and ``45`` degress.

    """

    def __init__(
        self,
        rotate=(-30, 30),
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.Rotate
        super().__init__(
            rotate=rotate,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class ScaleX(ImgAug):
    """Apply affine scaling on the x-axis to input data.

    This is a wrapper around :class:`Affine`.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.geometric.Affine`.

    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Analogous to ``scale`` in :class:`Affine`, except that this scale
        value only affects the x-axis. No dictionary input is allowed.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    fit_output : bool, optional
        See :class:`Affine`.

    backend : str, optional
        See :class:`Affine`.

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
    >>> aug = iaa.ScaleX((0.5, 1.5))

    Create an augmenter that scales images along the width to sizes between
    ``50%`` and ``150%``. This does not change the image shape (i.e. height
    and width), only the pixels within the image are remapped and potentially
    new ones are filled in.

    """

    def __init__(
        self,
        scale=(0.5, 1.5),
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.ScaleX
        super().__init__(
            scale=scale,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class ScaleY(ImgAug):
    """Apply affine scaling on the y-axis to input data.

    This is a wrapper around :class:`Affine`.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.geometric.Affine`.

    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Analogous to ``scale`` in :class:`Affine`, except that this scale
        value only affects the y-axis. No dictionary input is allowed.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    fit_output : bool, optional
        See :class:`Affine`.

    backend : str, optional
        See :class:`Affine`.

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
    >>> aug = iaa.ScaleY((0.5, 1.5))

    Create an augmenter that scales images along the height to sizes between
    ``50%`` and ``150%``. This does not change the image shape (i.e. height
    and width), only the pixels within the image are remapped and potentially
    new ones are filled in.

    """

    def __init__(
        self,
        scale=(0.5, 1.5),
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.ScaleY
        super().__init__(
            scale=scale,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class ShearX(ImgAug):
    """Apply affine shear on the x-axis to input data.

    This is a wrapper around :class:`Affine`.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.geometric.Affine`.

    Parameters
    ----------
    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Analogous to ``shear`` in :class:`Affine`, except that this shear
        value only affects the x-axis. No dictionary input is allowed.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    fit_output : bool, optional
        See :class:`Affine`.

    backend : str, optional
        See :class:`Affine`.

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
    >>> aug = iaa.ShearX((-20, 20))

    Create an augmenter that shears images along the x-axis by random amounts
    between ``-20`` and ``20`` degrees.

    """

    def __init__(
        self,
        shear=(-30, 30),
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.ShearX
        super().__init__(
            shear=shear,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class ShearY(ImgAug):
    """Apply affine shear on the y-axis to input data.

    This is a wrapper around :class:`Affine`.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.geometric.Affine`.

    Parameters
    ----------
    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Analogous to ``shear`` in :class:`Affine`, except that this shear
        value only affects the y-axis. No dictionary input is allowed.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    fit_output : bool, optional
        See :class:`Affine`.

    backend : str, optional
        See :class:`Affine`.

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
    >>> aug = iaa.ShearY((-20, 20))

    Create an augmenter that shears images along the y-axis by random amounts
    between ``-20`` and ``20`` degrees.

    """

    def __init__(
        self,
        shear=(-30, 30),
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.ShearY
        super().__init__(
            shear=shear,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class TranslateX(ImgAug):
    """Apply affine translation on the x-axis to input data.

    This is a wrapper around :class:`Affine`.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.geometric.Affine`.

    Parameters
    ----------
    percent : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Analogous to ``translate_percent`` in :class:`Affine`, except that
        this translation value only affects the x-axis. No dictionary input
        is allowed.

    px : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Analogous to ``translate_px`` in :class:`Affine`, except that
        this translation value only affects the x-axis. No dictionary input
        is allowed.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    fit_output : bool, optional
        See :class:`Affine`.

    backend : str, optional
        See :class:`Affine`.

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
    >>> aug = iaa.TranslateX(px=(-20, 20))

    Create an augmenter that translates images along the x-axis by
    ``-20`` to ``20`` pixels.

    >>> aug = iaa.TranslateX(percent=(-0.1, 0.1))

    Create an augmenter that translates images along the x-axis by
    ``-10%`` to ``10%`` (relative to the x-axis size).

    """

    def __init__(
        self,
        percent=None,
        px=None,
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.TranslateX
        super().__init__(
            percent=percent,
            px=px,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class TranslateY(ImgAug):
    """Apply affine translation on the y-axis to input data.

    This is a wrapper around :class:`Affine`.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.geometric.Affine`.

    Parameters
    ----------
    percent : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Analogous to ``translate_percent`` in :class:`Affine`, except that
        this translation value only affects the y-axis. No dictionary input
        is allowed.

    px : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Analogous to ``translate_px`` in :class:`Affine`, except that
        this translation value only affects the y-axis. No dictionary input
        is allowed.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :class:`Affine`.

    fit_output : bool, optional
        See :class:`Affine`.

    backend : str, optional
        See :class:`Affine`.

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
    >>> aug = iaa.TranslateY(px=(-20, 20))

    Create an augmenter that translates images along the y-axis by
    ``-20`` to ``20`` pixels.

    >>> aug = iaa.TranslateY(percent=(-0.1, 0.1))

    Create an augmenter that translates images along the y-axis by
    ``-10%`` to ``10%`` (relative to the y-axis size).

    """

    def __init__(
        self,
        percent=None,
        px=None,
        order=1,
        cval=0,
        mode="constant",
        fit_output=False,
        backend="auto",
        **kwargs
    ):
        self.augmenter = iaa.TranslateY
        super().__init__(
            percent=percent,
            px=px,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            **kwargs
        )


class WithPolarWarping(ImgAug):
    """Augmenter that applies other augmenters in a polar-transformed space.

    This augmenter first transforms an image into a polar representation,
    then applies its child augmenter, then transforms back to cartesian
    space. The polar representation is still in the image's input dtype
    (i.e. ``uint8`` stays ``uint8``) and can be visualized. It can be thought
    of as an "unrolled" version of the image, where previously circular lines
    appear straight. Hence, applying child augmenters in that space can lead
    to circular effects. E.g. replacing rectangular pixel areas in the polar
    representation with black pixels will lead to curved black areas in
    the cartesian result.

    This augmenter can create new pixels in the image. It will fill these
    with black pixels. For segmentation maps it will fill with class
    id ``0``. For heatmaps it will fill with ``0.0``.

    This augmenter is limited to arrays with a height and/or width of
    ``32767`` or less.

    .. warning::

        When augmenting coordinates in polar representation, it is possible
        that these are shifted outside of the polar image, but are inside the
        image plane after transforming back to cartesian representation,
        usually on newly created pixels (i.e. black backgrounds).
        These coordinates are currently not removed. It is recommended to
        not use very strong child transformations when also augmenting
        coordinate-based augmentables.

    .. warning::

        For bounding boxes, this augmenter suffers from the same problem as
        affine rotations applied to bounding boxes, i.e. the resulting
        bounding boxes can have unintuitive (seemingly wrong) appearance.
        This is due to coordinates being "rotated" that are inside the
        bounding box, but do not fall on the object and actually are
        background.
        It is recommended to use this augmenter with caution when augmenting
        bounding boxes.

    .. warning::

        For polygons, this augmenter should not be combined with
        augmenters that perform automatic polygon recovery for invalid
        polygons, as the polygons will frequently appear broken in polar
        representation and their "fixed" version will be very broken in
        cartesian representation. Augmenters that perform such polygon
        recovery are currently ``PerspectiveTransform``, ``PiecewiseAffine``
        and ``ElasticTransformation``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: no (2)
        * ``float16``: yes; tested (3)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (4)

        - (1) OpenCV produces error
          ``TypeError: Expected cv::UMat for argument 'src'``
        - (2) OpenCV produces array of nothing but zeros.
        - (3) Mapepd to ``float32``.
        - (4) Mapped to ``uint8``.

    Parameters
    ----------
    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images after they were transformed
        to polar representation.

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
    >>> aug = iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1)))

    Apply cropping and padding in polar representation, then warp back to
    cartesian representation.

    >>> aug = iaa.WithPolarWarping(
    >>>     iaa.Affine(
    >>>         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    >>>         rotate=(-35, 35),
    >>>         scale=(0.8, 1.2),
    >>>         shear={"x": (-15, 15), "y": (-15, 15)}
    >>>     )
    >>> )

    Apply affine transformations in polar representation.

    >>> aug = iaa.WithPolarWarping(iaa.AveragePooling((2, 8)))

    Apply average pooling in polar representation. This leads to circular
    bins.

    """

    def __init__(self, children, **kwargs):
        self.augmenter = iaa.WithPolarWarping
        super().__init__(children=children, **kwargs)


class CenterCropToAspectRatio(ImgAug):
    """Crop images equally on all sides until they reach an aspect ratio.

    This is the same as :class:`~imgaug.augmenters.size.CropToAspectRatio`, but
    uses ``position="center"`` by default, which spreads the crop amounts
    equally over all image sides, while
    :class:`~imgaug.augmenters.size.CropToAspectRatio` by default spreads
    them randomly.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        See :func:`CropToAspectRatio.__init__`.

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
    >>> aug = iaa.CenterCropToAspectRatio(2.0)

    Create an augmenter that crops each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, aspect_ratio, **kwargs):
        self.augmenter = iaa.CenterCropToAspectRatio
        super().__init__(aspect_ratio=aspect_ratio, **kwargs)


class CenterCropToFixedSize(ImgAug):
    """Take a crop from the center of each image.

    This is an alias for :class:`~imgaug.augmenters.size.CropToFixedSize` with
    ``position="center"``.

    .. note::

        If images already have a width and/or height below the provided
        width and/or height then this augmenter will do nothing for the
        respective axis. Hence, resulting images can be smaller than the
        provided axis sizes.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width : int or None
        See :func:`CropToFixedSize.__init__`.

    height : int or None
        See :func:`CropToFixedSize.__init__`.

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
    >>> crop = iaa.CenterCropToFixedSize(height=20, width=10)

    Create an augmenter that takes ``20x10`` sized crops from the center of
    images.

    """

    def __init__(self, width, height, **kwargs):
        self.augmenter = iaa.CenterCropToFixedSize
        super().__init__(width=width, height=height, **kwargs)


class CenterCropToMultiplesOf(ImgAug):
    """Crop images equally on all sides until H/W are multiples of given values.

    This is the same as :class:`~imgaug.augmenters.size.CropToMultiplesOf`,
    but uses ``position="center"`` by default, which spreads the crop amounts
    equally over all image sides, while
    :class:`~imgaug.augmenters.size.CropToMultiplesOf` by default spreads
    them randomly.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        See :func:`CropToMultiplesOf.__init__`.

    height_multiple : int or None
        See :func:`CropToMultiplesOf.__init__`.

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
    >>> aug = iaa.CenterCropToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that crops images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, width_multiple, height_multiple, **kwargs):
        self.augmenter = iaa.CenterCropToMultiplesOf
        super().__init__(
            width_multiple=width_multiple, height_multiple=height_multiple, **kwargs
        )


class CenterCropToPowersOf(ImgAug):
    """Crop images equally on all sides until H/W is a power of a base.

    This is the same as :class:`~imgaug.augmenters.size.CropToPowersOf`, but
    uses ``position="center"`` by default, which spreads the crop amounts
    equally over all image sides, while
    :class:`~imgaug.augmenters.size.CropToPowersOf` by default spreads them
    randomly.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        See :func:`CropToPowersOf.__init__`.

    height_base : int or None
        See :func:`CropToPowersOf.__init__`.

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
    >>> aug = iaa.CropToPowersOf(height_base=3, width_base=2)

    Create an augmenter that crops each image down to powers of ``3`` along
    the y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e.
    2, 4, 8, 16, ...).
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, width_base, height_base, **kwargs):
        self.augmenter = iaa.CenterCropToPowersOf
        super().__init__(width_base=width_base, height_base=height_base, **kwargs)


class CenterCropToSquare(ImgAug):
    """Crop images equally on all sides until their height/width are identical.

    In contrast to :class:`~imgaug.augmenters.size.CropToSquare`, this
    augmenter always tries to spread the columns/rows to remove equally over
    both sides of the respective axis to be cropped.
    :class:`~imgaug.augmenters.size.CropToAspectRatio` by default spreads the
    croppings randomly.

    This augmenter is identical to :class:`~imgaug.augmenters.size.CropToSquare`
    with ``position="center"``, and thereby the same as
    :class:`~imgaug.augmenters.size.CropToAspectRatio` with
    ``aspect_ratio=1.0, position="center"``.

    Images with axis sizes of ``0`` will not be altered.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
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
    >>> aug = iaa.CenterCropToSquare()

    Create an augmenter that crops each image until its square, i.e. height
    and width match.
    The rows to be cropped will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, **kwargs):
        self.augmenter = iaa.CenterCropToSquare
        super().__init__(**kwargs)


class CenterPadToAspectRatio(ImgAug):
    """Pad images equally on all sides until H/W matches an aspect ratio.

    This is the same as :class:`~imgaug.augmenters.size.PadToAspectRatio`, but
    uses ``position="center"`` by default, which spreads the pad amounts
    equally over all image sides, while
    :class:`~imgaug.augmenters.size.PadToAspectRatio` by default spreads them
    randomly.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        See :func:`PadToAspectRatio.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

    deterministic : bool, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.PadToAspectRatio(2.0)

    Create am augmenter that pads each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, aspect_ratio, pad_mode="constant", pad_cval=0, **kwargs):
        self.augmenter = iaa.CenterPadToAspectRatio
        super().__init__(
            aspect_ratio=aspect_ratio, pad_mode=pad_mode, pad_cval=pad_cval, **kwargs
        )


class CenterPadToFixedSize(ImgAug):
    """Pad images equally on all sides up to given minimum heights/widths.

    This is an alias for :class:`~imgaug.augmenters.size.PadToFixedSize`
    with ``position="center"``. It spreads the pad amounts equally over
    all image sides, while :class:`~imgaug.augmenters.size.PadToFixedSize`
    by defaults spreads them randomly.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width : int or None
        See :func:`PadToFixedSize.__init__`.

    height : int or None
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

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
    >>> aug = iaa.CenterPadToFixedSize(height=20, width=30)

    Create an augmenter that pads images up to ``20x30``, with the padded
    rows added *equally* on the top and bottom (analogous for the padded
    columns).

    """

    def __init__(self, width, height, pad_mode="constant", pad_cval=0, **kwargs):
        self.augmenter = iaa.CenterPadToFixedSize
        super().__init__(
            width=width, height=height, pad_mode=pad_mode, pad_cval=pad_cval, **kwargs
        )


class CenterPadToMultiplesOf(ImgAug):
    """Pad images equally on all sides until H/W are multiples of given values.

    This is the same as :class:`~imgaug.augmenters.size.PadToMultiplesOf`, but
    uses ``position="center"`` by default, which spreads the pad amounts
    equally over all image sides, while
    :class:`~imgaug.augmenters.size.PadToMultiplesOf` by default spreads them
    randomly.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        See :func:`PadToMultiplesOf.__init__`.

    height_multiple : int or None
        See :func:`PadToMultiplesOf.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToMultiplesOf.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToMultiplesOf.__init__`.

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
    >>> aug = iaa.CenterPadToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that pads images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(
        self, width_multiple, height_multiple, pad_mode="constant", pad_cval=0, **kwargs
    ):
        self.augmenter = iaa.CenterPadToMultiplesOf
        super().__init__(
            width_multiple=width_multiple,
            height_multiple=height_multiple,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            **kwargs
        )


class CenterPadToPowersOf(ImgAug):
    """Pad images equally on all sides until H/W is a power of a base.

    This is the same as :class:`~imgaug.augmenters.size.PadToPowersOf`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class:`~imgaug.augmenters.size.PadToPowersOf`
    by default spreads them randomly.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        See :func:`PadToPowersOf.__init__`.

    height_base : int or None
        See :func:`PadToPowersOf.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToPowersOf.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToPowersOf.__init__`.

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
    >>> aug = iaa.CenterPadToPowersOf(height_base=5, width_base=2)

    Create an augmenter that pads each image to powers of ``3`` along the
    y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e. 2,
    4, 8, 16, ...).
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(
        self, width_base, height_base, pad_mode="constant", pad_cval=0, **kwargs
    ):
        self.augmenter = iaa.CenterPadToPowersOf
        super().__init__(
            width_base=width_base,
            height_base=height_base,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            **kwargs
        )


class CenterPadToSquare(ImgAug):
    """Pad images equally on all sides until their height & width are identical.

    This is the same as :class:`~imgaug.augmenters.size.PadToSquare`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class:`~imgaug.augmenters.size.PadToSquare`
    by default spreads them randomly. This augmenter is thus also identical to
    :class:`~imgaug.augmenters.size.PadToAspectRatio` with
    ``aspect_ratio=1.0, position="center"``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToAspectRatio.__init__`.

    deterministic : bool, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CenterPadToSquare()

    Create an augmenter that pads each image until its square, i.e. height
    and width match.
    The rows to be padded will be spread *equally* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, pad_mode="constant", pad_cval=0, **kwargs):
        self.augmenter = iaa.CenterPadToSquare
        super().__init__(pad_mode=pad_mode, pad_cval=pad_cval, **kwargs)


class Crop(ImgAug):
    """Crop images, i.e. remove columns/rows of pixels at the sides of images.

    This augmenter allows to extract smaller-sized subimages from given
    full-sized input images. The number of pixels to cut off may be defined
    in absolute values or as fractions of the image sizes.

    This augmenter will never crop images below a height or width of ``1``.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropAndPad`.

    Parameters
    ----------
    px : None or int or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop on each side of the image.
        Expected value range is ``[0, inf)``.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If ``None``, then pixel-based cropping will not be used.
            * If ``int``, then that exact number of pixels will always be
              cropped.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be cropped by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              crop by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (crop by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (crop by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to crop from that parameter).

    percent : None or int or float or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``0.1``, the augmenter will
        always crop ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``[0.0, 1.0)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based cropping will not be
              used.
            * If ``number``, then that fraction will always be cropped.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be cropped by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always crop by exactly that fraction), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (crop by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (crop by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to crop from that parameter).

    keep_size : bool, optional
        After cropping, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the cropped image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the crop amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

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
    >>> aug = iaa.Crop(px=(0, 10))

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``.

    >>> aug = iaa.Crop(px=(0, 10), sample_independently=False)

    Crop each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.Crop(px=(0, 10), keep_size=False)

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the cropped image back to the input image's size. This will decrease
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.Crop(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Crop the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Crop the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.Crop(percent=(0, 0.1))

    Crop each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would crop by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.Crop(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Crops each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    """

    def __init__(
        self, px=None, percent=None, keep_size=True, sample_independently=True, **kwargs
    ):
        self.augmenter = iaa.Crop
        super().__init__(
            px=px,
            percent=percent,
            keep_size=keep_size,
            sample_independently=sample_independently,
            **kwargs
        )


class CropAndPad(ImgAug):
    """Crop/pad images by pixel amounts or fractions of image sizes.

    Cropping removes pixels at the sides (i.e. extracts a subimage from
    a given full image). Padding adds pixels to the sides (e.g. black pixels).

    This augmenter will never crop images below a height or width of ``1``.

    .. note::

        This augmenter automatically resizes images back to their original size
        after it has augmented them. To deactivate this, add the
        parameter ``keep_size=False``.

    **Supported dtypes**:

    if (keep_size=False):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    if (keep_size=True):

        minimum of (
            ``imgaug.augmenters.size.CropAndPad(keep_size=False)``,
            :func:`~imgaug.imgaug.imresize_many_images`
        )

    Parameters
    ----------
    px : None or int or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop (negative values) or pad (positive values)
        on each side of the image. Either this or the parameter `percent` may
        be set, not both at the same time.

            * If ``None``, then pixel-based cropping/padding will not be used.
            * If ``int``, then that exact number of pixels will always be
              cropped/padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be cropped/padded by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              crop/pad by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (crop/pad by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (crop/pad by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to crop/pad from that parameter).

    percent : None or number or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop (negative values) or pad (positive values)
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``-0.1``, the augmenter will
        always crop away ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``(-1.0, inf)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based cropping/padding will not be
              used.
            * If ``number``, then that fraction will always be cropped/padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be cropped/padded by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always crop/pad by exactly that percent value), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (crop/pad by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (crop/pad by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to crop/pad from that parameter).

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        Padding mode to use. The available modes match the numpy padding modes,
        i.e. ``constant``, ``edge``, ``linear_ramp``, ``maximum``, ``median``,
        ``minimum``, ``reflect``, ``symmetric``, ``wrap``. The modes
        ``constant`` and ``linear_ramp`` use extra values, which are provided
        by ``pad_cval`` when necessary. See :func:`~imgaug.imgaug.pad` for
        more details.

            * If ``imgaug.ALL``, then a random mode from all available modes
              will be sampled per image.
            * If a ``str``, it will be used as the pad mode for all images.
            * If a ``list`` of ``str``, a random one of these will be sampled
              per image and used as the mode.
            * If ``StochasticParameter``, a random mode will be sampled from
              this parameter per image.

    pad_cval : number or tuple of number list of number or imgaug.parameters.StochasticParameter, optional
        The constant value to use if the pad mode is ``constant`` or the end
        value to use if the mode is ``linear_ramp``.
        See :func:`~imgaug.imgaug.pad` for more details.

            * If ``number``, then that value will be used.
            * If a ``tuple`` of two ``number`` s and at least one of them is
              a ``float``, then a random number will be uniformly sampled per
              image from the continuous interval ``[a, b]`` and used as the
              value. If both ``number`` s are ``int`` s, the interval is
              discrete.
            * If a ``list`` of ``number``, then a random value will be chosen
              from the elements of the ``list`` and used as the value.
            * If ``StochasticParameter``, a random value will be sampled from
              that parameter per image.

    keep_size : bool, optional
        After cropping and padding, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the cropped/padded image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the crop/pad amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

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
    >>> aug = iaa.CropAndPad(px=(-10, 0))

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[-10..0]``.

    >>> aug = iaa.CropAndPad(px=(0, 10))

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding happens by
    zero-padding, i.e. it adds black pixels (default setting).

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode="edge")

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding uses the
    ``edge`` mode from numpy's pad function, i.e. the pixel colors around
    the image sides are repeated.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=["constant", "edge"])

    Similar to the previous example, but uses zero-padding (``constant``) for
    half of the images and ``edge`` padding for the other half.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    Similar to the previous example, but uses any available padding mode.
    In case the padding mode ends up being ``constant`` or ``linear_ramp``,
    and random intensity is uniformly sampled (once per image) from the
    discrete interval ``[0..255]`` and used as the intensity of the new
    pixels.

    >>> aug = iaa.CropAndPad(px=(0, 10), sample_independently=False)

    Pad each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.CropAndPad(px=(0, 10), keep_size=False)

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the padded image back to the input image's size. This will increase
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.CropAndPad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Pad the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Pad the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.CropAndPad(percent=(0, 0.1))

    Pad each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would pad by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.CropAndPad(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Pads each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    >>> aug = iaa.CropAndPad(px=(-10, 10))

    Sample uniformly per image and side a value ``v`` from the discrete range
    ``[-10..10]``. Then either crop (negative sample) or pad (positive sample)
    the side by ``v`` pixels.

    """

    def __init__(
        self,
        px=None,
        percent=None,
        pad_mode="constant",
        pad_cval=0,
        keep_size=True,
        sample_independently=True,
        **kwargs
    ):
        self.augmenter = iaa.CropAndPad
        super().__init__(
            px=px,
            percent=percent,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            keep_size=keep_size,
            sample_independently=sample_independently,
            **kwargs
        )


class CropToAspectRatio(ImgAug):
    """Crop images until their width/height matches an aspect ratio.

    This augmenter removes either rows or columns until the image reaches
    the desired aspect ratio given in ``width / height``. The cropping
    operation is stopped once the desired aspect ratio is reached or the image
    side to crop reaches a size of ``1``. If any side of the image starts
    with a size of ``0``, the image will not be changed.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        The desired aspect ratio, given as ``width/height``. E.g. a ratio
        of ``2.0`` denotes an image that is twice as wide as it is high.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

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
    >>> aug = iaa.CropToAspectRatio(2.0)

    Create an augmenter that crops each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, aspect_ratio, position="uniform", **kwargs):
        self.augmenter = iaa.CropToAspectRatio
        super().__init__(aspect_ratio=aspect_ratio, position=position, **kwargs)


class CropToFixedSize(ImgAug):
    """Crop images down to a predefined maximum width and/or height.

    If images are already at the maximum width/height or are smaller, they
    will not be cropped. Note that this also means that images will not be
    padded if they are below the required width/height.

    The augmenter randomly decides per image how to distribute the required
    cropping amounts over the image axis. E.g. if 2px have to be cropped on
    the left or right to reach the required width, the augmenter will
    sometimes remove 2px from the left and 0px from the right, sometimes
    remove 2px from the right and 0px from the left and sometimes remove 1px
    from both sides. Set `position` to ``center`` to prevent that.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    width : int or None
        Crop images down to this maximum width.
        If ``None``, image widths will not be altered.

    height : int or None
        Crop images down to this maximum height.
        If ``None``, image heights will not be altered.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
         Sets the center point of the cropping, which determines how the
         required cropping amounts are distributed to each side. For a
         ``tuple`` ``(a, b)``, both ``a`` and ``b`` are expected to be in
         range ``[0.0, 1.0]`` and describe the fraction of cropping applied
         to the left/right (low/high values for ``a``) and the fraction
         of cropping applied to the top/bottom (low/high values for ``b``).
         A cropping position at ``(0.5, 0.5)`` would be the center of the
         image and distribute the cropping equally over all sides. A cropping
         position at ``(1.0, 0.0)`` would be the right-top and would apply
         100% of the required cropping to the right and top sides of the image.

            * If string ``uniform`` then the share of cropping is randomly
              and uniformly distributed over each side.
              Equivalent to ``(Uniform(0.0, 1.0), Uniform(0.0, 1.0))``.
            * If string ``normal`` then the share of cropping is distributed
              based on a normal distribution, leading to a focus on the center
              of the images.
              Equivalent to
              ``(Clip(Normal(0.5, 0.45/2), 0, 1),
              Clip(Normal(0.5, 0.45/2), 0, 1))``.
            * If string ``center`` then center point of the cropping is
              identical to the image center.
              Equivalent to ``(0.5, 0.5)``.
            * If a string matching regex
              ``^(left|center|right)-(top|center|bottom)$``, e.g.
              ``left-top`` or ``center-bottom`` then sets the center point of
              the cropping to the X-Y position matching that description.
            * If a tuple of float, then expected to have exactly two entries
              between ``0.0`` and ``1.0``, which will always be used as the
              combination the position matching (x, y) form.
            * If a ``StochasticParameter``, then that parameter will be queried
              once per call to ``augment_*()`` to get ``Nx2`` center positions
              in ``(x, y)`` form (with ``N`` the number of images).
            * If a ``tuple`` of ``StochasticParameter``, then expected to have
              exactly two entries that will both be queried per call to
              ``augment_*()``, each for ``(N,)`` values, to get the center
              positions. First parameter is used for ``x`` coordinates,
              second for ``y`` coordinates.

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
    >>> aug = iaa.CropToFixedSize(width=100, height=100)

    For image sides larger than ``100`` pixels, crop to ``100`` pixels. Do
    nothing for the other sides. The cropping amounts are randomly (and
    uniformly) distributed over the sides of the image.

    >>> aug = iaa.CropToFixedSize(width=100, height=100, position="center")

    For sides larger than ``100`` pixels, crop to ``100`` pixels. Do nothing
    for the other sides. The cropping amounts are always equally distributed
    over the left/right sides of the image (and analogously for top/bottom).

    >>> aug = iaa.Sequential([
    >>>     iaa.PadToFixedSize(width=100, height=100),
    >>>     iaa.CropToFixedSize(width=100, height=100)
    >>> ])

    Pad images smaller than ``100x100`` until they reach ``100x100``.
    Analogously, crop images larger than ``100x100`` until they reach
    ``100x100``. The output images therefore have a fixed size of ``100x100``.

    """

    def __init__(self, width, height, position="uniform", **kwargs):
        self.augmenter = iaa.CropToFixedSize
        super().__init__(width=width, height=height, position=position, **kwargs)


class CropToMultiplesOf(ImgAug):
    """Crop images down until their height/width is a multiple of a value.

    .. note::

        For a given axis size ``A`` and multiple ``M``, if ``A`` is in the
        interval ``[0 .. M]``, the axis will not be changed.
        As a result, this augmenter can still produce axis sizes that are
        not multiples of the given values.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        Multiple for the width. Images will be cropped down until their
        width is a multiple of this value.
        If ``None``, image widths will not be altered.

    height_multiple : int or None
        Multiple for the height. Images will be cropped down until their
        height is a multiple of this value.
        If ``None``, image heights will not be altered.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

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
    >>> aug = iaa.CropToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that crops images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, width_multiple, height_multiple, position="uniform", **kwargs):
        self.augmenter = iaa.CropToMultiplesOf
        super().__init__(
            width_multiple=width_multiple,
            height_multiple=height_multiple,
            position=position,
            **kwargs
        )


class CropToPowersOf(ImgAug):
    """Crop images until their height/width is a power of a base.

    This augmenter removes pixels from an axis with size ``S`` leading to the
    new size ``S'`` until ``S' = B^E`` is fulfilled, where ``B`` is a
    provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
    interval ``[1 .. inf)``.

    .. note::

        This augmenter does nothing for axes with size less than ``B^1 = B``.
        If you have images with ``S < B^1``, it is recommended
        to combine this augmenter with a padding augmenter that pads each
        axis up to ``B``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        Base for the width. Images will be cropped down until their
        width fulfills ``width' = width_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image widths will not be altered.

    height_base : int or None
        Base for the height. Images will be cropped down until their
        height fulfills ``height' = height_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image heights will not be altered.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

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
    >>> aug = iaa.CropToPowersOf(height_base=3, width_base=2)

    Create an augmenter that crops each image down to powers of ``3`` along
    the y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e.
    2, 4, 8, 16, ...).
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, width_base, height_base, position="uniform", **kwargs):
        self.augmenter = iaa.CropToPowersOf
        super().__init__(
            width_base=width_base, height_base=height_base, position=position, **kwargs
        )


class CropToSquare(ImgAug):
    """Crop images until their width and height are identical.

    This is identical to :class:`~imgaug.augmenters.size.CropToAspectRatio`
    with ``aspect_ratio=1.0``.

    Images with axis sizes of ``0`` will not be altered.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

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
    >>> aug = iaa.CropToSquare()

    Create an augmenter that crops each image until its square, i.e. height
    and width match.
    The rows to be cropped will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, position="uniform", **kwargs):
        self.augmenter = iaa.CropToSquare
        super().__init__(position=position, **kwargs)


class KeepSizeByResize(ImgAug):
    """Resize images back to their input sizes after applying child augmenters.

    Combining this with e.g. a cropping augmenter as the child will lead to
    images being resized back to the input size after the crop operation was
    applied. Some augmenters have a ``keep_size`` argument that achieves the
    same goal (if set to ``True``), though this augmenter offers control over
    the interpolation mode and which augmentables to resize (images, heatmaps,
    segmentation maps).

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.imresize_many_images`.

    Parameters
    ----------
    children : Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images. These augmenters may change
        the image size.

    interpolation : KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing images.
        Can take any value that :func:`~imgaug.imgaug.imresize_single_image`
        accepts, e.g. ``cubic``.

            * If this is ``KeepSizeByResize.NO_RESIZE`` then images will not
              be resized.
            * If this is a single ``str``, it is expected to have one of the
              following values: ``nearest``, ``linear``, ``area``, ``cubic``.
            * If this is a single integer, it is expected to have a value
              identical to one of: ``cv2.INTER_NEAREST``,
              ``cv2.INTER_LINEAR``, ``cv2.INTER_AREA``, ``cv2.INTER_CUBIC``.
            * If this is a ``list`` of ``str`` or ``int``, it is expected that
              each ``str``/``int`` is one of the above mentioned valid ones.
              A random one of these values will be sampled per image.
            * If this is a ``StochasticParameter``, it will be queried once per
              call to ``_augment_images()`` and must return ``N`` ``str`` s or
              ``int`` s (matching the above mentioned ones) for ``N`` images.

    interpolation_heatmaps : KeepSizeByResize.SAME_AS_IMAGES or KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing heatmaps.
        Meaning and valid values are similar to `interpolation`. This
        parameter may also take the value ``KeepSizeByResize.SAME_AS_IMAGES``,
        which will lead to copying the interpolation modes used for the
        corresponding images. The value may also be returned on a per-image
        basis if `interpolation_heatmaps` is provided as a
        ``StochasticParameter`` or may be one possible value if it is
        provided as a ``list`` of ``str``.

    interpolation_segmaps : KeepSizeByResize.SAME_AS_IMAGES or KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing segmentation maps.
        Similar to `interpolation_heatmaps`.
        **Note**: For segmentation maps, only ``NO_RESIZE`` or nearest
        neighbour interpolation (i.e. ``nearest``) make sense in the vast
        majority of all cases.

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
    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False)
    >>> )

    Apply random cropping to input images, then resize them back to their
    original input sizes. The resizing is done using this augmenter instead
    of the corresponding internal resizing operation in ``Crop``.

    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False),
    >>>     interpolation="nearest"
    >>> )

    Same as in the previous example, but images are now always resized using
    nearest neighbour interpolation.

    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False),
    >>>     interpolation=["nearest", "cubic"],
    >>>     interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES,
    >>>     interpolation_segmaps=iaa.KeepSizeByResize.NO_RESIZE
    >>> )

    Similar to the previous example, but images are now sometimes resized
    using linear interpolation and sometimes using nearest neighbour
    interpolation. Heatmaps are resized using the same interpolation as was
    used for the corresponding image. Segmentation maps are not resized and
    will therefore remain at their size after cropping.

    """

    def __init__(
        self,
        children,
        interpolation="cubic",
        interpolation_heatmaps="SAME_AS_IMAGES",
        interpolation_segmaps="nearest",
        **kwargs
    ):
        self.augmenter = iaa.KeepSizeByResize
        super().__init__(
            children=children,
            interpolation=interpolation,
            interpolation_heatmaps=interpolation_heatmaps,
            interpolation_segmaps=interpolation_segmaps,
            **kwargs
        )


class Pad(ImgAug):
    """Pad images, i.e. adds columns/rows of pixels to them.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.CropAndPad`.

    Parameters
    ----------
    px : None or int or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to pad on each side of the image.
        Expected value range is ``[0, inf)``.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If ``None``, then pixel-based padding will not be used.
            * If ``int``, then that exact number of pixels will always be
              padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be padded by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              pad by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (pad by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (pad by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to pad from that parameter).

    percent : None or int or float or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to pad
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``0.1``, the augmenter will
        always pad ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``[0.0, inf)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based padding will not be
              used.
            * If ``number``, then that fraction will always be padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be padded by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always pad by exactly that fraction), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (pad by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (pad by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to pad from that parameter).

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        Padding mode to use. The available modes match the numpy padding modes,
        i.e. ``constant``, ``edge``, ``linear_ramp``, ``maximum``, ``median``,
        ``minimum``, ``reflect``, ``symmetric``, ``wrap``. The modes
        ``constant`` and ``linear_ramp`` use extra values, which are provided
        by ``pad_cval`` when necessary. See :func:`~imgaug.imgaug.pad` for
        more details.

            * If ``imgaug.ALL``, then a random mode from all available modes
              will be sampled per image.
            * If a ``str``, it will be used as the pad mode for all images.
            * If a ``list`` of ``str``, a random one of these will be sampled
              per image and used as the mode.
            * If ``StochasticParameter``, a random mode will be sampled from
              this parameter per image.

    pad_cval : number or tuple of number list of number or imgaug.parameters.StochasticParameter, optional
        The constant value to use if the pad mode is ``constant`` or the end
        value to use if the mode is ``linear_ramp``.
        See :func:`~imgaug.imgaug.pad` for more details.

            * If ``number``, then that value will be used.
            * If a ``tuple`` of two ``number`` s and at least one of them is
              a ``float``, then a random number will be uniformly sampled per
              image from the continuous interval ``[a, b]`` and used as the
              value. If both ``number`` s are ``int`` s, the interval is
              discrete.
            * If a ``list`` of ``number``, then a random value will be chosen
              from the elements of the ``list`` and used as the value.
            * If ``StochasticParameter``, a random value will be sampled from
              that parameter per image.

    keep_size : bool, optional
        After padding, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the padded image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the pad amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

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
    >>> aug = iaa.Pad(px=(0, 10))

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding happens by
    zero-padding, i.e. it adds black pixels (default setting).

    >>> aug = iaa.Pad(px=(0, 10), pad_mode="edge")

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding uses the
    ``edge`` mode from numpy's pad function, i.e. the pixel colors around
    the image sides are repeated.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=["constant", "edge"])

    Similar to the previous example, but uses zero-padding (``constant``) for
    half of the images and ``edge`` padding for the other half.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    Similar to the previous example, but uses any available padding mode.
    In case the padding mode ends up being ``constant`` or ``linear_ramp``,
    and random intensity is uniformly sampled (once per image) from the
    discrete interval ``[0..255]`` and used as the intensity of the new
    pixels.

    >>> aug = iaa.Pad(px=(0, 10), sample_independently=False)

    Pad each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.Pad(px=(0, 10), keep_size=False)

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the padded image back to the input image's size. This will increase
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.Pad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Pad the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Pad the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.Pad(percent=(0, 0.1))

    Pad each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would pad by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.Pad(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Pads each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    """

    def __init__(
        self,
        px=None,
        percent=None,
        pad_mode="constant",
        pad_cval=0,
        keep_size=True,
        sample_independently=True,
        **kwargs
    ):
        self.augmenter = iaa.Pad
        super().__init__(
            px=px,
            percent=percent,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            keep_size=keep_size,
            sample_independently=sample_independently,
            **kwargs
        )


class PadToAspectRatio(ImgAug):
    """Pad images until their width/height matches an aspect ratio.

    This augmenter adds either rows or columns until the image reaches
    the desired aspect ratio given in ``width / height``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        The desired aspect ratio, given as ``width/height``. E.g. a ratio
        of ``2.0`` denotes an image that is twice as wide as it is high.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

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
    >>> aug = iaa.PadToAspectRatio(2.0)

    Create an augmenter that pads each image until its aspect ratio is as
    close as possible to ``2.0`` (i.e. two times as many pixels along the
    x-axis than the y-axis).
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(
        self,
        aspect_ratio,
        pad_mode="constant",
        pad_cval=0,
        position="uniform",
        **kwargs
    ):
        self.augmenter = iaa.PadToAspectRatio
        super().__init__(
            aspect_ratio=aspect_ratio,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
            **kwargs
        )


class PadToFixedSize(ImgAug):
    """Pad images to a predefined minimum width and/or height.

    If images are already at the minimum width/height or are larger, they will
    not be padded. Note that this also means that images will not be cropped if
    they exceed the required width/height.

    The augmenter randomly decides per image how to distribute the required
    padding amounts over the image axis. E.g. if 2px have to be padded on the
    left or right to reach the required width, the augmenter will sometimes
    add 2px to the left and 0px to the right, sometimes add 2px to the right
    and 0px to the left and sometimes add 1px to both sides. Set `position`
    to ``center`` to prevent that.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.size.pad`.

    Parameters
    ----------
    width : int or None
        Pad images up to this minimum width.
        If ``None``, image widths will not be altered.

    height : int or None
        Pad images up to this minimum height.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.CropAndPad.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.CropAndPad.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        Sets the center point of the padding, which determines how the
        required padding amounts are distributed to each side. For a ``tuple``
        ``(a, b)``, both ``a`` and ``b`` are expected to be in range
        ``[0.0, 1.0]`` and describe the fraction of padding applied to the
        left/right (low/high values for ``a``) and the fraction of padding
        applied to the top/bottom (low/high values for ``b``). A padding
        position at ``(0.5, 0.5)`` would be the center of the image and
        distribute the padding equally to all sides. A padding position at
        ``(0.0, 1.0)`` would be the left-bottom and would apply 100% of the
        required padding to the bottom and left sides of the image so that
        the bottom left corner becomes more and more the new image
        center (depending on how much is padded).

            * If string ``uniform`` then the share of padding is randomly and
              uniformly distributed over each side.
              Equivalent to ``(Uniform(0.0, 1.0), Uniform(0.0, 1.0))``.
            * If string ``normal`` then the share of padding is distributed
              based on a normal distribution, leading to a focus on the
              center of the images.
              Equivalent to
              ``(Clip(Normal(0.5, 0.45/2), 0, 1),
              Clip(Normal(0.5, 0.45/2), 0, 1))``.
            * If string ``center`` then center point of the padding is
              identical to the image center.
              Equivalent to ``(0.5, 0.5)``.
            * If a string matching regex
              ``^(left|center|right)-(top|center|bottom)$``, e.g. ``left-top``
              or ``center-bottom`` then sets the center point of the padding
              to the X-Y position matching that description.
            * If a tuple of float, then expected to have exactly two entries
              between ``0.0`` and ``1.0``, which will always be used as the
              combination the position matching (x, y) form.
            * If a ``StochasticParameter``, then that parameter will be queried
              once per call to ``augment_*()`` to get ``Nx2`` center positions
              in ``(x, y)`` form (with ``N`` the number of images).
            * If a ``tuple`` of ``StochasticParameter``, then expected to have
              exactly two entries that will both be queried per call to
              ``augment_*()``, each for ``(N,)`` values, to get the center
              positions. First parameter is used for ``x`` coordinates,
              second for ``y`` coordinates.

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
    >>> aug = iaa.PadToFixedSize(width=100, height=100)

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels. Do
    nothing for the other edges. The padding is randomly (uniformly)
    distributed over the sides, so that e.g. sometimes most of the required
    padding is applied to the left, sometimes to the right (analogous
    top/bottom).

    >>> aug = iaa.PadToFixedSize(width=100, height=100, position="center")

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels. Do
    nothing for the other image sides. The padding is always equally
    distributed over the left/right and top/bottom sides.

    >>> aug = iaa.PadToFixedSize(width=100, height=100, pad_mode=ia.ALL)

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels and
    use any possible padding mode for that. Do nothing for the other image
    sides. The padding is always equally distributed over the left/right and
    top/bottom sides.

    >>> aug = iaa.Sequential([
    >>>     iaa.PadToFixedSize(width=100, height=100),
    >>>     iaa.CropToFixedSize(width=100, height=100)
    >>> ])

    Pad images smaller than ``100x100`` until they reach ``100x100``.
    Analogously, crop images larger than ``100x100`` until they reach
    ``100x100``. The output images therefore have a fixed size of ``100x100``.

    """

    def __init__(
        self,
        width,
        height,
        pad_mode="constant",
        pad_cval=0,
        position="uniform",
        **kwargs
    ):
        self.augmenter = iaa.PadToFixedSize
        super().__init__(
            width=width,
            height=height,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
            **kwargs
        )


class PadToMultiplesOf(ImgAug):
    """Pad images until their height/width is a multiple of a value.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        Multiple for the width. Images will be padded until their
        width is a multiple of this value.
        If ``None``, image widths will not be altered.

    height_multiple : int or None
        Multiple for the height. Images will be padded until their
        height is a multiple of this value.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

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
    >>> aug = iaa.PadToMultiplesOf(height_multiple=10, width_multiple=6)

    Create an augmenter that pads images to multiples of ``10`` along
    the y-axis (i.e. 10, 20, 30, ...) and to multiples of ``6`` along the
    x-axis (i.e. 6, 12, 18, ...).
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(
        self,
        width_multiple,
        height_multiple,
        pad_mode="constant",
        pad_cval=0,
        position="uniform",
        **kwargs
    ):
        self.augmenter = iaa.PadToMultiplesOf
        super().__init__(
            width_multiple=width_multiple,
            height_multiple=height_multiple,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
            **kwargs
        )


class PadToPowersOf(ImgAug):
    """Pad images until their height/width is a power of a base.

    This augmenter adds pixels to an axis with size ``S`` leading to the
    new size ``S'`` until ``S' = B^E`` is fulfilled, where ``B`` is a
    provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
    interval ``[1 .. inf)``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        Base for the width. Images will be padded down until their
        width fulfills ``width' = width_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image widths will not be altered.

    height_base : int or None
        Base for the height. Images will be padded until their
        height fulfills ``height' = height_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

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
    >>> aug = iaa.PadToPowersOf(height_base=3, width_base=2)

    Create an augmenter that pads each image to powers of ``3`` along the
    y-axis (i.e. 3, 9, 27, ...) and powers of ``2`` along the x-axis (i.e. 2,
    4, 8, 16, ...).
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(
        self,
        width_base,
        height_base,
        pad_mode="constant",
        pad_cval=0,
        position="uniform",
        **kwargs
    ):
        self.augmenter = iaa.PadToPowersOf
        super().__init__(
            width_base=width_base,
            height_base=height_base,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
            **kwargs
        )


class PadToSquare(ImgAug):
    """Pad images until their height and width are identical.

    This augmenter is identical to
    :class:`~imgaug.augmenters.size.PadToAspectRatio` with ``aspect_ratio=1.0``.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`~imgaug.augmenters.size.PadToFixedSize.__init__`.

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
    >>> aug = iaa.PadToSquare()

    Create an augmenter that pads each image until its square, i.e. height
    and width match.
    The rows to be padded will be spread *randomly* over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, pad_mode="constant", pad_cval=0, position="uniform", **kwargs):
        self.augmenter = iaa.PadToSquare
        super().__init__(
            pad_mode=pad_mode, pad_cval=pad_cval, position=position, **kwargs
        )


class Resize(ImgAug):
    """Augmenter that resizes images to specified heights and widths.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.imresize_many_images`.

    Parameters
    ----------
    size : 'keep' or int or float or tuple of int or tuple of float or list of int or list of float or imgaug.parameters.StochasticParameter or dict
        The new size of the images.

            * If this has the string value ``keep``, the original height and
              width values will be kept (image is not resized).
            * If this is an ``int``, this value will always be used as the new
              height and width of the images.
            * If this is a ``float`` ``v``, then per image the image's height
              ``H`` and width ``W`` will be changed to ``H*v`` and ``W*v``.
            * If this is a ``tuple``, it is expected to have two entries
              ``(a, b)``. If at least one of these are ``float`` s, a value
              will be sampled from range ``[a, b]`` and used as the ``float``
              value to resize the image (see above). If both are ``int`` s, a
              value will be sampled from the discrete range ``[a..b]`` and
              used as the integer value to resize the image (see above).
            * If this is a ``list``, a random value from the ``list`` will be
              picked to resize the image. All values in the ``list`` must be
              ``int`` s or ``float`` s (no mixture is possible).
            * If this is a ``StochasticParameter``, then this parameter will
              first be queried once per image. The resulting value will be used
              for both height and width.
            * If this is a ``dict``, it may contain the keys ``height`` and
              ``width`` or the keys ``shorter-side`` and ``longer-side``. Each
              key may have the same datatypes as above and describes the
              scaling on x and y-axis or the shorter and longer axis,
              respectively. Both axis are sampled independently. Additionally,
              one of the keys may have the value ``keep-aspect-ratio``, which
              means that the respective side of the image will be resized so
              that the original aspect ratio is kept. This is useful when only
              resizing one image size by a pixel value (e.g. resize images to
              a height of ``64`` pixels and resize the width so that the
              overall aspect ratio is maintained).

    interpolation : imgaug.ALL or int or str or list of int or list of str or imgaug.parameters.StochasticParameter, optional
        Interpolation to use.

            * If ``imgaug.ALL``, then a random interpolation from ``nearest``,
              ``linear``, ``area`` or ``cubic`` will be picked (per image).
            * If ``int``, then this interpolation will always be used.
              Expected to be any of the following:
              ``cv2.INTER_NEAREST``, ``cv2.INTER_LINEAR``, ``cv2.INTER_AREA``,
              ``cv2.INTER_CUBIC``
            * If string, then this interpolation will always be used.
              Expected to be any of the following:
              ``nearest``, ``linear``, ``area``, ``cubic``
            * If ``list`` of ``int`` / ``str``, then a random one of the values
              will be picked per image as the interpolation.
            * If a ``StochasticParameter``, then this parameter will be
              queried per image and is expected to return an ``int`` or
              ``str``.

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
    >>> aug = iaa.Resize(32)

    Resize all images to ``32x32`` pixels.

    >>> aug = iaa.Resize(0.5)

    Resize all images to ``50`` percent of their original size.

    >>> aug = iaa.Resize((16, 22))

    Resize all images to a random height and width within the discrete
    interval ``[16..22]`` (uniformly sampled per image).

    >>> aug = iaa.Resize((0.5, 0.75))

    Resize all any input image so that its height (``H``) and width (``W``)
    become ``H*v`` and ``W*v``, where ``v`` is uniformly sampled from the
    interval ``[0.5, 0.75]``.

    >>> aug = iaa.Resize([16, 32, 64])

    Resize all images either to ``16x16``, ``32x32`` or ``64x64`` pixels.

    >>> aug = iaa.Resize({"height": 32})

    Resize all images to a height of ``32`` pixels and keeps the original
    width.

    >>> aug = iaa.Resize({"height": 32, "width": 48})

    Resize all images to a height of ``32`` pixels and a width of ``48``.

    >>> aug = iaa.Resize({"height": 32, "width": "keep-aspect-ratio"})

    Resize all images to a height of ``32`` pixels and resizes the
    x-axis (width) so that the aspect ratio is maintained.

    >>> aug = iaa.Resize(
    >>>     {"shorter-side": 224, "longer-side": "keep-aspect-ratio"})

    Resize all images to a height/width of ``224`` pixels, depending on which
    axis is shorter and resize the other axis so that the aspect ratio is
    maintained.

    >>> aug = iaa.Resize({"height": (0.5, 0.75), "width": [16, 32, 64]})

    Resize all images to a height of ``H*v``, where ``H`` is the original
    height and ``v`` is a random value sampled from the interval
    ``[0.5, 0.75]``. The width/x-axis of each image is resized to either
    ``16`` or ``32`` or ``64`` pixels.

    >>> aug = iaa.Resize(32, interpolation=["linear", "cubic"])

    Resize all images to ``32x32`` pixels. Randomly use either ``linear``
    or ``cubic`` interpolation.

    """

    def __init__(self, size, interpolation="cubic", **kwargs):
        self.augmenter = iaa.Resize
        super().__init__(size=size, interpolation=interpolation, **kwargs)
