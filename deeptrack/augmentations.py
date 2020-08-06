''' Features that augment images

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
'''

from deeptrack.features import Feature
from deeptrack.image import Image
import numpy as np
from typing import Callable
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class Augmentation(Feature):
    '''Base abstract augmentation class.

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
    feature : Feature, optional
        The parent feature. If None, the 
    load_size : int
        Number of results to resolve from the parent feature.
    updates_per_reload : int
        Number of times `.update()` is called before resolving new results
        from the parent feature.
    update_properties : Callable or None
        Function called on the output of the method `.get()`. Overrides
        the default behaviour, allowing full control over how to update
        the properties of the output to account for the augmentation.
    '''

    __distributed__ = False
    def __init__(self, 
                 feature: Feature = None, 
                 load_size: int = 1, 
                 updates_per_reload: int = 1, 
                 update_properties: Callable or None = None, 
                 **kwargs):
        self.feature = feature 

        def get_preloaded_results(load_size, number_of_updates):
            # Dummy property that loads results from the parent when
            # number of properties=0
            if number_of_updates == 0 and self.feature:
                self.preloaded_results = self._load(load_size)

            return None
        
        def get_number_of_updates(updates_per_reload=1):
            # Updates the number of updates. The very first update is not counted.
            if not hasattr(self.properties["number_of_updates"], "_current_value"):
                return 0
            return (self.properties["number_of_updates"].current_value + 1) % updates_per_reload

        if not update_properties:
            update_properties = self.update_properties
        
        super().__init__(
            load_size=load_size, 
            updates_per_reload=updates_per_reload, 
            index=kwargs.pop("index", False) or (lambda load_size: np.random.randint(load_size)), 
            number_of_updates=get_number_of_updates,
            preloaded_results=kwargs.pop("preloaded_results", False) or get_preloaded_results,
            update_properties=lambda: update_properties,
            **kwargs)


    def _process_and_get(self, *args, update_properties=None, index=0, **kwargs):
        # Loads a result from storage
        if self.feature:
            image_list_of_lists = self.preloaded_results[index]
        else:
            image_list_of_lists = args[0]

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
                new_list_of_lists.append([
                    [Image(self.get(Image(image), **kwargs)).merge_properties_from(image) for image in image_list]
                ])
            else: 
                new_list_of_lists.append(
                    Image(self.get(Image(image_list), **kwargs)).merge_properties_from(image_list)
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
    

    def _load(self, load_size):
        # Resolves parent and stores result
        preloaded_results = []
        for _ in range(load_size):
            if isinstance(self.feature, list):  
                [_feature.update() for _feature in self.feature]
                list_of_results = [_feature.resolve(_augmentation_index=index) for index, _feature in enumerate(self.feature)]
                preloaded_results.append(list_of_results)
            else:
                self.feature.update()
                result = self.feature.resolve(_augmentation_index=0)
                preloaded_results.append(result)
        return preloaded_results

    def update_properties(*args, **kwargs):
        pass



class PreLoad(Augmentation):
    '''Simple storage with no augmentation.

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
    
    '''

    def get(self, image, **kwargs):
        return image



class Crop(Augmentation):
    ''' Crops a regions of an image.

    Parameters
    ----------
    feature : feature or list of features
        Feature(s) to augment.
    corner : tuple of ints or Callable[Image] or "random"
        Top left corner of the cropped region. Can be a tuple of ints,

    '''
    def __init__(self, *args, corner="random", crop_size=(64, 64), **kwargs):
        super().__init__(*args, corner=corner, crop_size=crop_size, **kwargs)
    
    def get(self, image, corner, crop_size, **kwargs):
        if corner == "random":
            # Ensure seed is consistent
            slice_start = np.random.randint([0] * len(crop_size), np.array(image.shape[:len(crop_size)]) - crop_size)
            
        elif callable(corner):
            slice_start = corner(image)

        else:
            slice_start = corner
    
        slices = tuple([slice(slice_start_i, slice_start_i + crop_size_i) for slice_start_i, crop_size_i in zip(slice_start, crop_size)])
        
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
    ''' Flips images left-right.

    Updates all properties called "position" to flip the second index.
    '''

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
                    new_position = (position[0], image.shape[1] - position[1] - 1, *position[2:])
                    prop["position"] = new_position



class FlipUD(Augmentation):
    ''' Flips images up-down.

    Updates all properties called "position" by flipping the first index.
    '''

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
    ''' Flips images along the main diagonal.

    Updates all properties called "position" by swapping the first and second index.
    '''

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
    ''' Interfaces imagaug augmentations.
    '''

    def __init__(self, *args, augmentation, **kwargs):
        self.augmentation = augmentation
        super().__init__(*args, **kwargs)
    
    def get(self, image, **kwargs):
        argument_names = get_kwarg_names(self.augmentation)
        class_options = {}
        for key in argument_names:
            if key in kwargs:
                class_options[key] = kwargs[key]
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ia.seed(entropy=np.random.randint(2**31 - 1))
            return self.augmentation(**class_options)(image=image)

class Pad(ImgAug):
    """Pad images, i.e. adds columns/rows of pixels to them.

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
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, augmentation=iaa.Pad, **kwargs)

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
    choice and will lead to a water-like effect. For a complete list of
    allowed parameters, please see the imgaug documentation.
    Code here was initially inspired by
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    For a detailed explanation, see ::
        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the distortion field. Higher values mean that pixels are
        moved further with respect to the distortion field's direction.
        Should be a value from interval ``[1.0, inf]``. Set this to around
        ``10 * sigma`` for visible effects.
            * If number, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.
    sigma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Corresponds to the standard deviation of the gaussian kernel used
        in the original algorithm. Here, for performance reasons, it denotes
        half of an average blur kernel size. (Only for ``sigma<1.5`` is
        a gaussian kernel actually used.)
        Higher values (for ``128x128`` images around 5.0) lead to more
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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, augmentation=iaa.ElasticTransformation, **kwargs)



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
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, augmentation=iaa.Affine, **kwargs)