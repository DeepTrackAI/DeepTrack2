"""Classes to augment images.

This module provides the `augmentations` DeepTrack2 classes
that manipulates a scatterer or array object with various transformations.

When used in a training pipeline, these augmentations synthetically
increase the volume of training data for data-driven learning models.

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

- `Pad`: Pads image with values.

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

from __future__ import annotations
from typing import Callable, Any

import warnings
import random

import numpy as np
import scipy.ndimage as ndimage
# from scipy.ndimage import gaussian_filter
# from scipy.ndimage.interpolation import map_coordinates

from deeptrack import utils, TORCH_AVAILABLE
from deeptrack.features import Feature
from deeptrack.image import Image # TBE
from deeptrack.types import PropertyLike
from deeptrack.scatterers import ScatteredVolume, ScatteredField
from deeptrack.backend import xp, config

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class Augmentation(Feature):
    """Base abstract augmentation class.

    This class provides the template for the other augmentation
    classes to inherit from.

    Parameters
    ----------
    time_consistent: boolean
       Whether to augment all images in a sequence equally.

    Methods
    -------
    `_process_and_get(elements, time_consistent, **kwargs) -> list[list]`
        Augments a list of scatterers or arrays and returns an output of the same type.   

    `_augment_element(element, **kwargs)`
        Augments a single scatterer or array element.

    `_augment_array(array, **kwargs)`
        Augments a single array element, dispatching to the appropriate backend method.

    # `_get_numpy(data, **kwargs) -> np.ndarray`
    #     Abstract method to augment a single array element using numpy.

    # `_get_torch(data, **kwargs) -> torch.Tensor`
    #     Abstract method to augment a single array element using torch.
    
    """

    def __init__(
        self: Augmentation,
        time_consistent: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(time_consistent=time_consistent, **kwargs)


    def _process_and_get(
        self: Augmentation,
        elements: list[ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor] | ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor | None,
        time_consistent: PropertyLike[bool],
        **kwargs
    ) -> list[list]:
        """Augments a list of scatterers or arrays and returns an output of the same type.
        
        This method processes the input elements, which can be a single scatterer or array, a list of scatterers or 
        arrays, or a list of lists of scatterers or arrays (for sequence batches). It applies the augmentation to each 
        element while respecting the `time_consistent` property, which ensures that all images in a sequence are 
        augmented in the same way if set to True.
        
        """
        #  None input
        if elements is None:
            return elements

        # Single element
        if not isinstance(elements, list):
            return self._augment_element(elements, **kwargs)
    
        # list-of-lists (sequence batches)
        if len(elements) > 0 and isinstance(elements[0], list):
            out = []
            for seq in elements:
                if time_consistent:
                    self.seed()
                out.append([self._augment_element(x, **kwargs) for x in seq])
            return out

        # flat list (most common in pipelines)
        if time_consistent:
            self.seed()
        return [self._augment_element(x, **kwargs) for x in elements]
           
    
    def _augment_element(
        self: Augmentation, 
        element: ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor, 
        **kwargs: Any,
    ) -> ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor:
        """Augments a single scatterer or array element.

        """

        if isinstance(element, (ScatteredVolume, ScatteredField)):
            new_volume = element.copy()
            old_shape = new_volume.array.shape
            new_volume.array = self._augment_array(
                new_volume.array, **kwargs
            )
            new_volume = self._update_properties(new_volume, old_shape, new_volume.array.shape, **kwargs)
            return new_volume

        # Arrays
        return self._augment_array(element, **kwargs)

    def _augment_array(self, array, **kwargs):

        # TBE *CM* this is a bit hacky, but it allows us to use the old style get() method for augmentations 
        # that haven't been updated yet, while still allowing new style get() methods to work. We check for 
        # the old style get() method first, and if it exists, we use it. If not, we check for the backend 
        # and use the appropriate method. This way, we can gradually update augmentations to the new style 
        # without breaking existing ones.
        if hasattr(self, "get") and type(self).get is not Augmentation.get:
            return self.get(array, **kwargs)

        backend = self.get_backend()

        if hasattr(self, "_get_xp"):
            xp = np if backend == "numpy" else torch
            return self._get_xp(array, xp=xp, **kwargs)

        if backend == "numpy":
            return self._get_numpy(array, **kwargs)

        if backend == "torch":
            return self._get_torch(array, **kwargs)

        raise RuntimeError(f"Unknown backend: {backend}")
    
    def _update_properties(self, element, old_shape, new_shape, **kwargs):
        return element


class Reuse(Feature):
    """Caches and reuses the output of a feature.

    `Reuse` wraps another feature and avoids recomputing it at every call.
    Instead, it stores up to `storage` previously computed outputs and
    reuses them for a controlled number of calls.

    A new output from `feature` is computed when:

    - The cache contains fewer than `storage` elements, or
    - The internal call counter is a multiple of `uses * storage`.

    Otherwise, one of the cached outputs is returned (uniformly at random).

    This is useful when a computationally expensive feature should only
    be evaluated intermittently while still producing varying outputs
    through reuse.

    Parameters
    ----------
    feature : Feature
        The feature whose output should be cached and reused.

    uses : PropertyLike[int], default=2
        Number of times each cached output is reused before triggering
        a new evaluation cycle.

    storage : PropertyLike[int], default=1
        Maximum number of outputs from `feature` stored in the cache.


    Methods
    -------
    `get(image: np.ndarray | torch.Tensor, uses: PropertyLike[int], storage: PropertyLike[int], **kwargs) -> np.ndarray | torch.Tensor`
        Abstract method which performs the `Reuse` augmentation.

    """

    __distributed__ = False

    def __init__(
        self: Reuse,
        feature: Feature,
        uses: PropertyLike[int] = 2,
        storage: PropertyLike[int] = 1,
        **kwargs
    ):
        super().__init__(uses=uses, storage=storage, **kwargs)
        self.feature = self.add_feature(feature)
        self.counter = 0
        self.cache = []

    def get(
        self: Reuse,
        data: np.ndarray | torch.Tensor,
        uses: int,
        storage: int,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Abstract method which performs the `Reuse` augmentation.

        """

        recompute = (
            len(self.cache) < storage
            or self.counter % (uses * storage) == 0
        )

        if recompute:
            output = self.feature(data)
            self.cache.append(output)
            self.cache = self.cache[-storage:]
        else:
            index = self.counter % storage
            output = self.cache[index]

        self.counter += 1

        return output


class FlipLR(Augmentation):
    """Flips images left-right.

    If scattered volume or field, updates all properties called "position"
    to flip the second index (width axis) of the image.

    Parameters
    ----------
    p: float
       Probability of flipping, default is 0.5.

    augment: bool
       Whether to perform the augmentation.

    Methods
    -------
    `_get_xp(image: np.ndarray | torch.Tensor, xp: Any, augment: PropertyLike[bool], **kwargs) -> np.ndarray | torch.Tensor`
        Abstract method which performs the `FlipLR` augmentation.

    `_update_properties(element: ScatteredVolume | ScatteredField, old_shape: tuple, new_shape: tuple, augment: PropertyLike[bool], **kwargs) -> ScatteredVolume | ScatteredField`
        Abstract method to update the properties of the scattered volume or field.
       
    """

    def __init__(
        self: FlipLR,
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

    def _get_xp(
        self: FlipLR, 
        array: np.ndarray | torch.Tensor, 
        xp: Any, 
        augment: bool, 
        **kwargs,
    ) -> np.ndarray | torch.Tensor:

        if not augment:
            return array
        
        if xp.__name__ == "torch":
            return xp.flip(array, dims=(1,))
        
        return xp.flip(array, axis=1)        

    def _update_properties(
        self: FlipLR, 
        element: ScatteredVolume | ScatteredField, 
        old_shape: tuple, 
        new_shape: tuple, 
        **kwargs,
    ) -> ScatteredVolume | ScatteredField:


        if hasattr(element, "properties"):
            for prop in element.properties:
                if "position" in prop:
                    pos = prop["position"]
                    W = old_shape[1]
                    new_pos = pos.clone() if hasattr(pos, "clone") else pos.copy()
                    new_pos[..., 1] = W - 1 - new_pos[..., 1]
                    prop["position"] = new_pos

        return element


class FlipUD(Augmentation):
    """Flips images up-down.

    If scattered volume or field, updates all properties called "position"
    to flip the first index (height axis) of the image.
    """

    def __init__(
        self: FlipUD,
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

    def _get_xp(
        self,
        array: np.ndarray | torch.Tensor,
        xp: Any,
        augment: bool,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:

        if not augment:
            return array

        if xp.__name__ == "torch":
            return xp.flip(array, dims=(0,))

        return xp.flip(array, axis=0)

    def _update_properties(
        self,
        element: ScatteredVolume | ScatteredField,
        old_shape: tuple,
        new_shape: tuple,
        **kwargs,
    ) -> ScatteredVolume | ScatteredField:

        if hasattr(element, "properties"):
            for prop in element.properties:
                if "position" in prop:
                    pos = prop["position"]
                    H = old_shape[0]

                    new_pos = (
                        pos.clone() if hasattr(pos, "clone") else pos.copy()
                    )

                    new_pos[..., 0] = H - 1 - new_pos[..., 0]
                    prop["position"] = new_pos

        return element


class FlipDiagonal(Augmentation):
    """Flips images along the main diagonal (transpose).

    If scattered volume or field, swaps position coordinates
    (y, x) -> (x, y).
    """

    def __init__(
        self: FlipDiagonal,
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

    def _get_xp(
        self,
        array: np.ndarray | torch.Tensor,
        xp: Any,
        augment: bool,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:

        if not augment:
            return array

        if xp.__name__ == "torch":
            return array.transpose(0, 1)

        return xp.swapaxes(array, 0, 1)

    def _update_properties(
        self,
        element: ScatteredVolume | ScatteredField,
        old_shape: tuple,
        new_shape: tuple,
        **kwargs,
    ) -> ScatteredVolume | ScatteredField:

        if hasattr(element, "properties"):
            for prop in element.properties:
                if "position" in prop:
                    pos = prop["position"]

                    new_pos = (
                        pos.clone() if hasattr(pos, "clone") else pos.copy()
                    )

                    # swap y and x
                    tmp = new_pos[..., 0].clone() if hasattr(new_pos, "clone") else new_pos[..., 0].copy()
                    new_pos[..., 0] = new_pos[..., 1]
                    new_pos[..., 1] = tmp

                    prop["position"] = new_pos

        return element


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
    scale: float or tuple of floats or list of floats or dict
        Scaling factor to use, where ``1.0`` denotes "no change" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.
        If two values are provided (using tuple, list, or dict),
        the two first dimensions of the input are scaled individually.

    translate: float or tuple of floats or list of floats or dict
        Translation in pixels.

    translate_px: float or tuple of floats or list of floats or dict
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

    cval: float
        The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to ``constant``.

    mode: str
        Parameter that defines newly created pixels.
        May take the same values as in :func:`scipy.ndimage.affine_transform`,
        i.e. ``constant``, ``nearest``, ``reflect`` or ``wrap``.

    Methods
    -------
    `_process_properties(properties: dict) -> dict`
        Processes the properties of the image.
    `get(image: Image | np.ndarray, scale: PropertyLike[float], translate: PropertyLike[float], rotate: PropertyLike[float], shear: PropertyLike[float], **kwargs) -> Image`
        Abstract method which performs the `Affine` augmentation.

    """

    def __init__(
        self: Affine,
        scale: PropertyLike[float] = 1,
        translate: PropertyLike[float |  None] = None,
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

    # def _process_properties(
    #     self: Affine,
    #     properties: dict
    # ) -> dict:
        
    #     properties = super()._process_properties(properties)
    #     # Make translate tuple.
    #     translate = properties["translate"]
    #     if isinstance(translate, (float, int)):
    #         translate = (translate, translate)
    #     if isinstance(translate, dict):
    #         translate = (translate["x"], translate["y"])
    #     properties["translate"] = translate

    #     # Make scale tuple.
    #     scale = properties["scale"]
    #     if isinstance(scale, (float, int)):
    #         scale = (scale, scale)
    #     if isinstance(scale, dict):
    #         scale = (scale["x"], scale["y"])
    #     properties["scale"] = scale

    #     return properties

    # def get(
    #     self: Affine,
    #     image: Image | np.ndarray,
    #     scale: float,
    #     translate: float,
    #     rotate: float,
    #     shear: float,
    #     **kwargs
    # ) -> Image:
    #     """Abstract method which performs the `Affine` augmentation.
        
    #     Affine transformations include:
    #     - `Translation`
    #     - `Scaling`
    #     - `Rotation`
    #     - `Shearing`

    #     """    
    #     assert (
    #         image.ndim == 2 or image.ndim == 3
    #     ), "Affine only supports 2-dimensional or 3-dimension inputs, got {0}"\
    #     .format(image.ndim)

    #     dx, dy = translate
    #     fx, fy = scale

    #     cr = np.cos(rotate)
    #     sr = np.sin(rotate)

    #     k = np.tan(shear)

    #     scale_map = np.array([[1 / fx, 0], [0, 1 / fy]])
    #     rotation_map = np.array([[cr, sr], [-sr, cr]])
    #     shear_map = np.array([[1, 0], [-k, 1]])

    #     mapping = scale_map @ rotation_map @ shear_map

    #     shape = image.shape
    #     center = np.array(shape[:2]) / 2

    #     d = center - np.dot(mapping, center) - np.array([dy, dx])

    #     # Clean up kwargs.
    #     kwargs.pop("input", False)
    #     kwargs.pop("matrix", False)
    #     kwargs.pop("offset", False)
    #     kwargs.pop("output", False)

    #     # Call affine_transform.
    #     if image.ndim == 2:
    #         new_image = utils.safe_call(
    #             ndimage.affine_transform,
    #             input=image,
    #             matrix=mapping,
    #             offset=d,
    #             **kwargs,
    #         )

    #         new_image = Image(new_image)
    #         new_image.merge_properties_from(image)
    #         image = new_image

    #     elif image.ndim == 3:
    #         for z in range(shape[-1]):
    #             image[:, :, z] = utils.safe_call(
    #                 ndimage.affine_transform,
    #                 input=image[:, :, z],
    #                 matrix=mapping,
    #                 offset=d,
    #                 **kwargs,
    #             )

    #     # Map positions.
    #     if hasattr(image, "properties"):
    #         inverse_mapping = np.linalg.inv(mapping)
    #         for prop in image.properties:
    #             if "position" in prop:
    #                 position = np.array(prop["position"])

    #                 inverted = (
    #                     np.dot(
    #                         inverse_mapping,
    #                         (position[..., :2] - center + np.array([dy, dx]))[
    #                             ..., np.newaxis
    #                         ],
    #                     )
    #                     .squeeze()
    #                     .transpose()
    #                 ) + center

    #                 position[..., :2] = inverted

    #                 prop["position"] = position

    #     return image

    def _build_mapping(self, array, scale, translate, rotate, shear):

        xp = np if isinstance(array, np.ndarray) else torch

        dx, dy = translate
        fx, fy = scale

        cr = xp.cos(xp.asarray(rotate))
        sr = xp.sin(xp.asarray(rotate))
        k = xp.tan(xp.asarray(shear))

        scale_map = xp.asarray([[1 / fx, 0], [0, 1 / fy]])
        rotation_map = xp.asarray([[cr, sr], [-sr, cr]])
        shear_map = xp.asarray([[1, 0], [-k, 1]])

        mapping = scale_map @ rotation_map @ shear_map

        shape = array.shape
        center = xp.asarray(shape[:2]) / 2
        offset = center - mapping @ center - xp.asarray([dy, dx])

        return mapping, offset

    def _get_numpy(
        self,
        array: np.ndarray,
        scale,
        translate,
        rotate,
        shear,
        order=1,
        cval=0.0,
        mode="reflect",
        **kwargs,
    ):

        from scipy.ndimage import affine_transform

        dx, dy = translate
        fx, fy = scale

        cr = np.cos(rotate)
        sr = np.sin(rotate)
        k = np.tan(shear)

        scale_map = np.array([[1 / fx, 0], [0, 1 / fy]])
        rotation_map = np.array([[cr, sr], [-sr, cr]])
        shear_map = np.array([[1, 0], [-k, 1]])

        matrix = scale_map @ rotation_map @ shear_map

        shape = array.shape
        center = np.array(shape[:2]) / 2
        offset = center - matrix @ center - np.array([dy, dx])

        self._last_affine = {
            "mapping": matrix,
            "offset": offset,
        }

        if array.ndim == 2:

            return affine_transform(
                array,
                matrix=matrix,
                offset=offset,
                order=order,
                mode=mode,
                cval=cval,
            )

        elif array.ndim == 3:

            out = np.empty_like(array)

            for c in range(array.shape[-1]):
                out[..., c] = affine_transform(
                    array[..., c],
                    matrix=matrix,
                    offset=offset,
                    order=order,
                    mode=mode,
                    cval=cval,
                )

            return out

        else:
            raise ValueError("Affine only supports 2D or 3D arrays.")
    
    def _get_torch(
        self,
        array: torch.Tensor,
        scale,
        translate,
        rotate,
        shear,
        order=1,
        cval=0.0,
        mode="reflect",
        **kwargs,
    ):

        import torch.nn.functional as F

        if array.ndim not in (2, 3):
            raise ValueError("Affine only supports 2D or 3D tensors.")

        device = array.device
        dtype = array.dtype

        dx, dy = translate
        fx, fy = scale

        cr = torch.cos(torch.tensor(rotate, dtype=dtype, device=device))
        sr = torch.sin(torch.tensor(rotate, dtype=dtype, device=device))
        k = torch.tan(torch.tensor(shear, dtype=dtype, device=device))

        scale_map = torch.tensor([[1 / fx, 0], [0, 1 / fy]], dtype=dtype, device=device)
        rotation_map = torch.stack([
            torch.stack([cr, sr]),
            torch.stack([-sr, cr])
        ])
        shear_map = torch.tensor([[1, 0], [-k, 1]], dtype=dtype, device=device)

        matrix = scale_map @ rotation_map @ shear_map

        H, W = array.shape[:2]
        center = torch.tensor([H / 2, W / 2], dtype=dtype, device=device)

        offset = center - matrix @ center - torch.tensor([dy, dx], dtype=dtype, device=device)

        # Store transform BEFORE grid_sample
        self._last_affine = {
            "mapping": matrix,
            "offset": offset,
        }

        # Build grid
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device),
            torch.arange(W, dtype=dtype, device=device),
            indexing="ij",
        )

        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)

        inv_matrix = torch.linalg.inv(matrix)

        warped = (inv_matrix @ (coords - offset).T).T

        y_warp = warped[:, 0]
        x_warp = warped[:, 1]

        # Align_corners=True assumption
        x_norm = 2 * x_warp / (W - 1) - 1
        y_norm = 2 * y_warp / (H - 1) - 1

        grid = torch.stack([x_norm, y_norm], dim=-1).view(1, H, W, 2)

        if array.ndim == 2:
            tensor = array.unsqueeze(0).unsqueeze(0)
        else:
            tensor = array.permute(2, 0, 1).unsqueeze(0)

        mode_map = {
            0: "nearest",
            1: "bilinear",
        }

        padding_mode = {
            "reflect": "reflection",
            "nearest": "border",
            "constant": "zeros",
        }.get(mode, "reflection")

        out = F.grid_sample(
            tensor,
            grid,
            mode=mode_map.get(order, "bilinear"),
            padding_mode=padding_mode,
            align_corners=True,  # DO NOT CHANGE
        )

        if array.ndim == 2:
            return out.squeeze(0).squeeze(0)

        return out.squeeze(0).permute(1, 2, 0)

    def _update_properties(
        self,
        element,
        old_shape,
        new_shape,
        **kwargs,
    ):
        """
        Update geometric metadata (positions, directions) after affine transform.

        Convention A:
            - All metadata (positions, directions) are NumPy arrays.
            - Backend affects only image resampling, not metadata.
        """

        if not hasattr(element, "properties"):
            return element

        # Retrieve mapping and offset used for resampling
        mapping = self._last_affine["mapping"]
        offset = self._last_affine["offset"]

        # If backend was torch, convert transform to numpy
        if self.get_backend() == "torch":
            mapping = mapping.detach().cpu().numpy()
            offset = offset.detach().cpu().numpy()

        inverse_mapping = np.linalg.inv(mapping)

        for prop in element.properties:

            # Update positions
            if "position" in prop:

                pos = prop["position"]

                # Enforce numpy metadata convention
                if not isinstance(pos, np.ndarray):
                    pos = np.asarray(pos)

                new_pos = pos.copy()

                coords = new_pos[..., :2]

                transformed = (
                    inverse_mapping @ (coords - offset)[..., None]
                ).squeeze(-1)

                new_pos[..., :2] = transformed
                prop["position"] = new_pos

            # Update direction vectors
            if "direction" in prop:

                direction = prop["direction"]

                if not isinstance(direction, np.ndarray):
                    direction = np.asarray(direction)

                new_dir = direction.copy()

                coords = new_dir[..., :2]

                transformed_dir = (
                    inverse_mapping @ coords[..., None]
                ).squeeze(-1)

                new_dir[..., :2] = transformed_dir
                prop["direction"] = new_dir

        return element


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

    Methods
    -------
    `get(image: Image | np.ndarray, sigma: PropertyLike[float], alpha: PropertyLike[float], ignore_last_dim: PropertyLike[bool], **kwargs) -> Image`
        Abstract method which performs the `ElasticTransformation` augmentation.

    """

    def __init__(
        self: ElasticTransformation,
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

    def _get_numpy(
        self: ElasticTransformation,
        image: np.ndarray,
        sigma: float,
        alpha: float,
        ignore_last_dim: bool,
        **kwargs
    ) -> np.ndarray:
        """Abstract method which performs the `ElasticTransformation` augmentation.

        """

        from scipy.ndimage import gaussian_filter, map_coordinates

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

        return image


class Crop(Augmentation):
    """Crops a regions of an image.

    Parameters
    ----------
    feature: Feature or list of Features
        Feature(s) to augment.

    crop: int or tuple of ints or list of ints or Callable[Image]->tuple of ints
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

    Methods
    -------
    `get(image: Image | np.ndarray, corner: PropertyLike[str], crop: PropertyLike[int], crop_mode: PropertyLike[str], **kwargs) -> Image`
        Abstract method which performs the `Crop` augmentation.

    """

    def __init__(
        self: Crop,
        *args,
        crop: int | list[int] | tuple[int] | Callable[[Image], tuple[int]] = (64, 64),        
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
        self: Crop,
        image: Image | np.ndarray,
        corner: str,
        crop: int | list[int] | tuple[int],
        crop_mode: str,
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
    multiple: int or tuple of ints or tuple of none
        Images will be cropped down until their width is a multiple of
        this value. If a tuple, it is assumed to be a multiple per axis.
        A value of None or -1 indicates to skip that axis.
    
    corner: str
        Top left corner of the cropped region. Can be a tuple of ints,
        a function that returns a tuple of ints or the string random.
        If corner is placed so that the cropping cannot be performed,
        the modulo of the corner with the allowed region is used.

    """

    def __init__(
        self: CropToMultiplesOf,
        multiple: PropertyLike[int | tuple[int] | tuple[None]] = 1,
        corner: PropertyLike[str] = "random",
        **kwargs
    ) -> None:
        
        kwargs.pop("crop", False)
        kwargs.pop("crop_mode", False)

        def image_to_crop(
            image: Image | np.ndarray
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

    Methods
    -------
    `get(image: Image | np.ndarray, eps: PropertyLike[float], **kwargs) -> Image`
        Abstract method which performs the `CropTight` augmentation.

    """

    def __init__(
        self: CropTight,
        eps: PropertyLike[float] = 1e-10,
        **kwargs
    ) -> None:
        super().__init__(eps=eps, **kwargs)

    def get(
        self: CropTight,
        image: Image | np.ndarray,
        eps: float,
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
    """Pads an image by adding extra pixels along specified axes.

    This augmentation uses `numpy.pad` internally but redefines `pad_width` 
    as `px`, allowing padding along multiple axes (left, right, top, bottom, 
    before_axis_3, after_axis_3, ...).

    Parameters
    ----------
    px : list of ints or tuple of ints
        Amount of padding for each axis, specified as a tuple (left, right, 
        top, bottom, etc.).

    mode : str
        Padding mode, same as in `numpy.pad`.

    cval : float
        Value to fill in new pixels, same as in `numpy.pad`.

    Methods
    -------
    `get(image: Image | np.ndarray, px: PropertyLike[int], **kwargs) -> Image`
        Abstract method which performs the `Pad` augmentation.
    `_image_wrap_process_and_get(images: list[Image] | list[np.ndarray], **kwargs) -> list[Image]`
        Simple method which wraps an `Image` in a `list`.

    Returns
    -------
    Image
    The padded image.

    """

    def __init__(
        self: Pad,
        px: list[int] | tuple[int] = (0, 0, 0, 0),
        mode: PropertyLike[str] = "constant",
        cval: PropertyLike[float] = 0,
        **kwargs
    ) -> None:
        super().__init__(px=px, mode=mode, cval=cval, **kwargs)


    def _get_numpy(self, image, px, mode="constant", cval=0, **kwargs):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Pad (numpy) expects ndarray, got {type(image)}")

        if image.ndim < 2:
            raise ValueError("Pad expects at least 2D array (H, W[, C])")

        spatial_ndim = image.ndim - 1  # channel-last

        if callable(px):
            px = px(image)

        if isinstance(px, int):
            padding = [(px, px)] * spatial_ndim
        else:
            if len(px) != 2 * spatial_ndim:
                raise ValueError(
                    f"px must have length {2 * spatial_ndim} for channel-last data"
                )
            padding = [(px[i], px[i + 1]) for i in range(0, len(px), 2)]

        # Do NOT pad channels
        padding.append((0, 0))
        out = np.pad(image, pad_width=padding, mode=mode, constant_values=cval)
        return out
    

    def _get_torch(self, image, px, mode="constant", cval=0, **kwargs):

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Pad (torch) expects Tensor, got {type(image)}")

        if image.ndim < 2:
            raise ValueError("Pad expects at least 2D tensor (H, W[, C])")

        spatial_ndim = image.ndim - 1  # channel-last

        if callable(px):
            px = px(image)

        if isinstance(px, int):
            pad_pairs = [(px, px)] * spatial_ndim
        else:
            if len(px) != 2 * spatial_ndim:
                raise ValueError(
                    f"px must have length {2 * spatial_ndim} for channel-last data"
                )
            pad_pairs = [(px[i], px[i + 1]) for i in range(0, len(px), 2)]

        # torch wants reverse order, flattened
        # also: do NOT pad channels
        pad_pairs.append((0, 0))
        pad = [v for pair in reversed(pad_pairs) for v in pair]

        if mode == "constant":
            return F.pad(image, pad, mode="constant", value=cval)

        return F.pad(image, pad, mode=mode)



    # def get(
    #     self: Pad,
    #     image: Image | np.ndarray,
    #     px: int,
    #     **kwargs
    # ) -> Image:
    #     """Abstract method which performs the `Pad` augmentation.

    #     """    
    #     padding = []
    #     if callable(px):
    #         px = px(image)
    #     elif isinstance(px, int):
    #         padding = [(px, px)] * image.ndim

    #     for idx in range(0, len(px), 2):
    #         padding.append((px[idx], px[idx + 1]))

    #     while len(padding) < image.ndim:
    #         padding.append((0, 0))

    #     return utils.safe_call(
    #         np.pad,
    #         positional_args=(image, padding),
    #         **kwargs,
    #         )
 

    # def _image_wrap_process_and_get(
    #     self: Pad,
    #     images: list[Image] | list[np.ndarray],
    #     **kwargs
    # ) -> list[Image]:
    #     """Simple method which wraps an `Image` in a `list`.
        
    #     """
    #     results = [self.get(image, **kwargs) for image in images]

    #     # for idx, result in enumerate(results):
    #     #    if isinstance(result, tuple):
    #     #    results[idx] = Image(result[0]).merge_properties_from(images[idx])
    #     #    else:
    #     #    Image(results[idx]).merge_properties_from(images[idx])
    #     return results


class PadToMultiplesOf(Pad):
    """Pad images until their height/width is a multiple of a value.

    Parameters
    ----------

    multiple: int or tuple of int or tuple of none
        Images will be padded until their width is a multiple of
        this value. If a tuple, it is assumed to be a multiple per axis.
        A value of None or -1 indicates to skip that axis.

    """

    def __init__(
        self: PadToMultiplesOf,
        multiple: PropertyLike[int | tuple[int] | tuple[None]] = 1,
        **kwargs
    ) -> None:
        
        def amount_to_pad(
            image: Image | np.ndarray
        ) -> list[int]:
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
