""" Classes to augment images.

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
    >>> image = dt.Value(optics(particle)) 
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

from deeptrack import utils, TORCH_AVAILABLE
from deeptrack.features import Feature
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
        out = []
        for x in elements:
            if time_consistent:
                self.seed()
            out.append(self._augment_element(x, **kwargs))
        return out  
    
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

        # Normalize translate
        if isinstance(translate, (int, float)):
            dx = dy = translate
        else:
            dx, dy = translate

        # Normalize scale
        if isinstance(scale, (int, float)):
            fx = fy = scale
        else:
            fx, fy = scale

        cr = np.cos(rotate)
        sr = np.sin(rotate)
        k = np.tan(shear)

        scale_map = np.array([[1 / fy, 0], [0, 1 / fx]])
        rotation_map = np.array([[cr, sr], [-sr, cr]])
        shear_map = np.array([[1, 0], [-k, 1]])

        matrix = scale_map @ rotation_map @ shear_map

        shape = array.shape
        center = (np.array(shape[:2], dtype=float) -1)/ 2
        offset = center - matrix @ center - np.array([dy, dx], dtype=float)

        forward = np.linalg.inv(matrix)
        forward_offset = - forward @ offset

        self._last_affine = {
            "forward": forward,
            "forward_offset": forward_offset,
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

        if array.ndim not in (2, 3):
            raise ValueError("Affine only supports 2D or 3D tensors.")

        device = array.device
        dtype = array.dtype

        if isinstance(translate, (int, float)):
            dx = dy = float(translate)
        else:
            dx, dy = translate

        if isinstance(scale, (int, float)):
            fx = fy = float(scale)
        else:
            fx, fy = scale

        # --- Build affine matrix exactly as numpy ---
        cr = torch.cos(torch.tensor(rotate, dtype=dtype, device=device))
        sr = torch.sin(torch.tensor(rotate, dtype=dtype, device=device))
        k = torch.tan(torch.tensor(shear, dtype=dtype, device=device))

        scale_map = torch.tensor([[1 / fy, 0], [0, 1 / fx]], dtype=dtype, device=device)

        rotation_map = torch.stack([
            torch.stack([cr, sr]),
            torch.stack([-sr, cr])
        ])

        shear_map = torch.tensor([[1, 0], [-k, 1]], dtype=dtype, device=device)

        matrix = scale_map @ rotation_map @ shear_map

        # --- IMPORTANT: use (H-1)/2 center ---
        H, W = array.shape[:2]
        center = torch.tensor(
            [(H - 1) / 2, (W - 1) / 2],
            dtype=dtype,
            device=device,
        )

        offset = center - matrix @ center - torch.tensor(
            [dy, dx], dtype=dtype, device=device
        )

        # Store for metadata update
        forward = torch.linalg.inv(matrix)
        forward_offset = - forward @ offset

        self._last_affine = {
            "forward": forward,
            "forward_offset": forward_offset,
        }

        # --- Build output pixel coordinate grid (pixel space) ---
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device),
            torch.arange(W, dtype=dtype, device=device),
            indexing="ij",
        )

        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)

        # --- Apply exact SciPy mapping in pixel space ---
        warped = (matrix @ coords.T).T + offset

        y_warp = warped[:, 0]
        x_warp = warped[:, 1]

        # --- Optional wrap mode support ---
        if mode == "wrap":
            x_warp = torch.remainder(x_warp, W)
            y_warp = torch.remainder(y_warp, H)
            padding_mode = "zeros"
        else:
            padding_mode = {
                "reflect": "reflection",
                "nearest": "border",
                "constant": "zeros",
            }.get(mode, "reflection")

        # --- Convert to normalized coordinates ---
        x_norm = 2.0 * x_warp / (W - 1) - 1.0
        y_norm = 2.0 * y_warp / (H - 1) - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.view(1, H, W, 2)

        # --- Prepare tensor for grid_sample ---
        if array.ndim == 2:
            tensor = array.unsqueeze(0).unsqueeze(0)
        else:
            tensor = array.permute(2, 0, 1).unsqueeze(0)

        mode_map = {
            0: "nearest",
            1: "bilinear",
        }

        out = F.grid_sample(
            tensor,
            grid,
            mode=mode_map.get(order, "bilinear"),
            padding_mode=padding_mode,
            align_corners=True,
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
        """Update geometric metadata (positions, directions) after affine transform.

            - All metadata (positions, directions) are NumPy arrays.
            - Backend affects only image resampling, not metadata.
        """

        if not isinstance(element.properties, dict):
            return element

        props = element.properties

        # Nothing to do if no position/direction
        if "position" not in props and "direction" not in props:
            return element

        forward = self._last_affine["forward"]
        forward_offset = self._last_affine["forward_offset"]


        # If backend was torch, convert transform to numpy
        if self.get_backend() == "torch":
            forward = forward.detach().cpu().numpy()
            forward_offset = forward_offset.detach().cpu().numpy()

        # Update positions
        if "position" in props:

            pos = np.asarray(props["position"], dtype=float).copy()
            coords = pos[..., :2]

            transformed = (
                forward @ coords[..., None]
            ).squeeze(-1) + forward_offset

            pos[..., :2] = transformed
            props["position"] = pos

        # Update direction vectors
        if "direction" in props:

            direction = np.asarray(props["direction"], dtype=float).copy()

            coords = direction[..., :2]

            transformed_dir = (
                forward @ coords[..., None]
            ).squeeze(-1)

            direction[..., :2] = transformed_dir
            props["direction"] = direction

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

        shape_full = image.shape

        if ignore_last_dim:
            out = np.empty_like(image)
            for z in range(image.shape[-1]):
                out[..., z] = utils.safe_call(
                    map_coordinates,
                    input=image[..., z],
                    coordinates=coordinates,
                    **kwargs,
                ).reshape(shape)
        else:
            out = utils.safe_call(
                map_coordinates,
                input=image,
                coordinates=coordinates,
                **kwargs,
            ).reshape(shape_full)

        return out
    
    def _get_torch(
        self,
        image: torch.Tensor,
        sigma: float,
        alpha: float,
        ignore_last_dim: bool,
        order: int = 1,
        cval: float = 0.0,
        mode: str = "constant",
        **kwargs,
    ) -> torch.Tensor:


        if image.ndim not in (2, 3):
            raise ValueError("ElasticTransformation only supports 2D or 3D tensors.")

        device = image.device
        dtype = image.dtype

        # Reshape to (N=1, C, H, W)
        if image.ndim == 2:
            H, W = image.shape
            C = 1
            image_ = image.unsqueeze(0).unsqueeze(0)
        else:
            H, W, C = image.shape
            image_ = image.permute(2, 0, 1).unsqueeze(0)

        # Build Gaussian kernel
        def gaussian_kernel_1d(sigma):
            radius = int(3 * sigma)
            coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
            kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            return kernel

        kernel = gaussian_kernel_1d(sigma)
        kernel_x = kernel.view(1, 1, 1, -1)
        kernel_y = kernel.view(1, 1, -1, 1)

        def smooth(field):
            field = field.unsqueeze(0).unsqueeze(0)
            field = F.conv2d(field, kernel_x, padding=(0, kernel_x.shape[-1] // 2))
            field = F.conv2d(field, kernel_y, padding=(kernel_y.shape[-2] // 2, 0))
            return field.squeeze(0).squeeze(0)

        # Create displacement fields
        if ignore_last_dim or C == 1:
            # Shared displacement for all channels
            noise_y = torch.rand((H, W), device=device, dtype=dtype)
            noise_x = torch.rand((H, W), device=device, dtype=dtype)

            delta_y = smooth(noise_y) * alpha
            delta_x = smooth(noise_x) * alpha

            delta_y = delta_y.unsqueeze(0)  # (1,H,W)
            delta_x = delta_x.unsqueeze(0)

        else:
            # Independent displacement per channel
            noise_y = torch.rand((C, H, W), device=device, dtype=dtype)
            noise_x = torch.rand((C, H, W), device=device, dtype=dtype)

            delta_y = torch.stack([smooth(n) for n in noise_y]) * alpha
            delta_x = torch.stack([smooth(n) for n in noise_x]) * alpha

        # Build base grid
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )

        yy = yy.unsqueeze(0).expand(C, -1, -1)
        xx = xx.unsqueeze(0).expand(C, -1, -1)

        yy = yy + delta_y
        xx = xx + delta_x

        if mode == "wrap":
            yy = torch.remainder(yy, H)
            xx = torch.remainder(xx, W)

        # Normalize to [-1, 1]
        xx = 2.0 * xx / (W - 1) - 1.0
        yy = 2.0 * yy / (H - 1) - 1.0

        grid = torch.stack([xx, yy], dim=-1)  # (C,H,W,2)

        # grid_sample expects (N,H,W,2)
        # So we loop over channels if needed
        outputs = []

        mode_map = {
            0: "nearest",
            1: "bilinear",
            3: "bicubic",
        }

        padding_mode = {
            "constant": "zeros",
            "nearest": "border",
            "reflect": "reflection",
            "wrap": "zeros",
        }.get(mode, "zeros")

        for c in range(C):
            out_c = F.grid_sample(
                image_[:, c:c+1],
                grid[c:c+1],
                mode=mode_map.get(order, "bilinear"),
                padding_mode=padding_mode,
                align_corners=True,
            )
            outputs.append(out_c)

        out = torch.cat(outputs, dim=1)

        # Restore original shape
        if image.ndim == 2:
            return out.squeeze(0).squeeze(0)

        return out.squeeze(0).permute(1, 2, 0)



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
        crop: int | list[int] | tuple[int] | Callable[np.ndarray | torch.Tensor, tuple[int]] = (64, 64),        
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


    def _get_xp(
        self: Crop, 
        array: np.ndarray | torch.Tensor,
        crop: int | list[int] | tuple[int] | Callable[np.ndarray | torch.Tensor, tuple[int]],
        crop_mode: str, 
        corner: str | tuple[int] | Callable[[np.ndarray | torch.Tensor], tuple[int]],
        xp: Any, 
        **kwargs,
    ) -> np.ndarray | torch.Tensor:

        # Normalize crop
        if callable(crop):
            crop = crop(array)

        if isinstance(crop, int):
            crop = (crop,) * array.ndim

        crop = [c if c is not None else array.shape[i]
                for i, c in enumerate(crop)]

        if crop_mode == "retain":
            crop_amount = np.array(array.shape) - np.array(crop)
        elif crop_mode == "remove":
            crop_amount = np.array(crop)
        else:
            raise ValueError(f"Unrecognized crop_mode {crop_mode}")

        crop_amount = np.maximum(crop_amount, 0)
        crop_amount = np.minimum(np.array(array.shape) - 1, crop_amount)

        # Determine corner
        if isinstance(corner, str) and corner == "random":
            slice_start = [np.random.randint(int(m) + 1) for m in crop_amount]
        elif callable(corner):
            slice_start = corner(array)
        else:
            slice_start = corner

        slice_start = [int(c) % (int(m) + 1)
                    for c, m in zip(slice_start, crop_amount)]

        slice_end = [
            a - c + s
            for a, s, c in zip(array.shape, slice_start, crop_amount)
        ]

        slices = tuple(
            slice(s0, s1)
            for s0, s1 in zip(slice_start, slice_end)
        )

        out = array[slices]

        # Store for metadata update
        self._last_crop = {
            "start": tuple(slice_start),
        }

        return out
    
    def _update_properties(self, element, old_shape, new_shape, **kwargs):

        if not isinstance(getattr(element, "properties", None), dict):
            return element

        props = element.properties

        if not hasattr(self, "_last_crop"):
            return element

        start_y, start_x = self._last_crop["start"][:2]

        # Update position (y, x)
        if "position" in props and props["position"] is not None:

            pos = np.asarray(props["position"], dtype=float).copy()
            pos[..., 0] -= start_y
            pos[..., 1] -= start_x
            props["position"] = pos

        # Update output_region
        # Convention: (ymin, xmin, ymax, xmax)
        if "output_region" in props and props["output_region"] is not None:

            ymin, xmin, ymax, xmax = props["output_region"]

            new_ymin = ymin + start_y
            new_xmin = xmin + start_x
            new_ymax = new_ymin + new_shape[0]
            new_xmax = new_xmin + new_shape[1]

            props["output_region"] = (
                new_ymin,
                new_xmin,
                new_ymax,
                new_xmax,
            )

        return element


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
        self,
        multiple: PropertyLike[int | tuple[int] | tuple[None]] = 1,
        corner: PropertyLike[str] = "random",
        **kwargs,
    ) -> None:

        kwargs.pop("crop", None)
        kwargs.pop("crop_mode", None)

        def image_to_crop(image):

            shape = image.shape
            mul = self.multiple()

            if not isinstance(mul, (list, tuple, np.ndarray)):
                mul = (mul,) * len(shape)

            new_shape = list(shape)

            for i, (dim, m) in enumerate(zip(shape, mul)):
                if m is not None and m != -1:
                    new_shape[i] = int((dim // m) * m)

            return tuple(new_shape)

        super().__init__(
            crop=lambda: image_to_crop,
            crop_mode="retain",
            corner=corner,
            multiple=multiple,
            **kwargs,
        )


class CropTight(Augmentation):
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
    `get(image: np.ndarray | torch.Tensor, eps: PropertyLike[float], **kwargs) -> np.ndarray | torch.Tensor`
        Abstract method which performs the `CropTight` augmentation.

    """

    def __init__(
        self: CropTight,
        eps: PropertyLike[float] = 1e-10,
        **kwargs
    ) -> None:
        super().__init__(eps=eps, **kwargs)

    def _get_numpy(
        self: CropTight, 
        image: np.ndarray, 
        eps: float, 
        **kwargs
    ) -> np.ndarray:

        mask = image > eps

        keep_z = np.any(mask, axis=(0, 1))
        keep_y = np.any(mask, axis=(1, 2))
        keep_x = np.any(mask, axis=(0, 2))

        ys = np.where(keep_y)[0]
        xs = np.where(keep_x)[0]
        zs = np.where(keep_z)[0]

        if len(ys) == 0 or len(xs) == 0 or len(zs) == 0:
            # nothing survives — return minimal array
            self._last_crop = dict(
                ymin=0, xmin=0, ymax=0, xmax=0, zmin=0, zmax=0
            )
            return image[0:1, 0:1, 0:1]

        ymin = ys[0]
        ymax = ys[-1] + 1

        xmin = xs[0]
        xmax = xs[-1] + 1

        zmin = zs[0]
        zmax = zs[-1] + 1

        self._last_crop = dict(
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
            zmin=zmin,
            zmax=zmax,
        )

        return image[ymin:ymax, xmin:xmax, zmin:zmax]


    def _get_torch(self, image: torch.Tensor, eps: float, **kwargs):

        mask = image > eps

        keep_z = torch.any(mask, dim=(0, 1))
        keep_y = torch.any(mask, dim=(1, 2))
        keep_x = torch.any(mask, dim=(0, 2))

        ys = torch.nonzero(keep_y, as_tuple=True)[0]
        xs = torch.nonzero(keep_x, as_tuple=True)[0]
        zs = torch.nonzero(keep_z, as_tuple=True)[0]

        if len(ys) == 0 or len(xs) == 0 or len(zs) == 0:
            self._last_crop = dict(
                ymin=0, xmin=0, ymax=0, xmax=0, zmin=0, zmax=0
            )
            return image[0:1, 0:1, 0:1]

        ymin = int(ys[0])
        ymax = int(ys[-1]) + 1

        xmin = int(xs[0])
        xmax = int(xs[-1]) + 1

        zmin = int(zs[0])
        zmax = int(zs[-1]) + 1

        self._last_crop = dict(
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
            zmin=zmin,
            zmax=zmax,
        )

        return image[ymin:ymax, xmin:xmax, zmin:zmax]

    def _update_properties(self, element, old_shape, new_shape, **kwargs):

        if not isinstance(element.properties, dict):
            return element

        if "position" in element.properties:
            pos = np.asarray(element.properties["position"], dtype=float).copy()
            pos[0] -= self._last_crop["ymin"]
            pos[1] -= self._last_crop["xmin"]
            element.properties["position"] = pos

        if "output_region" in element.properties:
            ymin, xmin, ymax, xmax = element.properties["output_region"]

            element.properties["output_region"] = (
                ymin + self._last_crop["ymin"],
                xmin + self._last_crop["xmin"],
                ymin + self._last_crop["ymax"],
                xmin + self._last_crop["xmax"],
            )

        return element


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
    `get(image: np.ndarray | torch.Tensor, px: PropertyLike[int], **kwargs) -> np.ndarray | torch.Tensor`
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

    def _get_numpy(
        self: Pad, 
        image: np.ndarray, 
        px: list[int] | tuple[int], 
        mode: str = "constant", 
        cval: float = 0, 
        **kwargs,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Pad (numpy) expects ndarray, got {type(image)}")

        if image.ndim < 2:
            raise ValueError("Pad expects at least 2D array (H, W[, C])")

        if callable(px):
            px = px(image)

        if isinstance(px, int):
            padding = [(px, px)] * image.ndim
        else:
            padding = []
            for idx in range(0, len(px), 2):
                padding.append((px[idx], px[idx + 1]))

        # Fill missing dims with zero padding
        while len(padding) < image.ndim:
            padding.append((0, 0))

        self._last_padding = padding

        return np.pad(
            image,
            padding,
            mode=mode,
            constant_values=cval,
        )
    
    def _get_torch(
        self: Pad, 
        image: torch.Tensor, 
        px: list[int] | tuple[int], 
        mode: str = "constant", 
        cval: float = 0, 
        **kwargs
    ) -> torch.Tensor:

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Pad (torch) expects Tensor, got {type(image)}")

        if image.ndim < 2:
            raise ValueError("Pad expects at least 2D tensor (H, W[, C])")

        if callable(px):
            px = px(image)

        if isinstance(px, int):
            padding = [(px, px)] * image.ndim
        else:
            padding = []
            for idx in range(0, len(px), 2):
                padding.append((px[idx], px[idx + 1]))

            while len(padding) < image.ndim:
                padding.append((0, 0))

        # Store SAME format as numpy
        self._last_padding = padding

        pad_list = []
        for before, after in reversed(padding):
            pad_list.extend([before, after])

        return F.pad(
            image,
            pad_list,
            mode="constant",
            value=cval,
        )

    def _update_properties(
        self,
        element,
        old_shape,
        new_shape,
        **kwargs,
    ):

        if not isinstance(element.properties, dict):
            return element

        padding = self._last_padding

        props = element.properties

        # Shift position
        if "position" in props:
            pos = np.asarray(props["position"], dtype=float).copy()

            # Only shift first two dims (y, x)
            pos[0] += padding[0][0]
            pos[1] += padding[1][0]

            props["position"] = pos

        # Update output_region (ymin, xmin, ymax, xmax)
        if "output_region" in props:
            ymin, xmin, ymax, xmax = props["output_region"]

            new_region = (
                ymin - padding[0][0],
                xmin - padding[1][0],
                ymax + padding[0][1],
                xmax + padding[1][1],
            )

            props["output_region"] = new_region

        return element


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
        self,
        multiple: PropertyLike[int | tuple[int] | tuple[None]] = 1,
        **kwargs,
    ) -> None:

        def amount_to_pad(image: np.ndarray | torch.Tensor) -> list[int]:

            shape = image.shape
            multiple_value = multiple#self.multiple()

            if not isinstance(multiple_value, (list, tuple, np.ndarray)):
                multiple_value = (multiple_value,) * image.ndim

            if len(multiple_value) < image.ndim:
                multiple_value = tuple(multiple_value) + (None,) * (image.ndim - len(multiple_value))

            px = [0] * (image.ndim * 2)

            for i, (dim, mul) in enumerate(zip(shape, multiple_value)):

                if mul is None or mul == -1:
                    continue

                to_add = (-dim) % mul

                before = to_add // 2
                after = to_add - before

                px[2 * i] = before
                px[2 * i + 1] = after

            return px
        
        super().__init__(px=lambda: amount_to_pad, multiple=multiple, **kwargs)


# TODO: add resizing by rescaling
