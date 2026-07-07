"""Augmentation utilities.

This module provides feature classes for applying spatial transformations
and augmentations to images, arrays, scattered fields, or scattered volumes.
Supported transformations include flipping, affine transformations,
elastic deformations, cropping, and padding.

When used in a training pipeline, these augmentations synthetically
increase the amount of training data for data-driven learning models.

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

    To avoid redundant computations, the `Reuse` feature caches a fixed number
    of outputs (`storage`) and reuses each cached output a specified number of
    times (`uses`) before recomputing.

- **Cropping**

    Enables different methods to crop an image. Region-specific cropping,
    cropping based on multiples of the height/width of an image, and crop to
    remove empty space at edges of an image.

- **Padding**

    Padding operations allow you to extend the shape of an image by
    adding extra pixels around its edges, which is essential for ensuring
    that the shape of the image stays consistent.

Module Structure
----------------
Classes:

- `Augmentation`: Base class for augmentations.
- `Reuse`: Stores and reuses feature outputs.
- `FlipLR`: Flips image left to right.
- `FlipUD`: Flips an image up-down.
- `FlipDiagonal`: Flips image along the diagonal.
- `Affine`: Translation, scaling, rotation, shearing.
- `ElasticTransformation`: Transform using a displacement field.
- `Crop`: Crop regions of an image.
- `CropToMultiplesOf`: Crops image until height/width is multiple of a value.
- `CropTight`: Crops an array to remove empty space along its edges.
- `Pad`: Pads image with values.
- `PadToMultiplesOf`: Pad images until height/width is a multiple of a value.

Examples
--------
>>> import deeptrack as dt

Flip an image of a particle up-down then flips left-right:

>>> particle = dt.PointParticle(intensity=1)
>>> optics = dt.Fluorescence()
>>> image = optics(particle) >> dt.FlipUD(p=1.0) >> dt.FlipLR(p=1.0)
>>> image.plot();

Reuse the output of a pipeline twice, augmented randomly by FlipLR.

>>> import matplotlib.pyplot as plt
>>>
>>> particle = dt.PointParticle(intensity=1)
>>> optics = dt.Fluorescence()
>>> base = optics(particle)
>>> pipeline = dt.Reuse(base, uses=2) >> dt.FlipLR()
>>>
>>> fig, ax = plt.subplots(1, 8, figsize=(12, 3))
>>> for i in range(8):
>>>     img = pipeline.new()
>>>     ax[i].imshow(img, cmap="gray")
>>>     ax[i].axis("off")
>>> plt.tight_layout()
>>> plt.show()

"""

from __future__ import annotations
from typing import Callable, Any

import numpy as np

from deeptrack import utils, TORCH_AVAILABLE
from deeptrack.backend.core import DeepTrackDataDict
from deeptrack.features import Feature
from deeptrack.types import PropertyLike
from deeptrack.optical.scatterers import ScatteredVolume, ScatteredField
from deeptrack.backend import xp, config

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


__all__ = [
    "Augmentation",
    "Reuse",
    "FlipLR",
    "FlipUD",
    "FlipDiagonal",
    "Affine",
    "ElasticTransformation",
    "Crop",
    "CropToMultiplesOf",
    "CropTight",
    "Pad",
    "PadToMultiplesOf",
]


class Augmentation(Feature):
    """Base class for augmentation features.

    This class defines the interface for spatial augmentations applied to
    arrays, scattered fields, or scattered volumes. Subclasses implement the
    actual transformation logic while this class handles dispatching,
    batching, and backend selection.

    Supported inputs include:
    - NumPy arrays
    - Torch tensors
    - `ScatteredVolume` and `ScatteredField` objects

    When applied to scattered objects, both the underlying array and relevant
    metadata (e.g., positions) may be updated.

    Parameters
    ----------
    time_consistent: bool, optional
        If `True`, the same augmentation parameters are applied to all elements
        in a sequence. This is useful for time-series data where each frame
        must undergo the same transformation. Defaults to `False`.

    Methods
    -------
    `_process_and_get(elements, time_consistent, **kwargs) -> list`
        Augments a list of scatterers or arrays and returns an output of the
        same type.
    `_augment_element(element, **kwargs) -> volyme | field | array | tensor `
        Augments a single scatterer or array element.
    `_augment_array(array, **kwargs) -> array | tensor`
        Augments a single array element, dispatching to the appropriate backend
        method.
    `_get_xp(array, xp, **kwargs) -> array | tensor`
        Backend-agnostic implementation using the provided array module
        (`numpy` or `torch`).
    `_get_numpy(array, **kwargs) -> array`
        NumPy-specific implementation.
    `_get_torch(array, **kwargs) -> tensor`
        PyTorch-specific implementation.
    `_update_properties(element, old_shape, new_shape, ...) -> volume | field`
        Updates the properties of a `ScatteredVolume` or `ScatteredField`
        after the array has been augmented.

    Notes
    -----
    Subclasses typically implement one of the following:
    - `_get_xp(array, xp, **kwargs)`
    - `_get_numpy(array, **kwargs)`
    - `_get_torch(array, **kwargs)`
    If `._get_xp()` is implemented, it will be used for both backends.

    """

    def __init__(
        self: Augmentation,
        time_consistent: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Augmentation feature.

        This constructor initializes the augmentation feature with the
        specified parameters. The `time_consistent` parameter determines
        whether the same augmentation parameters are applied to all elements in
        a sequence, which is important for time-series data.

        Parameters
        ----------
        time_consistent: bool, optional
            If `True`, the same augmentation parameters are applied to all
            elements in a sequence. This is useful for time-series data where
            each frame must undergo the same transformation. Defaults to
            `False`.
        **kwargs: Any
            Keyword arguments used to configure the feature. Each keyword
            argument is wrapped as a `Property` and added to the feature's
            `properties` attribute. These properties are resolved dynamically
            at call time and passed to the `.get()` method.

        """

        super().__init__(time_consistent=time_consistent, **kwargs)

    def _process_and_get(
        self: Augmentation,
        elements: (
            list[ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor]
            | ScatteredVolume
            | ScatteredField
            | np.ndarray
            | torch.Tensor
            | None
        ),
        time_consistent: PropertyLike[bool],
        **kwargs: Any,
    ) -> (
        list[ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor]
        | ScatteredVolume
        | ScatteredField
        | np.ndarray
        | torch.Tensor
        | None
    ):
        """Apply the augmentation to the provided elements.

        The input may be a single element, a list of elements, or a list of
        lists of elements (for sequence batches). The augmentation is applied
        to each element while respecting the `time_consistent` property,
        which ensures that all images in a sequence are augmented in the same
        way if set to True.

        Parameters
        ----------
        elements: list[...] | ... | None
            Elements to be augmented.
        time_consistent: PropertyLike[bool]
            If True, the same augmentation parameters are applied to all
            elements in a sequence.
        **kwargs: Any
            Additional keyword arguments passed to the augmentation methods.

        Returns
        -------
        Same type as input
            The augmented elements.

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
        """Augment a single element.

        If the element is a `ScatteredVolume` or `ScatteredField`, the
        underlying array is augmented and the associated metadata (e.g.,
        positions) may be updated accordingly.
        For NumPy arrays or Torch tensors, the augmentation is applied directly
        to the array.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor
            The element to be augmented.
        **kwargs: Any
            Additional keyword arguments passed to the augmentation methods.

        Returns
        -------
        ScatteredVolume | ScatteredField | np.ndarray | torch.Tensor
            The augmented element.

        """

        if isinstance(element, (ScatteredVolume, ScatteredField)):
            new_volume = element.copy()
            old_shape = new_volume.array.shape
            new_volume.array = self._augment_array(new_volume.array, **kwargs)
            new_volume = self._update_properties(
                new_volume,
                old_shape,
                new_volume.array.shape,
                **kwargs,
            )
            return new_volume

        # Arrays
        return self._augment_array(element, **kwargs)

    def _augment_array(
        self: Augmentation,
        array: np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Augment a single array.

        This method dispatches the augmentation to the appropriate backend
        implementation depending on the active DeepTrack backend.
        Subclasses typically implement one of the following methods:
        - `_get_xp(array, xp, **kwargs)`
            Backend-agnostic implementation using either `numpy` or `torch`.
        - `_get_numpy(array, **kwargs)`
            NumPy-specific implementation.
        - `_get_torch(array, **kwargs)`
            PyTorch-specific implementation.
        If `_get_xp` is defined it takes precedence and will be used for both
        backends.

        Parameters
        ----------
        array: np.ndarray | torch.Tensor
            The array to augment.
        **kwargs: Any
            Additional keyword arguments passed to the augmentation method.

        Returns
        -------
        np.ndarray | torch.Tensor
            The augmented array.

        """

        backend = self.get_backend()

        if hasattr(self, "_get_xp"):
            xp = np if backend == "numpy" else torch
            return self._get_xp(array, xp=xp, **kwargs)

        if backend == "numpy":
            return self._get_numpy(array, **kwargs)

        if backend == "torch":
            return self._get_torch(array, **kwargs)

        raise RuntimeError(f"Unknown backend: {backend}")

    def _update_properties(
        self: Augmentation,
        element: ScatteredVolume | ScatteredField,
        old_shape: tuple,
        new_shape: tuple,
        **kwargs: Any,
    ) -> ScatteredVolume | ScatteredField:
        """Update metadata after an augmentation.

        This method is called after the array contained in a `ScatteredVolume`
        or `ScatteredField` has been augmented. Subclasses may override this
        method to update spatial metadata (e.g., particle positions) when the
        geometry of the array changes.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            The scattered object whose array has been augmented.
        old_shape: tuple[int, ...]
            Shape of the array before augmentation.
        new_shape: tuple[int, ...]
            Shape of the array after augmentation.
        **kwargs: Any
            Additional keyword arguments passed from the augmentation method.

        Returns
        -------
        ScatteredVolume | ScatteredField
            The updated scattered object.

        """

        return element


class Reuse(Feature):
    """Cache and reuse the output of another feature.

    `Reuse` wraps a feature and avoids recomputing it at every evaluation.
    Instead, it stores up to `storage` previously computed outputs and
    reuses them multiple times.

    The cache is filled until it contains `storage` outputs. Afterwards,
    each cached output is reused `uses` times before a new evaluation
    cycle begins.

    This is useful when an expensive feature should only be evaluated
    occasionally while still producing varying outputs through reuse.

    Parameters
    ----------
    feature: Feature
        Feature whose output should be cached.
    uses: PropertyLike[int], optional
        Number of times each cached output is reused. Defaults to `2`.
    storage: PropertyLike[int], optional
        Maximum number of cached outputs stored. Defaults to `1`.

    Methods
    -------
    `get(data, uses, storage, **kwargs) -> np.ndarray | torch.Tensor`
        Implements the caching and reuse logic. Evaluates the wrapped feature
        only when necessary and otherwise returns cached outputs.

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> particle = dt.PointParticle(
    ...     intensity=1,
    ...     position=lambda: np.random.rand(2) * 64
    ... )
    >>> optics = dt.Fluorescence()
    >>> base = optics(particle)

    >>> pipeline = dt.Reuse(base, uses=2, storage=2)
    >>> fig, ax = plt.subplots(1, 8, figsize=(12, 3))
    >>> for i in range(8):
    ...     ax[i].imshow(pipeline.new(), cmap="gray")
    ...     ax[i].axis("off")
    >>> plt.show();

    """

    __distributed__ = False

    def __init__(
        self: Reuse,
        feature: Feature,
        uses: PropertyLike[int] = 2,
        storage: PropertyLike[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(uses=uses, storage=storage, **kwargs)
        self.feature = self.add_feature(feature)
        self.counter = 0
        self.cache = []
        self._cache_dependency_data = []

    @staticmethod
    def _copy_data_dict(data: DeepTrackDataDict) -> DeepTrackDataDict:
        """Create a shallow copy of a node data dictionary."""

        copied = DeepTrackDataDict()
        for key, data_object in data.dict.items():
            copied.create_index(key)
            copied[key].store(data_object.current_value())
            if not data_object.is_valid():
                copied[key].invalidate()

        return copied

    def _snapshot_feature_data(self: Reuse) -> dict[Any, DeepTrackDataDict]:
        """Snapshot cached values of the wrapped feature graph."""

        return {
            dependency: self._copy_data_dict(dependency.data)
            for dependency in self.feature.recurse_dependencies()
        }

    def _restore_feature_data(
        self: Reuse,
        snapshot: dict[Any, DeepTrackDataDict],
    ) -> None:
        """Restore cached values of the wrapped feature graph."""

        for dependency, data in snapshot.items():
            dependency.data = self._copy_data_dict(data)

    def get(
        self: Reuse,
        data: np.ndarray | torch.Tensor,
        uses: int,
        storage: int,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Return a cached output or recompute the wrapped feature.

        The cache stores up to `storage` outputs from the wrapped feature.
        Each cached output is reused `uses` times before a new evaluation
        cycle begins. Cached outputs are returned in cyclic order.

        Parameters
        ----------
        data: np.ndarray | torch.Tensor
            Input passed to the wrapped feature.
        uses: int
            Number of times each cached output is reused.
        storage: int
            Maximum number of outputs stored in the cache.
        **kwargs: Any
            Additional keyword arguments passed to the wrapped feature when
            recomputation is necessary.

        Returns
        -------
        np.ndarray | torch.Tensor
            Cached output if reuse is possible, otherwise a newly computed
            output from the wrapped feature.

        """

        recompute = (
            len(self.cache) < storage or self.counter % (uses * storage) == 0
        )

        if recompute:
            output = self.feature(data)
            self.cache.append(output)
            self._cache_dependency_data.append(self._snapshot_feature_data())
            if len(self.cache) > storage:
                self.cache.pop(0)
                self._cache_dependency_data.pop(0)
        else:
            index = self.counter % storage
            output = self.cache[index]
            self._restore_feature_data(self._cache_dependency_data[index])

        self.counter += 1

        return output


class FlipLR(Augmentation):
    """Flip images left-right.

    If the input is a `ScatteredVolume` or `ScatteredField`, the underlying
    array is flipped along the width axis and any `"position"` metadata is
    updated accordingly.

    Parameters
    ----------
    p: PropertyLike[float], optional
        Probability of performing the flip. Defaults to `0.5`.
    augment: PropertyLike[bool] | None
        Boolean controlling whether the augmentation is applied. If `None`,
        the augmentation is performed with probability `p`.

    Methods
    -------
    `_get_xp(image, xp, augment, **kwargs) -> np.ndarray | torch.Tensor`
        Abstract method which performs the `FlipLR` augmentation.
    `_update_properties(...) -> ScatteredVolume | ScatteredField`
        Abstract method to update the properties of the scattered volume or
        field.

    Examples
    --------
    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(intensity=1)
    >>> optics = dt.Fluorescence()
    >>> image = optics(particle) >> dt.FlipLR(p=1.0)
    >>> image.plot();

    """

    def __init__(
        self: FlipLR,
        p: PropertyLike[float] = 0.5,
        augment: PropertyLike[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the FlipLR augmentation.

        This constructor initializes the `FlipLR` augmentation with the
        specified parameters. The `p` parameter controls the probability of
        performing the flip, while the `augment` parameter can be used to
        directly control whether the augmentation is applied. If `augment` is
        set to `None`, the augmentation will be performed with probability `p`.
        This allows for flexible control over when the flip is applied, making
        it suitable for use in data augmentation pipelines where random
        transformations are desired.

        Parameters
        ----------
        p: PropertyLike[float], optional
            Probability of performing the flip. Defaults to `0.5`.
        augment: PropertyLike[bool] | None
            Boolean controlling whether the augmentation is applied. If `None`,
            the augmentation is performed with probability `p`.
         **kwargs: Any
            Additional keyword arguments used to configure the feature. Each
            keyword argument is wrapped as a `Property` and added to the
            feature's `properties` attribute. These properties are resolved
            dynamically at call time and passed to the `.get()` method.

        """

        super().__init__(
            p=p,
            augment=(
                (lambda p: np.random.rand() < p)
                if augment is None
                else augment
            ),
            **kwargs,
        )

    def _get_xp(
        self: FlipLR,
        array: np.ndarray | torch.Tensor,
        xp: Any,
        augment: bool,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Flip an array along the width axis.

        Parameters
        ----------
        array: np.ndarray | torch.Tensor
            Input array to be augmented.
        xp: module
            Backend module (`numpy` or `torch`) used for array operations.
        augment: bool
            Whether the flip should be applied.

        Returns
        -------
        np.ndarray | torch.Tensor
            Flipped array if `augment` is True, otherwise the input array.

        """

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
        """Update position metadata after a left-right flip.

        If the element contains `"position"` properties, their width
        coordinate is mirrored to match the flipped array.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            Scattered object whose array has been flipped.
        old_shape: tuple
            Shape of the array before augmentation.
        new_shape: tuple
            Shape of the array after augmentation.
        **kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        ScatteredVolume | ScatteredField
            The updated scattered object.

        """

        if hasattr(element, "properties"):
            for prop in element.properties:
                if "position" in prop:
                    pos = prop["position"]
                    W = old_shape[1]
                    new_pos = (
                        pos.clone() if hasattr(pos, "clone") else pos.copy()
                    )
                    new_pos[..., 1] = W - 1 - new_pos[..., 1]
                    prop["position"] = new_pos

        return element


class FlipUD(Augmentation):
    """Flip images up-down.

    If the input is a `ScatteredVolume` or `ScatteredField`, the underlying
    array is flipped along the height axis and any `"position"` metadata is
    updated accordingly.

    Parameters
    ----------
    p: PropertyLike[float], optional
        Probability of performing the flip. Defaults to `0.5`.
    augment: PropertyLike[bool] | None
        Boolean controlling whether the augmentation is applied. If `None`,
        the augmentation is performed with probability `p`.

    Methods
    -------
    `_get_xp(image, xp, augment, **kwargs) -> np.ndarray | torch.Tensor`
        Abstract method which performs the `FlipUD` augmentation.
    `_update_properties(...) -> ScatteredVolume | ScatteredField`
        Abstract method to update the properties of the scattered volume or
        field.

    Examples
    --------
    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(intensity=1)
    >>> optics = dt.Fluorescence()
    >>> image = optics(particle) >> dt.FlipUD(p=1.0)
    >>> image.plot();

    """

    def __init__(
        self: FlipUD,
        p: PropertyLike[float] = 0.5,
        augment: PropertyLike[bool] = None,
        **kwargs,
    ) -> None:
        """Initialize the FlipUD augmentation.

        This constructor initializes the `FlipUD` augmentation with the
        specified parameters. The `p` parameter controls the probability of
        performing the flip, while the `augment` parameter can be used to
        directly control whether the augmentation is applied. If `augment` is
        set to `None`, the augmentation will be performed with probability `p`.
        This allows for flexible control over when the flip is applied, making
        it suitable for use in data augmentation pipelines where random
        transformations are desired.

        Parameters
        ----------
        p: PropertyLike[float], optional
            Probability of performing the flip. Defaults to `0.5`.
        augment: PropertyLike[bool] | None
            Boolean controlling whether the augmentation is applied. If `None`,
            the augmentation is performed with probability `p`.
        **kwargs: Any
            Additional keyword arguments used to configure the feature. Each
            keyword argument is wrapped as a `Property` and added to the
            feature's `properties` attribute. These properties are resolved
            dynamically at call time and passed to the `.get()` method.

        """

        super().__init__(
            p=p,
            augment=(
                (lambda p: np.random.rand() < p)
                if augment is None
                else augment
            ),
            **kwargs,
        )

    def _get_xp(
        self,
        array: np.ndarray | torch.Tensor,
        xp: Any,
        augment: bool,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Flip an array along the height axis.

        Parameters
        ----------
        array: np.ndarray | torch.Tensor
            Input array to be augmented.
        xp: module
            Backend module (`numpy` or `torch`) used for array operations.
        augment: bool
            Whether the flip should be applied.

        Returns
        -------
        np.ndarray | torch.Tensor
            Flipped array if `augment` is True, otherwise the input array.

        """

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
        """Update position metadata after an up-down flip.

        If the element contains `"position"` properties, their height
        coordinate is mirrored to match the flipped array.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            Scattered object whose array has been flipped.
        old_shape: tuple
            Shape of the array before augmentation.
        new_shape: tuple
            Shape of the array after augmentation.
        **kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        ScatteredVolume | ScatteredField
            The updated scattered object.

        """

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
    """Flip images along the diagonal.

    If the input is a `ScatteredVolume` or `ScatteredField`, the underlying
    array is transposed and any `"position"` metadata is updated
    accordingly.

    Parameters
    ----------
    p: PropertyLike[float], optional
        Probability of performing the flip. Defaults to `0.5`.
    augment: PropertyLike[bool] | None
        Boolean controlling whether the augmentation is applied. If `None`,
        the augmentation is performed with probability `p`.

    Methods
    -------
    `_get_xp(image, xp, augment, **kwargs) -> np.ndarray | torch.Tensor`
        Abstract method which performs the `FlipDiagonal` augmentation.
    `_update_properties(...) -> ScatteredVolume | ScatteredField`
        Abstract method to update the properties of the scattered volume or
        field.

    Examples
    --------
    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(intensity=1)
    >>> optics = dt.Fluorescence()
    >>> image = optics(particle) >> dt.FlipDiagonal(p=1.0)
    >>> image.plot();

    """

    def __init__(
        self: FlipDiagonal,
        p: PropertyLike[float] = 0.5,
        augment: PropertyLike[bool] = None,
        **kwargs,
    ) -> None:
        """Initialize the FlipDiagonal augmentation.

        This constructor initializes the `FlipDiagonal` augmentation with the
        specified parameters. The `p` parameter controls the probability of
        performing the flip, while the `augment` parameter can be used to
        directly control whether the augmentation is applied. If `augment` is
        set to `None`, the augmentation will be performed with probability `p`.
        This allows for flexible control over when the flip is applied, making
        it suitable for use in data augmentation pipelines where random
        transformations are desired.

        Parameters
        ----------
        p: PropertyLike[float], optional
            Probability of performing the flip. Defaults to `0.5`.
        augment: PropertyLike[bool] | None
            Boolean controlling whether the augmentation is applied. If `None`,
            the augmentation is performed with probability `p`.
        **kwargs: Any
            Additional keyword arguments used to configure the feature. Each
            keyword argument is wrapped as a `Property` and added to the
            feature's `properties` attribute. These properties are resolved
            dynamically at call time and passed to the `.get()` method.

        """
        super().__init__(
            p=p,
            augment=(
                (lambda p: np.random.rand() < p)
                if augment is None
                else augment
            ),
            **kwargs,
        )

    def _get_xp(
        self,
        array: np.ndarray | torch.Tensor,
        xp: Any,
        augment: bool,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Flip an array along the diagonal.

        Parameters
        ----------
        array: np.ndarray | torch.Tensor
            Input array to be augmented.
        xp: module
            Backend module (`numpy` or `torch`) used for array operations.
        augment: bool
            Whether the flip should be applied.

        Returns
        -------
        np.ndarray | torch.Tensor
            Flipped array if `augment` is True, otherwise the input array.

        """

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
        """Update position metadata after a diagonal flip.

        If the element contains `"position"` properties, their height and
        width coordinates are swapped to match the transposed array.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            Scattered object whose array has been flipped.
        old_shape: tuple
            Shape of the array before augmentation.
        new_shape: tuple
            Shape of the array after augmentation.
        **kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        ScatteredVolume | ScatteredField
            The updated scattered object.

        """

        if hasattr(element, "properties"):
            for prop in element.properties:
                if "position" in prop:
                    pos = prop["position"]

                    new_pos = (
                        pos.clone() if hasattr(pos, "clone") else pos.copy()
                    )

                    # swap y and x
                    tmp = (
                        new_pos[..., 0].clone()
                        if hasattr(new_pos, "clone")
                        else new_pos[..., 0].copy()
                    )
                    new_pos[..., 0] = new_pos[..., 1]
                    new_pos[..., 1] = tmp

                    prop["position"] = new_pos

        return element


class Affine(Augmentation):
    """Apply affine transformations to images.

    This augmentation performs geometric transformations including:
    - translation
    - scaling
    - rotation
    - shearing

    Some transformations require interpolating between neighboring pixels
    to compute new output values. The `order` parameter controls the
    interpolation method.

    Parameters
    ----------
    scale: PropertyLike[float | tuple[float, float]]
        Scaling factor. A value of `1.0` corresponds to no scaling.
        If two values are provided, the height and width are scaled
        independently.
    translate: PropertyLike[float | tuple[float, float]] | None
        Translation in pixels along the height and width axes.
    translate_px: PropertyLike[float], optional
        Legacy alias for `translate`. Used when `translate` is not provided.
        Defaults to `0`.
    rotate: PropertyLike[float], optional
        Rotation angle in radians around the image center. Defaults to `0`.
    shear: PropertyLike[float], optional
        Shear angle in radians. Defaults to `0`.
    order: PropertyLike[int], optional
        Interpolation order used when resampling the image.
            * ``0``: ``Nearest-neighbor``
            * ``1``: ``Bi-linear`` (default)
            * ``2``: ``Bi-quadratic`` (not recommended by skimage)
            * ``3``: ``Bi-cubic``
            * ``4``: ``Bi-quartic``
            * ``5``: ``Bi-quintic``
    cval: PropertyLike[float], optional
        Constant value used to fill pixels when `mode="constant"`.
        Defaults to `0`.
    mode: PropertyLike[str], optional
        Boundary mode used when sampling outside the image domain.
        Options match `scipy.ndimage.affine_transform`.
        Defaults to `"reflect"`.

    Methods
    -------
    `get_numpy(image, **kwargs) -> np.ndarray`
        Applies the affine transformation to a NumPy array.
    `get_torch(image, **kwargs) -> torch.Tensor`
        Applies the affine transformation to a PyTorch tensor.
    `update_properties(...) -> ScatteredVolume | ScatteredField`
        Updates the properties of a `ScatteredVolume` or `ScatteredField`.

    """

    def __init__(
        self: Affine,
        scale: PropertyLike[float | tuple[float, float]] = 1,
        translate: PropertyLike[float | tuple[float, float] | None] = None,
        translate_px: PropertyLike[float] = 0.0,
        rotate: PropertyLike[float] = 0.0,
        shear: PropertyLike[float] = 0.0,
        order: PropertyLike[int] = 1,
        cval: PropertyLike[float] = 0.0,
        mode: PropertyLike[str] = "reflect",
        **kwargs,
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
    ) -> np.ndarray:
        """Apply the affine transformation to a NumPy array.

        Parameters
        ----------
        array: np.ndarray
            Input array to be augmented.
        scale: float | tuple[float, float]
            Scaling factor. A value of `1.0` corresponds to no scaling.
            If two values are provided, the height and width are scaled
            independently.
        translate: float | tuple[float, float] | None
            Translation in pixels along the height and width axes. If `None`,
            no translation is applied.
        rotate: float, optional
            Rotation angle in radians around the image center.
            Defaults to `0`.
        shear: float, optional
            Shear angle in radians. Defaults to `0`.
        order: int, optional
            Interpolation order used when resampling the image.
                * `0`: `Nearest-neighbor`
                * `1`: `Bi-linear` (default)
                * `2`: `Bi-quadratic` (not recommended by skimage)
                * `3`: `Bi-cubic`
                * `4`: `Bi-quartic`
                * `5`: `Bi-quintic`
        cval: float, optional
            Constant value used to fill pixels when `mode="constant"`.
            Defaults to `0`.
        mode: str, optional
            Boundary mode used when sampling outside the image domain. Options
            match `scipy.ndimage.affine_transform`. Supported modes include:
                * `reflect` (default)
                * `nearest`
                * `constant`
                * `wrap`

        Returns
        -------
        np.ndarray
            The augmented array after applying the affine transformation.

        Examples
        --------
        >>> import deeptrack as dt
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> particle = dt.PointParticle(intensity=1)
        >>> optics = dt.Fluorescence()
        >>> affine = dt.Affine(
        ...     scale=1.2,
        ...     translate=10,
        ...     rotate=np.pi / 6,
        ...     shear=np.pi / 12,
        ...     order=3,
        ...     cval=0,
        ...     mode="constant",
        ... )
        >>> pipeline = optics(particle) >> affine
        >>> image = pipeline.new()
        >>> plt.imshow(image, cmap="gray");

        """

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
        center = (np.array(shape[:2], dtype=float) - 1) / 2
        offset = center - matrix @ center - np.array([dy, dx], dtype=float)

        forward = np.linalg.inv(matrix)
        forward_offset = -forward @ offset

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
    ) -> torch.Tensor:
        """Apply the affine transformation to a PyTorch tensor.

        Parameters
        ----------
        array: torch.Tensor
            Input tensor to be augmented.
        scale: float | tuple[float, float]
            Scaling factor. A value of `1.0` corresponds to no scaling.
            If two values are provided, the height and width are scaled
            independently.
        translate: float | tuple[float, float] | None
            Translation in pixels along the height and width axes. If `None`,
            no translation is applied.
        rotate: float, optional
            Rotation angle in radians around the image center. Defaults to `0`.
        shear: float, optional
            Shear angle in radians. Defaults to `0`.
        order: int, optional
            Interpolation order used when resampling the image.
                * `0`: `Nearest-neighbor`
                * `1`: `Bi-linear` (default)
                * `2`: `Bi-quadratic` (not recommended by skimage)
                * `3`: `Bi-cubic`
                * `4`: `Bi-quartic`
                * `5`: `Bi-quintic`
        cval: float, optional
            Constant value used to fill pixels when `mode="constant"`. Note
            that PyTorch's `grid_sample` does not support a constant fill mode,
            so this parameter is ignored in the PyTorch implementation.
            Defaults to `0`.
        mode: str, optional
            Boundary mode used when sampling outside the image domain. Options
            match `scipy.ndimage.affine_transform`. Supported modes include:
                * `reflect` (default)
                * `nearest`
                * `constant` (treated as `zeros` in the PyTorch implementation)
                * `wrap` (only supported in the PyTorch implementation)
                If `mode="wrap"` is used in the PyTorch implementation, it will
                be treated as `mode="wrap"` to enable wrapping behavior. In the
                NumPy implementation, `mode="wrap"` is treated as
                `mode="constant"` with `cval=0` to avoid issues with negative
                indices in `scipy.ndimage.affine_transform`.

        Returns
        -------
        torch.Tensor
            The augmented tensor after applying the affine transformation.

        """

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

        scale_map = torch.tensor(
            [[1 / fy, 0], [0, 1 / fx]],
            dtype=dtype,
            device=device,
        )

        rotation_map = torch.stack(
            [torch.stack([cr, sr]), torch.stack([-sr, cr])]
        )

        shear_map = torch.tensor([[1, 0], [-k, 1]], dtype=dtype, device=device)

        matrix = scale_map @ rotation_map @ shear_map

        # --- IMPORTANT: use (H-1)/2 center ---
        H, W = array.shape[:2]
        center = torch.tensor(
            [(H - 1) / 2, (W - 1) / 2],
            dtype=dtype,
            device=device,
        )

        offset = (
            center
            - matrix @ center
            - torch.tensor([dy, dx], dtype=dtype, device=device)
        )

        # Store for metadata update
        forward = torch.linalg.inv(matrix)
        forward_offset = -forward @ offset

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
    ) -> ScatteredVolume | ScatteredField:
        """Update geometric metadata after an affine transform.

        Positions and direction vectors stored in the element metadata are
        transformed using the inverse affine mapping applied to the image.
        This ensures that geometric annotations remain consistent with the
        warped image.
        Metadata is always processed as NumPy arrays, independent of the
        backend used for image resampling.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            The element whose properties are to be updated.
        old_shape: tuple
            The shape of the image before augmentation.
        new_shape: tuple
            The shape of the image after augmentation.
        **kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        ScatteredVolume | ScatteredField
            The element with updated properties reflecting the affine
            transformation.

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

            transformed = (forward @ coords[..., None]).squeeze(
                -1
            ) + forward_offset

            pos[..., :2] = transformed
            props["position"] = pos

        # Update direction vectors
        if "direction" in props:

            direction = np.asarray(props["direction"], dtype=float).copy()

            coords = direction[..., :2]

            transformed_dir = (forward @ coords[..., None]).squeeze(-1)

            direction[..., :2] = transformed_dir
            props["direction"] = direction

        return element


class ElasticTransformation(Augmentation):
    """Apply elastic distortions to images.

    This augmentation generates a random displacement field that locally
    warps the input image. The displacement field is created by sampling
    random noise and smoothing it with a Gaussian kernel.
    The parameters `alpha` and `sigma` control the strength and smoothness
    of the distortion field respectively.

    Parameters
    ----------
    alpha: PropertyLike[float], optional
        Strength of the displacement field. Defaults to `20`.
    sigma: PropertyLike[float], optional
        Standard deviation of the Gaussian kernel used to smooth the
        displacement field. Defaults to `2`.
    ignore_last_dim: PropertyLike[bool], optional
        If `True`(default), the last dimension is assumed to represent channels
        and the same displacement field is applied to all channels.
    order: PropertyLike[int], optional
        Interpolation order used when resampling the image.
            * 0: Nearest-neighbor
            * 1: Bi-linear
            * 2: Bi-quadratic
            * 3: Bi-cubic (default)
            * 4: Bi-quartic
            * 5: Bi-quintic
    cval: PropertyLike[float], optional
        Constant value used when `mode="constant"`. Defaults to `0`.
    mode: PropertyLike[str], optional
        Boundary mode used when sampling outside the image domain.
        Matches `scipy.ndimage.map_coordinates`. Defaults to `"constant"`.

    Methods
    -------
    `get_numpy(image, **kwargs) -> np.ndarray`
        Applies the elastic transformation to a NumPy array.
    `get_torch(image, **kwargs) -> torch.Tensor`
        Applies the elastic transformation to a PyTorch tensor.

    Notes
    -----
    This augmentation does not update `"position"` metadata. It should not
    be used if labels depend on spatial coordinates derived from the image.

    Examples
    --------
    >>> import deeptrack as dt

    >>> particle = dt.Ellipse()
    >>> optics = dt.Fluorescence()
    >>> elastic = dt.ElasticTransformation(alpha=30, sigma=3)
    >>> image = optics(particle) >> elastic
    >>> image.plot();

    """

    def __init__(
        self: ElasticTransformation,
        alpha: PropertyLike[float] = 20,
        sigma: PropertyLike[float] = 2,
        ignore_last_dim: PropertyLike[bool] = True,
        order: PropertyLike[int] = 3,
        cval: PropertyLike[float] = 0,
        mode: PropertyLike[str] = "constant",
        **kwargs,
    ) -> None:
        """Initialize the elastic transformation.

        The parameters control the strength (`alpha`) and smoothness (`sigma`)
        of the displacement field, as well as interpolation and boundary
        behavior during resampling.

        Parameters
        ----------
        alpha: PropertyLike[float], optional
            Strength of the displacement field. Defaults to `20`.
        sigma: PropertyLike[float], optional
            Standard deviation of the Gaussian kernel used to smooth the
            displacement field. Defaults to `2`.
        ignore_last_dim: PropertyLike[bool], optional
            If `True` (optional), the last dimension is assumed to represent
            channels and the same displacement field is applied to all
            channels.
        order: PropertyLike[int], optional
            Interpolation order used when resampling the image.
                * 0: Nearest-neighbor
                * 1: Bi-linear
                * 2: Bi-quadratic
                * 3: Bi-cubic (default)
                * 4: Bi-quartic
                * 5: Bi-quintic
        cval: PropertyLike[float], optional
            Constant value used when `mode="constant"`. Defaults to `0`.
        mode: PropertyLike[str], optional
            Boundary mode used when sampling outside the image domain. Matches
            `scipy.ndimage.map_coordinates`. Supported modes include:
                * `reflect`
                * `nearest`
                * `constant` (default)
                * `wrap`

        """

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
        **kwargs,
    ) -> np.ndarray:
        """Apply elastic distortion to a NumPy array.

        A random displacement field is generated using Gaussian-smoothed
        noise and applied to the image using `scipy.ndimage.map_coordinates`.

        Parameters
        ----------
        image: np.ndarray
            Input image to transform.
        sigma: float
            Standard deviation of the Gaussian smoothing kernel.
        alpha: float
            Strength of the displacement field.
        ignore_last_dim: bool
            If True, the last dimension is treated as channels and the same
            displacement field is applied to all channels.

        Returns
        -------
        np.ndarray
            Distorted image.

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
            dDim = (
                np.transpose(grid, axes=(1, 0) + tuple(range(2, grid.ndim)))
                + delta
            )
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
        """Apply elastic distortion to a PyTorch tensor.

        A random displacement field is generated and smoothed using a
        Gaussian kernel implemented with convolution. The resulting field
        is applied using `torch.nn.functional.grid_sample`.

        Parameters
        ----------
        image: torch.Tensor
            Input tensor with shape `(H, W)` or `(H, W, C)`.
        sigma: float
            Standard deviation of the Gaussian smoothing kernel.
        alpha: float
            Strength of the displacement field.
        ignore_last_dim: bool
            If True, the same displacement field is applied to all channels.

        Returns
        -------
        torch.Tensor
            Distorted tensor with the same shape as the input.

        """

        if image.ndim not in (2, 3):
            raise ValueError(
                "ElasticTransformation only supports 2D or 3D tensors."
            )

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
            coords = torch.arange(
                -radius, radius + 1, device=device, dtype=dtype
            )
            kernel = torch.exp(-(coords**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            return kernel

        kernel = gaussian_kernel_1d(sigma)
        kernel_x = kernel.view(1, 1, 1, -1)
        kernel_y = kernel.view(1, 1, -1, 1)

        def smooth(field):
            field = field.unsqueeze(0).unsqueeze(0)
            field = F.conv2d(
                field, kernel_x, padding=(0, kernel_x.shape[-1] // 2)
            )
            field = F.conv2d(
                field, kernel_y, padding=(kernel_y.shape[-2] // 2, 0)
            )
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
                image_[:, c : c + 1],
                grid[c : c + 1],
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
    """Crop a region of an image.

    The cropped region can be specified either by defining the number of
    pixels to remove from the borders or by specifying the size of the
    output image.

    Parameters
    ----------
    crop: int | tuple[int, ...] | list[int] | Callable
        Defines the cropping amount. If an integer, the same value is used
        for all axes. If a tuple or list, values are interpreted per axis.
        If `crop_mode="retain"`, `crop` specifies the output size.
        If `crop_mode="remove"`, `crop` specifies the number of pixels
        removed from the borders.
        A callable may also be provided, which receives the input array and
        returns any of the above formats.
    crop_mode: PropertyLike[str], optional
        How the `crop` parameter is interpreted.
        - `"retain"`: `crop` specifies the output size. (default)
        - `"remove"`: `crop` specifies the number of pixels removed.
    corner: PropertyLike[str | tuple[int, int] | Callable], optional
        Top-left corner of the cropped region.
        - `"random"` selects a random valid corner. (default)
        - A tuple specifies the corner explicitly.
        - A callable receives the input array and returns a corner.

    Methods
    -------
    `_get_xp(...) -> np.ndarray | torch.Tensor`
        Internal method that performs cropping on either a NumPy array or a
        PyTorch tensor based on the specified parameters.

    Examples
    --------
    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(position=(32, 32))
    >>> optics = dt.Fluorescence()
    >>> crop = dt.Crop(crop=64, crop_mode="retain", corner=(0,0))
    >>> image = optics(particle) >> crop
    >>> image.plot();

    """

    def __init__(
        self: Crop,
        *args,
        crop: (
            int
            | list[int]
            | tuple[int]
            | Callable[[np.ndarray | torch.Tensor], tuple[int, ...]]
        ),
        crop_mode: PropertyLike[str] = "retain",
        corner: PropertyLike[str] = "random",
        **kwargs,
    ) -> None:
        """Initialize the cropping augmentation.

        The crop size and placement can be fixed, random, or computed
        dynamically from the input image.

        Parameters
        ----------
        crop: int | tuple[int, ...] | list[int] | Callable
            Defines the cropping amount. If an integer, the same value is used
            for all axes. If a tuple or list, values are interpreted per axis.
            If `crop_mode="retain"`, `crop` specifies the output size.
            If `crop_mode="remove"`, `crop` specifies the number of pixels
            removed from the borders.
            A callable may also be provided, which receives the input array and
            returns any of the above formats.
        crop_mode: PropertyLike[str], optional
            How the `crop` parameter is interpreted.
            - `"retain"`: `crop` specifies the output size. (default)
            - `"remove"`: `crop` specifies the number of pixels removed.
        corner: PropertyLike[str | tuple[int, int] | Callable], optional
            Top-left corner of the cropped region.
            - `"random"` selects a random valid corner. (default)
            - A tuple specifies the corner explicitly.
            - A callable receives the input array and returns a corner.

        """

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
        crop: (
            int
            | list[int]
            | tuple[int]
            | Callable[[np.ndarray | torch.Tensor], tuple[int, ...]]
        ),
        crop_mode: str,
        corner: (
            str
            | tuple[int]
            | Callable[[np.ndarray | torch.Tensor], tuple[int, ...]]
        ),
        xp: Any,
        **kwargs,
    ) -> np.ndarray | torch.Tensor:
        """Crop an array using the specified crop parameters.

        The cropping region is determined by `crop`, `crop_mode`, and `corner`.
        The same logic is used for both NumPy arrays and PyTorch tensors.
        The appropriate backend is selected automatically.

        Parameters
        ----------
        array: np.ndarray | torch.Tensor
            Input array to be cropped.
        crop: int | tuple[int, ...] | list[int] | Callable
            Defines the cropping amount. If an integer, the same value is used
            for all axes. If a tuple or list, values are interpreted per axis.
            If `crop_mode="retain"`, `crop` specifies the output size.
            If `crop_mode="remove"`, `crop` specifies the number of pixels
            removed from the borders.
            A callable may also be provided, which receives the input array and
            returns any of the above formats.
        crop_mode: str
            How the `crop` parameter is interpreted.
            - `"retain"`: `crop` specifies the output size.
            - `"remove"`: `crop` specifies the number of pixels removed.
        corner: str | tuple[int] | Callable
            Top-left corner of the cropped region.
            - `"random"` selects a random valid corner.
            - A tuple specifies the corner explicitly.
            - A callable receives the input array and returns a corner.
        xp: module
            The array library (e.g., `numpy` or `torch`) to use for
            computations.

        Returns
        -------
        np.ndarray | torch.Tensor
            Cropped array.

        """

        if callable(crop):
            crop = crop(array)

        if isinstance(crop, int):
            crop = (crop,) * array.ndim

        crop = [
            c if c is not None else array.shape[i] for i, c in enumerate(crop)
        ]

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

        slice_start = [
            int(c) % (int(m) + 1) for c, m in zip(slice_start, crop_amount)
        ]

        slice_end = [
            a - c + s for a, s, c in zip(array.shape, slice_start, crop_amount)
        ]

        slices = tuple(slice(s0, s1) for s0, s1 in zip(slice_start, slice_end))

        out = array[slices]

        # Store for metadata update
        self._last_crop = {
            "start": tuple(slice_start),
        }

        return out

    def _update_properties(
        self: Crop,
        element: ScatteredVolume | ScatteredField,
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> ScatteredVolume | ScatteredField:
        """Update metadata after cropping.

        Adjusts `"position"` coordinates to match the cropped image and
        updates `"output_region"` to reflect the new image bounds.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            The element whose properties are to be updated.
        old_shape: tuple[int, ...]
            The shape of the image before cropping.
        new_shape: tuple[int, ...]
            The shape of the image after cropping.
        **kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        ScatteredVolume | ScatteredField
            The element with updated properties reflecting the cropping.

        """

        if not hasattr(self, "_last_crop"):
            return element

        if not isinstance(getattr(element, "properties", None), dict):
            return element

        props = element.properties

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
    """Crop images so their dimensions are multiples of a given value.

    The image is cropped along each axis until its size becomes a multiple
    of the specified value.

    Parameters
    ----------
    multiple: PropertyLike[int | tuple[int | None, ...]]
        Target multiples for each axis. If a single integer is provided,
        the same multiple is applied to all axes.
        If a tuple is provided, each value corresponds to an axis.
        A value of `None` or `-1` indicates that the axis should not be
        constrained.

    corner: PropertyLike[str], optional
        Top-left corner of the cropped region.
        - `"random"` selects a random valid corner. (default)
        - A tuple specifies the corner explicitly.
        - A callable receives the input array and returns a corner.

    Examples
    --------
    >>> import deeptrack as dt

    >>> particle = dt.PointParticle(position=(32, 32))
    >>> optics = dt.Fluorescence()
    >>> crop_mult = dt.CropToMultiplesOf(multiple=5, corner=(0,0))
    >>> image = optics(particle) >> crop_mult
    >>> print(image.resolve().shape)

    """

    def __init__(
        self: CropToMultiplesOf,
        multiple: PropertyLike[int | tuple[int | None, ...]] = 1,
        corner: PropertyLike[str] = "random",
        **kwargs,
    ) -> None:
        """Initialize the CropToMultiplesOf augmentation.

        The image is cropped so that each dimension becomes a multiple of the
        specified value. Cropping is performed by reducing the image size
        along each axis while preserving the selected corner.

        Parameters
        ----------
        multiple: PropertyLike[int | tuple[int | None, ...]], optional
            Target multiple for each axis.
            - If a single integer is provided, the same multiple is applied
              to all axes.
            - If a tuple is provided, values correspond to individual axes.
            - A value of `None` or `-1` skips cropping for that axis.
            Defaults to `1`.

        corner: PropertyLike[str | tuple[int, ...] | Callable], optional
            Top-left corner of the cropped region.
            - `"random"` selects a random valid corner. (default)
            - A tuple specifies the corner explicitly.
            - A callable receives the input array and returns a corner.

        **kwargs: Any
            Additional keyword arguments passed to the parent `Crop`
            augmentation.

        """

        kwargs.pop("crop", None)
        kwargs.pop("crop_mode", None)

        def image_to_crop(image):
            """Determine the crop size.

            Determine the crop size based on the input image and target
            multiples.

            """

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
    """Crop an array to remove empty space.

    Removes leading and trailing indices along each axis where all values
    are below a threshold `eps`. This effectively crops the array to the
    smallest bounding box containing values larger than `eps`.
    Currently only supports 3D arrays of shape (H, W, Z).

    Parameters
    ----------
    eps: PropertyLike[float], optional
        Threshold below which values are considered empty. Defaults to `1e-10`.

    Methods
    -------
    `_get_numpy(image, **kwargs) -> np.ndarray`
        Applies tight cropping to a NumPy array.
    `_get_torch(image, **kwargs) -> torch.Tensor`
        Applies tight cropping to a PyTorch tensor.

    Examples
    --------
    >>> import deeptrack as dt
    >>> particle = dt.PointParticle(position=(32, 32))
    >>> optics = dt.Fluorescence()
    >>> crop_tight = dt.CropTight(eps=1e-5)
    >>> image = optics(particle) >> crop_tight
    >>> image.plot()

    """

    def __init__(
        self: CropTight,
        eps: PropertyLike[float] = 1e-10,
        **kwargs: Any,
    ) -> None:
        """Initialize the tight cropping augmentation.

        Parameters
        ----------
        eps: PropertyLike[float], optional
            Threshold below which values are considered empty.
            Defaults to `1e-10`.

        """

        super().__init__(eps=eps, **kwargs)

    def _get_numpy(
        self: CropTight,
        image: np.ndarray,
        eps: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Crop a NumPy array to its non-empty bounding box.

        Pixels with values below `eps` are treated as empty.

        Parameters
        ----------
        image: np.ndarray
            Input array to be cropped. Should be 3D.
        eps: float
            Threshold below which values are considered empty.
        kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        np.ndarray
            Cropped array containing all values above the threshold.

        """

        mask = image > eps

        keep_z = np.any(mask, axis=(0, 1))
        keep_y = np.any(mask, axis=(1, 2))
        keep_x = np.any(mask, axis=(0, 2))

        ys = np.where(keep_y)[0]
        xs = np.where(keep_x)[0]
        zs = np.where(keep_z)[0]

        if len(ys) == 0 or len(xs) == 0 or len(zs) == 0:
            # nothing survives — return minimal array
            self._last_crop = {
                "ymin": 0,
                "xmin": 0,
                "ymax": 0,
                "xmax": 0,
                "zmin": 0,
                "zmax": 0,
            }
            return image[0:1, 0:1, 0:1]

        ymin = ys[0]
        ymax = ys[-1] + 1

        xmin = xs[0]
        xmax = xs[-1] + 1

        zmin = zs[0]
        zmax = zs[-1] + 1

        self._last_crop = {
            "ymin": ymin,
            "xmin": xmin,
            "ymax": ymax,
            "xmax": xmax,
            "zmin": zmin,
            "zmax": zmax,
        }

        return image[ymin:ymax, xmin:xmax, zmin:zmax]

    def _get_torch(
        self: CropTight,
        image: torch.Tensor,
        eps: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Crop a PyTorch tensor to its non-empty bounding box.

        Pixels with values below `eps` are treated as empty.

        Parameters
        ----------
        image: torch.Tensor
            Input tensor to be cropped. Should be 3D.
        eps: float
            Threshold below which values are considered empty.
        kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        torch.Tensor
            Cropped tensor containing all values above the threshold.

        """

        mask = image > eps

        keep_z = torch.any(mask, dim=(0, 1))
        keep_y = torch.any(mask, dim=(1, 2))
        keep_x = torch.any(mask, dim=(0, 2))

        ys = torch.nonzero(keep_y, as_tuple=True)[0]
        xs = torch.nonzero(keep_x, as_tuple=True)[0]
        zs = torch.nonzero(keep_z, as_tuple=True)[0]

        if len(ys) == 0 or len(xs) == 0 or len(zs) == 0:
            self._last_crop = {
                "ymin": 0,
                "xmin": 0,
                "ymax": 0,
                "xmax": 0,
                "zmin": 0,
                "zmax": 0,
            }
            return image[0:1, 0:1, 0:1]

        ymin = int(ys[0])
        ymax = int(ys[-1]) + 1

        xmin = int(xs[0])
        xmax = int(xs[-1]) + 1

        zmin = int(zs[0])
        zmax = int(zs[-1]) + 1

        self._last_crop = {
            "ymin": ymin,
            "xmin": xmin,
            "ymax": ymax,
            "xmax": xmax,
            "zmin": zmin,
            "zmax": zmax,
        }

        return image[ymin:ymax, xmin:xmax, zmin:zmax]

    def _update_properties(
        self: CropTight,
        element: ScatteredVolume | ScatteredField,
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> ScatteredVolume | ScatteredField:
        """Update metadata after tight cropping.

        Adjusts `"position"` coordinates to remain consistent with the cropped
        image and updates `"output_region"` to reflect the new image bounds.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            The element whose properties are to be updated.
        old_shape: tuple[int, ...]
            The shape of the image before cropping.
        new_shape: tuple[int, ...]
            The shape of the image after cropping.
        kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        ScatteredVolume | ScatteredField
             The element with updated properties reflecting the tight cropping.

        """

        if not hasattr(self, "_last_crop"):
            return element

        if not isinstance(getattr(element, "properties", None), dict):
            return element

        if "position" in element.properties:
            pos = np.asarray(
                element.properties["position"], dtype=float
            ).copy()
            pos[..., 0] -= self._last_crop["ymin"]
            pos[..., 1] -= self._last_crop["xmin"]
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
    """Pad an image by adding extra pixels along specified axes.

    This augmentation adds padding to an image using functionality similar to
    `numpy.pad`. The padding is specified using a flat sequence `px` describing
    the number of pixels added before and after each axis.

    Parameters
    ----------
    px: PropertyLike[int | tuple[int, ...] | list[int]]
        Amount of padding for each axis, specified as a flat sequence
            (before_axis0, after_axis0, before_axis1, after_axis1, ...)
        If a single integer is provided, the same padding is applied before and
        after every axis.
    mode: PropertyLike[str], optional
        Padding mode used when extending the array. Supported modes follow
        `numpy.pad` and `torch.nn.functional.pad`. Defaults to `"constant"`.
    cval: PropertyLike[float], optional
        Constant value used when `mode="constant"`. Defaults to `0`.

    Methods
    -------
    `_get_numpy(image, **kwargs) -> np.ndarray`
        Apply padding to a NumPy array.

    `_get_torch(image, **kwargs) -> torch.Tensor`
        Apply padding to a PyTorch tensor.

    Returns
    -------
    np.ndarray or torch.Tensor
        The padded image.

    Examples
    --------
    >>> import deeptrack as dt
    >>> particle = dt.PointParticle(position=(32, 32))
    >>> optics = dt.Fluorescence()
    >>> pad = dt.Pad(px=(10, 10, 5, 5), mode="constant", cval=0)
    >>> image = optics(particle) >> pad
    >>> print(image.resolve().shape)

    """

    def __init__(
        self: Pad,
        px: PropertyLike[int | tuple[int, ...] | list[int]] = (0, 0, 0, 0),
        mode: PropertyLike[str] = "constant",
        cval: PropertyLike[float] = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the padding augmentation.

        Parameters
        ----------
        px: PropertyLike[int | tuple[int, ...] | list[int]], optional
            Amount of padding for each axis specified as
                (before_axis0, after_axis0, before_axis1, after_axis1, ...)
            Defaults to `(0, 0, 0, 0)`.
        mode: PropertyLike[str], optional
            Padding mode used when extending the array.
            Defaults to `"constant"`.
        cval: PropertyLike[float], optional
            Constant value used when `mode="constant"`. Defaults to `0`.
        **kwargs: Any
            Additional keyword arguments passed to the parent augmentation.

        """

        super().__init__(px=px, mode=mode, cval=cval, **kwargs)

    def _get_numpy(
        self: Pad,
        image: np.ndarray,
        px: PropertyLike[int | tuple[int, ...] | list[int]],
        mode: str = "constant",
        cval: float = 0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply padding to a NumPy array.

        Parameters
        ----------
        image: np.ndarray
            Input array to be padded.
        px: list[int] | tuple[int]
            Amount of padding specified as
                (before_axis0, after_axis0, before_axis1, after_axis1, ...)
        mode: str, optional
            Padding mode passed to `numpy.pad`. Defaults to `"constant"`.
        cval: float, optional
            Constant value used when `mode="constant"`. Defaults to `0`.
        kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        np.ndarray
            The padded array.

        """

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
        px: PropertyLike[int | tuple[int, ...] | list[int]],
        mode: str = "constant",
        cval: float = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply padding to a PyTorch tensor.

        Parameters
        ----------
        image: torch.Tensor
            Input tensor to be padded.
        px: list[int] | tuple[int]
            Amount of padding specified as
                (before_axis0, after_axis0, before_axis1, after_axis1, ...)
        mode: str, optional
            Padding mode passed to `torch.nn.functional.pad`.
            Defaults to `"constant"`.
        cval: float, optional
            Constant value used when `mode="constant"`. Defaults to `0`.
        kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        torch.Tensor
            The padded tensor.
        """

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
            mode=mode,
            value=cval if mode == "constant" else None,
        )

    def _update_properties(
        self: Pad,
        element: ScatteredVolume | ScatteredField,
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> ScatteredVolume | ScatteredField:
        """Update metadata after padding.

        Padding shifts spatial coordinates and expands the global output
        region. The `"position"` property is translated by the amount of
        padding added before each spatial axis. The `"output_region"` property
        is updated so that the padded image remains correctly aligned in the
        global coordinate system.

        Parameters
        ----------
        element: ScatteredVolume | ScatteredField
            The element whose properties are to be updated.
        old_shape: tuple[int, ...]
            Shape of the image before padding.
        new_shape: tuple[int, ...]
            Shape of the image after padding.
        **kwargs: Any
            Additional keyword arguments passed by the augmentation pipeline.

        Returns
        -------
        ScatteredVolume | ScatteredField
            The element with updated properties reflecting the applied padding.

        """

        if not hasattr(self, "_last_padding"):
            return element

        if not isinstance(getattr(element, "properties", None), dict):
            return element

        padding = self._last_padding

        props = element.properties

        # Shift position
        if "position" in props:
            pos = np.asarray(props["position"], dtype=float).copy()

            # Only shift first two dims (y, x)
            pos[..., 0] += padding[0][0]
            pos[..., 1] += padding[1][0]

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
    """Pad images so their dimensions become multiples of a given value.

    Padding is applied symmetrically along each axis so that the final image
    size is divisible by the specified multiple.

    Parameters
    ----------
    multiple: PropertyLike[int | tuple[int | None, ...]], optional
        Target multiple for each axis.
        - If a single integer is provided, the same multiple is applied to all
        axes.
        - If a tuple is provided, values correspond to individual axes.
        - A value of `None` or `-1` skips padding for that axis.
        Defaults to `1`.

    """

    def __init__(
        self: PadToMultiplesOf,
        multiple: PropertyLike[int | tuple[int | None, ...]] = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the PadToMultiplesOf augmentation.

        The image is padded symmetrically along each axis so that its final
        dimensions become multiples of the specified value.

        Parameters
        ----------
        multiple: PropertyLike[int | tuple[int | None, ...]], optional
            Target multiple for each axis.
            - If a single integer is provided, the same multiple is applied
            to all axes.
            - If a tuple is provided, values correspond to individual axes.
            - A value of `None` or `-1` skips padding for that axis.
            Defaults to `1`.

        **kwargs: Any
            Additional keyword arguments passed to the parent `Pad`
            augmentation (e.g. `mode`, `cval`).

        """

        def amount_to_pad(image: np.ndarray | torch.Tensor) -> list[int]:
            """Calculate the amount of padding.

            Calculate the amount of padding needed to make each dimension a
            multiple of the specified value.

            """

            shape = image.shape
            multiple_value = multiple  # self.multiple()

            if not isinstance(multiple_value, (list, tuple, np.ndarray)):
                multiple_value = (multiple_value,) * image.ndim

            if len(multiple_value) < image.ndim:
                multiple_value = tuple(multiple_value) + (None,) * (
                    image.ndim - len(multiple_value)
                )

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
