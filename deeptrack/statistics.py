""" Contains features that perform some statistics operation on the input.

These features reduce some dimension of the input by applying a statistical 
operation (sum, mean, etc.). They follow the syntax of the equivalent numpy
function, meaning that 'axis' and 'keepdims' are valid arguments. Moreover,
they all accept the `distributed` keyword, which determines if each image in
the input list should be handled individually or not.

Key Features:
-------------
- **Statistical Reducers**
    
    Reduce some dimension of the input by applying a statistical operation.

Module Structure
----------------

Helper functions:

- `_as_float_if_needed`: Convert integer/bool arrays to float for reducers that
    require floats.


Classes:

- `Reducer`: Base class for features that reduce input dimensionality using a
    statistical function.
- `Sum`: Computes the sum along the specified axis.
- `Prod`: Computes the product along the specified axis.
- `Mean`: Computes the arithmetic mean along the specified axis.
- `Median`: Computes the median along the specified axis.
- `Std`: Computes the standard deviation along the specified axis.
- `Variance`: Computes the variance along the specified axis.
- `Cumsum`: Computes the cumulative sum along the specified axis.
- `Min`: Computes the minimum value along the specified axis.
- `Max`: Computes the maximum value along the specified axis.
- `PeakToPeak`: Computes the range (max - min) along the specified axis.
- `Quantile`: Computes the q-th quantile along the specified axis.
- `Percentile`: Computes the q-th percentile along the specified axis.

Examples
--------
Reduce input dimensions using the `Sum` operation, with 'distributed' set
to True:

>>> import numpy as np
>>> from deeptrack import statistics
>>> input_values = [np.ones((2,)), np.zeros((2,))]
>>> sum_operation = statistics.Sum(axis=0, distributed=True)
>>> sum_result = sum_operation(input_values)
>>> print(sum_result)  # Output: [2, 0]

Reduce input dimensions using the `Sum` operation, with 'distributed' set
to False:

>>> sum_operation = statistics.Sum(axis=0, distributed=False)
>>> sum_result = sum_operation(input_values)
>>> print(sum_result)  # Output: [1, 1]

Reduce input dimensions using the `Mean` operation:

>>> mean_operation = statistics.Mean(axis=0, distributed=True)
>>> mean_result = mean_operation(input_values)
>>> print(mean_result)  # Output: [1, 0]

Reducers can be added to the pipeline in two ways:

>>> summed_pipeline = some_pipeline_of_features >> Sum(axis=0)
>>> summed_pipeline = Sum(some_pipeline_of_features, axis=0)

Combining the two ways is not supported, and the behaviour is not guaranteed.
For example:

>>> incorrectly_summed_pipline = some_feature >> Sum(
>>>     some_pipeline_of_features, axis=0
>>> )

However, other operators can be used in this way:

>>> correctly_summed_and_subtracted_pipline = some_feature - Sum(
>>>  some_pipeline_of_features, axis=0
>>> )

"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from deeptrack.features import Feature
from deeptrack.backend import xp, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

__all__ = [
    "Reducer",
    "Sum",
    "Prod",
    "Mean",
    "Median",
    "Std",
    "Variance",
    "Cumsum",
    "Min",
    "Max",
    "PeakToPeak",
    "Quantile",
    "Percentile",
]

if TYPE_CHECKING:
    import torch


def _as_float_if_needed(
        image: np.ndarray | torch.Tensor | list | tuple
    ) -> np.ndarray | torch.Tensor | list | tuple:
    """Convert integer/bool arrays to float for reducers that require floats.

    Some reducers (e.g., mean, std) require floating-point inputs to avoid
    issues with integer division or overflow. This function checks if the input
    array is of an integer or boolean type and converts it to float if 
    necessary.

    Parameters
    ----------
    image: array-like
        The input image or array to check and convert.

    Returns
    -------
    array-like
        The input image converted to float if it was of integer or boolean 
        type, otherwise the original image is returned.

    """

    if not hasattr(image, "dtype"):
        return image

    if xp.isdtype(image.dtype, "real floating"):
        return image

    if xp.isdtype(image.dtype, "complex floating"):
        return image

    return xp.astype(image, xp.float32)


class Reducer(Feature):
    """Base class that reduce input dimensionality with a statistical function.

    Parameters
    ----------
    function: Callable
        The function used to reduce the input.
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    axis: int or tuple of int
        The axis / axes to reduce over.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    **kwargs
        Additional keyword arguments passed to the parent class and the 
        function.

    Notes
    -----
    - The `distributed` keyword is passed to the parent class to determine
        how to handle the input list of images. If `distributed` is True, the
        reducer will be applied to each image in the list individually. If 
        False, the reducer will be applied to the entire list as a single 
        array.

    """

    def __init__(
        self: Reducer,
        function: Callable,
        feature: Feature | None = None,
        distributed: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Reducer feature.
        
        Parameters
        ----------
        function: Callable
            The function used to reduce the input.
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs
            Additional keyword arguments passed to the parent class and the
            function.
            
        """

        self.function = function

        if feature is not None:
            super().__init__(_input=feature, distributed=distributed, **kwargs)
        else:
            super().__init__(distributed=distributed, **kwargs)

    def _process_and_get(
        self: Reducer,
        image_list: list[np.ndarray | torch.Tensor],
        **feature_input: Any,
    ) -> list[np.ndarray | torch.Tensor]:
        """Process the input list of images and apply the reduction function.
        
        Parameters
        ----------
        image_list: list of array-like
            The list of images to process and reduce.
        **feature_input: dict
            Additional keyword arguments passed to the parent class and the
            function.
            
        Returns
        -------
        list of array-like
            The list of reduced images after applying the reduction function.

        """
        
        self.__distributed__ = feature_input["distributed"]
        return super()._process_and_get(image_list, **feature_input)

    def _as_backend_array(
        self: Reducer,
        image: np.ndarray | torch.Tensor | list | tuple,
    ) -> np.ndarray | torch.Tensor | list | tuple:
        """Convert the input image to a backend array if it is a list or tuple.

        This function checks if the input image is a list or tuple of arrays 
        and attempts to stack them into a single backend array. If stacking 
        fails (e.g., due to incompatible shapes), it falls back to converting 
        the list/tuple to a numpy array and then to the backend array. If the 
        input is a scalar, it converts it to a backend array. If the input is 
        already a backend array, it is returned as is.

        Parameters
        ----------
        image: array-like or list/tuple of array-like
            The input image or list/tuple of images to convert.

        Returns
        -------
        array-like or list/tuple of array-like
            The input image converted to a backend array if it was a 
            list/tuple, otherwise the original image is returned.
        
        """
        
        if isinstance(image, (list, tuple)):
            try:
                return xp.stack(image, axis=0)
            except (TypeError, ValueError):
                return xp.asarray(np.asarray(image))
            
        if np.isscalar(image):
            return xp.asarray(image)

        return image

    def get(
        self: Reducer,
        image: np.ndarray | torch.Tensor | list | tuple,
        axis: int | None,
        keepdims: bool | None = None,
        **kwargs: Any,
    ):
        """Apply the reduction function to the input image.

        Parameters
        ----------
        image: array-like or list/tuple of array-like
            The input image or list/tuple of images to reduce.
        axis: int or None
            The axis or axes along which the reduction is performed. If None,
            the reduction is performed over all axes.
        keepdims: bool or None
            Whether to keep the singleton dimensions after reducing or 
            squeezing them. If None, the default behavior of the reduction 
            function is used.
        **kwargs
            Additional keyword arguments passed to the parent class and the
            reduction function.

        Returns
        -------
        array-like
            The reduced image after applying the reduction function.

        """

        image = self._as_backend_array(image)
        
        if keepdims is None:
            return self.function(image, axis=axis)
        else:
            return self.function(image, axis=axis, keepdims=keepdims)


class Sum(Reducer):
    """Compute the sum along the specified axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the sum is performed. If None, the sum is
        performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the sum
        function.
    
    """

    def __init__(
        self: Sum,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Sum feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        axis: int or tuple of int or None
            The axis or axes along which the sum is performed. If None, the sum
            is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or 
            squeezing them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the sum
            function.
                
        """
        
        super().__init__(
            xp.sum,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Prod(Reducer):
    """Compute the product along the specified axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the product is performed. If None, the
        product is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the product
        function.
        
    """

    def __init__(
        self: Prod,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Prod feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        axis: int or tuple of int or None
            The axis or axes along which the product is performed. If None, the
            product is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or 
            squeezing them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the 
            product function.

        """
        super().__init__(
            xp.prod,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Mean(Reducer):
    """Compute the arithmetic mean along the specified axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the mean is performed. If None, the mean
        is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the mean
        function.

    """

    def __init__(
        self: Mean,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Mean feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        axis: int or tuple of int or None
            The axis or axes along which the mean is performed. If None, the 
            mean is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or 
            squeezing them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the 
            mean function.
    
        """
        
        def mean(
            image: np.ndarray | torch.Tensor | list | tuple, 
            axis: int | tuple[int, ...] | None = None, 
            keepdims: bool = False, 
            **kwargs: Any,
        ) -> np.ndarray | torch.Tensor:
            """Compute the mean of the input image along the specified axis.
            
            Parameters
            ----------
            image: array-like or list/tuple of array-like
                The input image or list/tuple of images to compute the mean of.
            axis: int or tuple of int or None
                The axis or axes along which the mean is performed. If None, 
                the mean is performed over all axes. 
            keepdims: bool
                Whether to keep the singleton dimensions after reducing or 
                squeezing them.
            **kwargs: Any
                Additional keyword arguments passed to the parent class and the
                mean function.

            Returns
            -------
            array-like
                The mean of the input image along the specified axis.
            
            """

            image = _as_float_if_needed(image)
            return xp.mean(image, axis=axis, keepdims=keepdims)
        
        super().__init__(
            mean,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Median(Reducer):
    """Compute the median along the specified axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the median is performed. If None, the
        median is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the median
        function.

    """

    def __init__(
        self: Median,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Median feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        axis: int or tuple of int or None
            The axis or axes along which the median is performed. If None, the
            median is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or squeezing
            them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the 
            median function.
        
        """

        def median(
            image: np.ndarray | torch.Tensor | list | tuple, 
            axis: int | tuple[int, ...] | None = None, 
            keepdims: bool = False, 
            **kwargs: Any,
        ) -> np.ndarray | torch.Tensor:
            """Compute the median of the input image along the specified axis.

            Parameters
            ----------
            image: array-like or list/tuple of array-like
                The input image or list/tuple of images to compute the median 
                of.
            axis: int or tuple of int or None
                The axis or axes along which the median is performed. If None, 
                the median is performed over all axes. 
            keepdims: bool
                Whether to keep the singleton dimensions after reducing or 
                squeezing them.
            **kwargs: Any
                Additional keyword arguments passed to the parent class and the 
                median function.

            Returns
            -------
            array-like
                The median of the input image along the specified axis.

            """

            image = _as_float_if_needed(image)
            return xp.quantile(image, 0.5, axis=axis, keepdims=keepdims)
        
        super().__init__(
            median,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Std(Reducer):
    """Compute the standard deviation along the specified axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the standard deviation is performed. If 
        None, the standard deviation is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the 
        standard deviation function.
    
    """

    def __init__(
        self: Std,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs,
    ):
        """Initialize the Std feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        axis: int or tuple of int or None
            The axis or axes along which the standard deviation is performed. 
            If None, the standard deviation is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or 
            squeezing them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the 
            standard deviation function.
        
        """
        
        def std(
            image: np.ndarray | torch.Tensor, 
            axis: int | tuple[int, ...] | None = None, 
            keepdims: bool = False, 
            **kwargs: Any,
        ) -> np.ndarray | torch.Tensor:
            """Compute the standard deviation along the specified axis.

            Parameters
            ----------
            image: array-like or list/tuple of array-like
                The input image or list/tuple of images to compute the standard
                deviation of.
            axis: int or tuple of int or None
                The axis or axes along which the standard deviation is 
                performed.
            keepdims: bool
                Whether to keep the singleton dimensions after reducing or 
                squeezing them.
            **kwargs: Any
                Additional keyword arguments passed to the parent class and the
                standard deviation function.

            Returns
            -------
            array-like
                The standard deviation of the input image along the specified 
                axis.

            """
        
            image = _as_float_if_needed(image)
            return xp.std(image, axis=axis, keepdims=keepdims)
        
        super().__init__(
            std,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Variance(Reducer):
    """Compute the variance along the specified axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the variance is performed. If None, the
        variance is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the 
        variance function.
        
    """

    def __init__(
        self: Variance,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs,
    ):
        """Initialize the Variance feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        axis: int or tuple of int or None
            The axis or axes along which the variance is performed. If None, 
            the variance is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or
            squeezing them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the
            variance function.
            
            """
        
        def variance(
            image: np.ndarray | torch.Tensor, 
            axis: int | tuple[int, ...] | None = None, 
            keepdims: bool = False, 
            **kwargs: Any,
        )-> np.ndarray | torch.Tensor:
            """Compute the variance along the specified axis.
            
            Parameters
            ----------
            image: array-like or list/tuple of array-like
                The input image or list/tuple of images to compute the variance
                of.
            axis: int or tuple of int or None
                The axis or axes along which the variance is performed.
            keepdims: bool
                Whether to keep the singleton dimensions after reducing or
                squeezing them.
            **kwargs: Any
                Additional keyword arguments passed to the parent class and the
                variance function.

            Returns
            -------
            array-like
                The variance of the input image along the specified axis.
            
            """
            
            image = _as_float_if_needed(image)
            return xp.var(image, axis=axis, keepdims=keepdims)
        
        super().__init__(
            variance,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Cumsum(Reducer):
    """Compute the cumulative sum along the specified axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the cumulative sum is performed. If None,
        the cumulative sum is performed over all axes.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the
        cumulative sum function.
        
    """

    def __init__(
        self: Cumsum,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        distributed: bool = True,
        **kwargs,
    ):
        """Initialize the Cumsum feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        axis: int or tuple of int or None
            The axis or axes along which the cumulative sum is performed. If 
            None, the cumulative sum is performed over all axes.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the
            cumulative sum function.
            
        """

        super().__init__(
            xp.cumsum,
            feature=feature,
            axis=axis,
            distributed=distributed,
            **kwargs,
        )


class Min(Reducer):
    """Return the minimum of an array or minimum along an axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the minimum is performed. If None, the
        minimum is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the minimum
        function.

    """

    def __init__(
        self: Min,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs,
    ):
        super().__init__(
            xp.min,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Max(Reducer):
    """Return the maximum of an array or maximum along an axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the maximum is performed. If None, the
        maximum is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the maximum
        function.
    
    """

    def __init__(
        self: Max,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            xp.max,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class PeakToPeak(Reducer):
    """Range of values (maximum - minimum) along an axis.
    
    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    axis: int or tuple of int or None
        The axis or axes along which the range is performed. If None, the range
        is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the range
        function.

    """

    def __init__(
        self: PeakToPeak,
        feature: Feature | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        def ptp(
            image: np.ndarray | torch.Tensor, 
            axis: int | tuple[int, ...] | None = None, 
            keepdims: bool = False, 
            **kwargs: Any,
        )-> np.ndarray | torch.Tensor:
            """Compute the range (max - min) along the specified axis.
            
            Parameters
            ----------
            image: array-like or list/tuple of array-like
                The input image or list/tuple of images to compute the range 
                of.
            axis: int or tuple of int or None
                The axis or axes along which the range is performed.
            keepdims: bool
                Whether to keep the singleton dimensions after reducing or
                squeezing them.
            **kwargs: Any
                Additional keyword arguments passed to the parent class and the
                range function.
            
            Returns
            -------
            array-like
                The range (max - min) of the input image along the specified 
                axis.
            
            """
            
            return xp.max(image, axis=axis, keepdims=keepdims) - xp.min(
                image, axis=axis, keepdims=keepdims
            )

        super().__init__(
            ptp,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Quantile(Reducer):
    """Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    q: float
        Quantile to compute, 0 through 1.
    axis: int or tuple of int or None
        The axis or axes along which the quantile is performed. If None, the
        quantile is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the 
        quantile function.

    """

    def __init__(
        self: Quantile,
        feature: Feature | None = None,
        q: float = 0.95,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Quantile feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        q: float
            Quantile to compute, 0 through 1.
        axis: int or tuple of int or None
            The axis or axes along which the quantile is performed. If None, 
            the quantile is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or 
            squeezing them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the
            quantile function.
                
        """

        def quantile(
            image: np.ndarray | torch.Tensor, 
            **kwargs: Any,
        ) -> np.ndarray | torch.Tensor:
            """Compute the q-th quantile along the specified axis.

            Parameters
            ----------
            image: array-like or list/tuple of array-like
                The input image or list/tuple of images to compute the quantile
                of.
            **kwargs: Any
                Additional keyword arguments passed to the parent class and the
                quantile function.

            Returns
            -------
            array-like
                The q-th quantile of the input image along the specified axis.
            
            """

            image = _as_float_if_needed(image)
            return xp.quantile(image, self.q(), **kwargs)

        super().__init__(
            quantile,
            feature=feature,
            q=q,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )


class Percentile(Reducer):
    """Compute the q-th percentile of the data along the specified axis.

    Parameters
    ----------
    feature: Feature, optional
        If not None, the output of this feature is used as the input.
    q: float
        Percentile to compute, 0 through 100.
    axis: int or tuple of int or None
        The axis or axes along which the percentile is performed. If None, the
        percentile is performed over all axes.
    keepdims: bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    distributed: bool
        Whether to apply the reducer to each image in the input list
        individually.
    **kwargs: Any
        Additional keyword arguments passed to the parent class and the
        percentile function.

    """

    def __init__(
        self: Percentile,
        feature: Feature | None = None,
        q: float = 95,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        distributed: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Percentile feature.
        
        Parameters
        ----------
        feature: Feature, optional
            If not None, the output of this feature is used as the input.
        q: float
            Percentile to compute, 0 through 100.
        axis: int or tuple of int or None
            The axis or axes along which the percentile is performed. If None,
            the percentile is performed over all axes.
        keepdims: bool
            Whether to keep the singleton dimensions after reducing or 
            squeezing them.
        distributed: bool
            Whether to apply the reducer to each image in the input list
            individually.
        **kwargs: Any
            Additional keyword arguments passed to the parent class and the
            percentile function.
                
        """
        
        def percentile(
            image: np.ndarray | torch.Tensor, 
            **kwargs: Any,
        ) -> np.ndarray | torch.Tensor:
            """Compute the q-th percentile along the specified axis.
            
            Parameters
            ----------
            image: array-like or list/tuple of array-like
                The input image or list/tuple of images to compute the 
                percentile of.
            **kwargs: Any
                Additional keyword arguments passed to the parent class and the
                percentile function.

            Returns
            -------
            array-like
                The q-th percentile of the input image along the specified axis.
            
            """
            
            image = _as_float_if_needed(image)
            return xp.quantile(image, self.q() / 100, **kwargs)

        super().__init__(
            percentile,
            feature=feature,
            q=q,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs,
        )
