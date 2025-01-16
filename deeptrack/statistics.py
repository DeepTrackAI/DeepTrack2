""" Contains features that perform some statistics operation on the input.

These features reduce some dimension of the input by applying a statistical 
operation (sum, mean, etc.). They follow the syntax of the equivalent numpy
function, meaning that 'axis' and 'keepdims' are valid arguments. Moreover,
they all accept the `distributed` keyword, which determines if each image in
the input list should be handled individually or not.

Module Structure
----------------
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

Example
-------
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


from typing import List

import numpy as np

from deeptrack import Image
from deeptrack import Feature


class Reducer(Feature):
    """Base class of features that reduce the dimensionality of the input.

    Parameters
    ==========
    function : Callable
        The function used to reduce the input.
    feature : Feature, optional
        If not None, the output of this feature is used as the input.
    distributed : bool
        Whether to apply the reducer to each image in the input list
        individually.
    axis : int or tuple of int
        The axis / axes to reduce over.
    keepdims : bool
        Whether to keep the singleton dimensions after reducing or squeezing
        them.
    """

    def __init__(self, function, feature=None, distributed=True, **kwargs):
        self.function = function

        if feature:
            super().__init__(_input=feature, distributed=distributed, **kwargs)
        else:
            super().__init__(distributed=distributed, **kwargs)

    def _process_and_get(self, image_list, **feature_input) -> List[Image]:
        self.__distributed__ = feature_input["distributed"]
        return super()._process_and_get(image_list, **feature_input)

    def get(self, image, axis, keepdims=None, **kwargs):
        if keepdims is None:
            return self.function(image, axis=axis)
        else:
            return self.function(image, axis=axis, keepdims=keepdims)


class Sum(Reducer):
    """Compute the sum along the specified axis"""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.sum,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Prod(Reducer):
    """Compute the product along the specified axis"""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.prod,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Mean(Reducer):
    """Compute the arithmetic mean along the specified axis."""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.mean,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Median(Reducer):
    """Compute the median along the specified axis."""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.median,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Std(Reducer):
    """Compute the standard deviation along the specified axis."""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.std,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Variance(Reducer):
    """Compute the variance along the specified axis."""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.var,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Cumsum(Reducer):
    """Compute the cummulative sum along the specified axis."""

    def __init__(self, feature=None, axis=None, distributed=True, **kwargs):
        super().__init__(
            np.cumsum,
            feature=feature,
            axis=axis,
            distributed=distributed,
            **kwargs
        )


class Min(Reducer):
    """Return the minimum of an array or minimum along an axis."""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.min,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Max(Reducer):
    """Return the maximum of an array or maximum along an axis."""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.max,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class PeakToPeak(Reducer):
    """Range of values (maximum - minimum) along an axis."""

    def __init__(
        self,
        feature=None,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        super().__init__(
            np.ptp,
            feature=feature,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Quantile(Reducer):
    """Compute the q-th quantile of the data along the specified axis.

    Parameters
    ==========
    q : float
       Quantile to compute (0 through 1).
    """

    def __init__(
        self,
        feature=None,
        q=0.95,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        def quantile(image, **kwargs):
            return np.quantile(image, self.q(), **kwargs)

        super().__init__(
            quantile,
            feature=feature,
            q=q,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )


class Percentile(Reducer):
    """Compute the q-th percentile of the data along the specified axis.

    Parameters
    ==========
    q : float
       Percentile to compute, (0 through 100).
    """

    def __init__(
        self,
        feature=None,
        q=95,
        axis=None,
        keepdims=False,
        distributed=True,
        **kwargs
    ):
        def percentile(image, **kwargs):
            return np.percentile(image, self.q(), **kwargs)

        super().__init__(
            percentile,
            feature=feature,
            q=q,
            axis=axis,
            keepdims=keepdims,
            distributed=distributed,
            **kwargs
        )
