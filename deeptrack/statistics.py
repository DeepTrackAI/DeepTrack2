""" Contains features that perform some statistics operation on the input.

These features reduce some dimension of the input by calculating some statistical metric. They
follow the syntax of the equivalent numpy function, meaning that axis and keepdims are valid arguments.
Moreover, they all accept the `distributed` keyword, which determines if each image in the input list
should be handled individually or not. For example:

>>> input_values = [np.ones((2,)), np.zeros((2,))]
>>> Sum(axis=0, distributed=True)(input_values) # [1+1, 0+0]
>>> [2, 0]
>>> Sum(axis=0, distributed=False)(input_values) # [1+0, 1+0]
>>> [1, 1]

Reducers can be added to the pipeline in two ways:

>>> some_pipeline_of_features
>>> summed_pipeline = some_pipeline_of_features >> Sum(axis=0)
>>> summed_pipeline = Sum(some_pipeline_of_features, axis=0)

Combining the two, eg:

>>> incorrectly_summed_pipline = some_feature >> Sum(some_pipeline_of_features, axis=0)

is not supported and the behaviour is not guaranteed. However, other operators can be used in this way:

>>> correctly_summed_and_subtracted_pipline = some_feature - Sum(some_pipeline_of_features, axis=0)
"""


from typing import List

import numpy as np

from .image import Image
from .features import Feature


class Reducer(Feature):
    """Base class of features that reduce the dimensionality of the input.



    Parameters
    ==========
    function : Callable
        The function used to reduce the input
    feature : Feature, optional
        If not None, the output of this feature is used as the input.
    distributed : boolean
        Whether to apply the reducer to each image in the input list individually.
    axis : int, tuple of int
        The axis / axes to reduce over
    keepdims : boolean
        Whether to keep the singleton dimensions after reducing or squeeze them.


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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
            np.cumsum, feature=feature, axis=axis, distributed=distributed, **kwargs
        )


class Min(Reducer):
    """Return the minimum of an array or minimum along an axis."""

    def __init__(
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
        self, feature=None, axis=None, keepdims=False, distributed=True, **kwargs
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
       Quantile to compute, 0 through 1.
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
       Percentile to compute, 0 through 100.

    """

    def __init__(
        self, feature=None, q=95, axis=None, keepdims=False, distributed=True, **kwargs
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
