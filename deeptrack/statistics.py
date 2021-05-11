from typing import List
from .features import Feature
from . import Image
import numpy as np


class Reducer(Feature):
    def __init__(self, function, feature=None, **kwargs):
        self.function = function

        if feature:
            super().__init__(_input=feature, **kwargs)
        else:
            super().__init__(**kwargs)

    def _process_and_get(self, image_list, **feature_input) -> List[Image]:

        self.__distributed__ = feature_input["distributed"]
        return super()._process_and_get(image_list, **feature_input)

    def get(self, image, axis, keepdims=None, **kwargs):
        if keepdims is None:
            return self.function(image, axis=axis)
        else:
            return self.function(image, axis=axis, keepdims=keepdims)


class Sum(Reducer):
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
    def __init__(self, feature=None, axis=None, distributed=True, **kwargs):
        super().__init__(
            np.cumsum, feature=feature, axis=axis, distributed=distributed, **kwargs
        )


class Min(Reducer):
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


class PeakToPeak(Reducer):
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
    def __init__(
        self,
        feature=None,
        q=0.95,
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
