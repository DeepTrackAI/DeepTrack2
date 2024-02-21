"""Features and tools for resolving sequences of images.

Classes
-------
Sequence
    Resolves a feature as a sequence.

Functions
---------
Sequential
    Converts a feature to be resolved as a sequence.
"""

from .features import Feature
from .properties import SequentialProperty
from .types import PropertyLike
import random
import numpy as np

class Sequence(Feature):
    """Resolves a feature as a sequence.

    The input feature is resolved `sequence_length` times, with the kwarg
    arguments `sequene_length` and `sequence_step` passed to all properties
    of the feature set.

    Parameters
    ----------
    feature : Feature
        The feature to resolve as a sequence.
    sequence_length : int
        The number of times to resolve the feature.

    Attributes
    ----------
    feature : Feature
        The feature to resolve as a sequence.
    """

    __distributed__ = False

    def __init__(
        self, feature: Feature, sequence_length: PropertyLike[int] = 1, **kwargs
    ):
        
        super().__init__(sequence_length=sequence_length, **kwargs)
        self.feature = self.add_feature(feature)
        # Require update
        # self.update()

    def get(self, input_list, sequence_length=None, **kwargs):

        outputs = input_list or []
        for sequence_step in range(sequence_length):
            np.random.seed(random.randint(0, 1000000))

            propagate_sequential_data(
                self.feature,
                sequence_step=sequence_step,
                sequence_length=sequence_length,
            )
            out = self.feature()

            outputs.append(out)

        if isinstance(outputs[0], (tuple, list)):
            outputs = tuple(zip(*outputs))

        return outputs


def Sequential(feature: Feature, **kwargs):
    """Converts a feature to be resolved as a sequence.

    Should be called on individual features, not combinations of features. All
    keyword arguments will be trated as sequential properties and will be
    passed to the parent feature.

    If a property from the keyword argument already exists on the feature, the
    existing property will be used to initilize the passed property (that is,
    it will be used for the first timestep).

    Parameters
    ----------
    feature : Feature
        Feature to make sequential.
    kwargs
        Keyword arguments to pass on as sequential properties of `feature`.

    """

    for property_name in kwargs.keys():

        if property_name in feature.properties:
            # Insert property with initialized value
            feature.properties[property_name] = SequentialProperty(
                feature.properties[property_name], **feature.properties
            )
        else:
            # insert empty property
            feature.properties[property_name] = SequentialProperty()

        feature.properties.add_dependency(feature.properties[property_name])
        feature.properties[property_name].add_child(feature.properties)

    for property_name, sampling_rule in kwargs.items():

        prop = feature.properties[property_name]

        all_kwargs = dict(
            previous_value=prop.previous_value,
            previous_values=prop.previous_values,
            sequence_length=prop.sequence_length,
            sequence_step=prop.sequence_step,
        )

        for key, val in feature.properties.items():
            if key == property_name:
                continue

            if isinstance(val, SequentialProperty):
                all_kwargs[key] = val
                all_kwargs["previous_" + key] = val.previous_values
            else:
                all_kwargs[key] = val
        if not prop.initialization:
            prop.initialization = prop.create_action(sampling_rule, **{k:all_kwargs[k] for k in all_kwargs if k != "previous_value"})

        prop.current = prop.create_action(sampling_rule, **all_kwargs)

    return feature


def propagate_sequential_data(X, **kwargs):
    for dep in X.recurse_dependencies():
        if isinstance(dep, SequentialProperty):
            for key, value in kwargs.items():
                if hasattr(dep, key):
                    getattr(dep, key).set_value(value)
