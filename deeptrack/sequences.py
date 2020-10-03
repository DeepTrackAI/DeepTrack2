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

from deeptrack.features import Feature
from deeptrack.properties import SequentialProperty


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

    def __init__(self, feature: Feature, sequence_length: int = 1, **kwargs):
        self.feature = feature
        super().__init__(sequence_length=sequence_length, **kwargs)

        # Require update
        self.update()

    def get(self, input_list, sequence_length=None, **kwargs):
        return (input_list or []) + [
            self.feature.resolve(
                sequence_length=sequence_length, sequence_step=sequence_step, **kwargs
            )
            for sequence_step in range(sequence_length)
        ]

    def update(self, **kwargs):
        super().update(**kwargs)

        sequence_length = self.properties["sequence_length"].current_value
        if "sequence_length" not in kwargs:
            kwargs["sequence_length"] = sequence_length

        self.feature.update(**kwargs)

        return self


def Sequential(feature: Feature, **kwargs):
    """Converts a feature to be resolved as a sequence.

    Should be called on individual features, not combinations of features. All keyword
    arguments will be trated as sequential properties and will be passed to the parent feature.

    If a property from the keyword argument already exists on the feature, the existing property
    will be used to initilize the passed property (that is, it will be used for the first timestep).

    Parameters
    ----------
    feature : Feature
        Feature to make sequential.
    kwargs
        Keyword arguments to pass on as sequential properties of `feature`.

    """

    for property_name, sampling_rule in kwargs.items():

        if property_name in feature.properties:
            initializer = feature.properties[property_name].sampling_rule
        else:
            initializer = sampling_rule

        feature.properties[property_name] = SequentialProperty(
            sampling_rule, initializer=initializer
        )
        feature.properties[property_name].parent = feature.properties

    return feature
