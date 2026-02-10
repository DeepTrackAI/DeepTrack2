"""Tools for evaluating and propagating sequences of features.

This module enables sequential evaluation of DeepTrack2 features by
resolving them over multiple time steps. It provides tools for propagating
values like `sequence_index` and `sequence_length` to all dependent
`SequentialProperty` attributes, allowing simulation of dynamic behaviors
(e.g., microsocpy videos).

Key Features
------------
- **Temporal Simulation via SequentialProperty**

    Features can be annotated with sampling rules that evolve over a sequence
    of time steps, enabling animations and simulations of time-dependent
    systems.

- **Graph-wide Sequential Data Propagation**

    Sequential information is passed to all relevant nodes in the feature
    graph.

Module Structure
----------------
Classes:

- `Sequence`

    It resolves a feature over multiple time steps, using a defined
    `sequence_length`. Injects sequential arguments into all dependent
    `SequentialProperty` attributes before each evaluation.

Functions:

- `Sequential(feature, **kwargs)`

    .. deprecated:: 2.0

    def Sequential(
        feature: Feature,
        **kwargs: Any,
    ) -> Feature

    Converts a feature to be resolved as a sequence. Replaced by
    `Feature.to_sequence()` and will be removed in a future release.

- `_propagate_sequential_data(feature, **kwargs)`

    def _propagate_sequential_data(
        feature: Feature,
        **kwargs: Any,
    ) -> None

    Recursively propagates keyword arguments like `sequence_index` and
    `sequence_length` to all `SequentialProperty` nodes in a feature graph.

Examples
--------
>>> import deeptrack as dt

Simulating a spinning ellipsoid.

Define imaging system:
>>> optics = dt.optics.Fluorescence(output_region=(0, 0, 32, 32))

Define a static ellipse:
>>> ellipse = dt.scatterers.Ellipse(
...     radius=(1e-6,0.5e-6),
...     position=(16, 16),
...     rotation=0.78,  # Initial rotation
... )

Define a rotation function that increments the previous angle:
>>> def rotate(sequence_length, previous_value):
...    return previous_value + 6.28 / sequence_length

Convert the ellipse to a sequential feature:
>>> rotating_ellipse = ellipse.to_sequential(rotation=rotate)

Compose with the optics:
>>> imaged_rotating_ellipse = optics(rotating_ellipse)

Wrap the full feature in a Sequence:
>>> imaged_rotating_ellipse_sequence = dt.Sequence(
...     imaged_rotating_ellipse,
...     sequence_length=50,
... )

Generate and display the result
>>> imaged_rotating_ellipse_sequence.update().plot();

"""

from __future__ import annotations

from typing import Any

from deeptrack.features import Feature
from deeptrack.properties import SequentialProperty
from deeptrack.types import PropertyLike


__all__ = ["Sequence"]


class Sequence(Feature):
    """Resolves a feature as a sequence.

    The `Sequence` class repeatedly evaluates a given feature
    `sequence_length` times. During each evaluation, the keyword arguments
    `sequence_length` and `sequence_index` are propagated to all
    `SequentialProperty` attributes of the feature, enabling dynamic updates at
    each timestep.

    This allows for temporal simulations or animations, where the same feature
    (e.g., a rotating particle or moving object) evolves over time with
    properties defined as sequential functions.

    Parameters
    ----------
    feature: Feature
        The feature to resolve as a sequence.
    sequence_length: int
        The number of times to evaluate the feature. It defaults to 1.
    kwargs: Any
        Additional keyword arguments to be passed to the base `Feature`.

    Attributes
    ----------
    feature: Feature
        The feature that is resolved multiple times to generate the sequence.
    __distributed__: bool
        This feature is not distributed across processes or devices.
        Always set to False.

    Methods
    -------
    `get(input_list: list[Feature], sequence_length: int, **kwargs: Any) -> list[Any] or tuple[list[Any], ...]`
        Resolves the wrapped feature `sequence_length` times. It returns a list
        (or tuple of lists) of resolved outputs.

    Examples
    --------
    >>> import deeptrack as dt

    Simulating a spinning ellipsoid.

    Define imaging system:
    >>> optics = dt.Fluorescence(output_region=(0, 0, 32, 32))

    Define a static ellipse:
    >>> ellipse = dt.Ellipse(
    ...     radius=(1e-6,0.5e-6),
    ...     position=(16, 16),
    ...     rotation=0.78,  # Initial rotation
    ... )

    Define a rotation function that increments the previous angle:
    >>> def rotate(sequence_length, previous_value):
    ...    return previous_value + 6.28 / sequence_length

    Convert the ellipse to a sequential feature:
    >>> rotating_ellipse = ellipse.to_sequential(rotation=rotate)

    Compose with the optics:
    >>> imaged_rotating_ellipse = optics(rotating_ellipse)

    Wrap the full feature in a Sequence:
    >>> imaged_rotating_ellipse_sequence = dt.Sequence(
    ...     imaged_rotating_ellipse,
    ...     sequence_length=50,
    ... )

    Generate and display the result
    >>> imaged_rotating_ellipse_sequence.update().plot();

    """

    __distributed__ = False

    feature: Feature

    def __init__(
        self: Sequence,
        feature: Feature,
        sequence_length: PropertyLike[int] = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize a Sequence object.

        This constructor wraps a feature to be resolved multiple times,
        propagating sequential information to any `SequentialProperty`
        attributes.

        Parameters
        ----------
        feature: Feature
            The feature to be resolved as a sequence.
        sequence_length: PropertyLike[int], optional
            Number of steps in the sequence. It defaults to 1.
        **kwargs: Any
            Additional keyword arguments passed to the base `Feature`.

        """

        super().__init__(sequence_length=sequence_length, **kwargs)
        self.feature = self.add_feature(feature)

    def get(
        self: Sequence,
        input_list: list[Feature],
        sequence_length: int | None = None,
        **kwargs: Any,
    ) -> list[Any] | tuple[list[Any], ...]:
        """Resolve the wrapped feature as a sequence of outputs.

        The method evaluates the feature `sequence_length` times, each time
        updating the `sequence_index` and propagating it to all dependent
        `SequentialProperty` attributes. The results are collected into a list.

        Parameters
        ----------
        input_list: list[Feature]
            A list of previously resolved outputs to extend. If empty, a new
            list is initialized.
        sequence_length: int, optional
            Number of times to evaluate the feature. If None, it is assumed
            to be handled externally or will raise an error.
        **kwargs: Any
            Unused, included for compatibility.

        Returns
        -------
        list[Any] or tuple[list[Any], ...]
            The sequence of resolved feature outputs. If the output of the
            feature is a tuple or list, the return is transposed into a tuple
            of lists.

        """

        outputs = input_list or []
        for sequence_index in range(sequence_length):
            #TODO ***BM*** ***AL*** Can this be erased?
            # np.random.seed(random.randint(0, 1000000))

            _propagate_sequential_data(
                self.feature,
                sequence_index=sequence_index,
                sequence_length=sequence_length,
            )
            out = self.feature()

            outputs.append(out)

        if isinstance(outputs[0], (tuple, list)):
            outputs = tuple(zip(*outputs))

        return outputs


def _propagate_sequential_data(
    feature: Feature,
    **kwargs: Any,
) -> None:
    """Propagate sequential data through the computational graph.

    This function updates the attributes of all `SequentialProperty` instances
    in the computational graph rooted at the given feature. It works by
    recursively traversing the feature's dependencies and setting the values
    of matching attributes using the provided keyword arguments.

    Parameters
    ----------
    feature: Feature
        The root feature whose dependent sequential properties will be updated.
    **kwargs: Any
        Attribute-value pairs to assign to matching fields in each
        `SequentialProperty`.

    """

    for dep in feature.recurse_dependencies():
        if isinstance(dep, SequentialProperty):
            for key, value in kwargs.items():
                if hasattr(dep, key):
                    getattr(dep, key).set_value(value)


def Sequential(feature: Feature, **kwargs: Any) -> Feature:  # DEPRECATED
    """Converts a feature to be resolved as a sequence.

    .. deprecated:: 2.0
        This function has been substituted by the `Feature.to_sequence()`
        method and will be removed in a future release.

    Should be called on individual features, not combinations of features. All
    keyword arguments will be treated as sequential properties and will be
    passed to the parent feature.

    If a property from the keyword argument already exists on the feature, the
    existing property will be used to initialize the passed property (that is,
    it will be used for the first timestep).

    Parameters
    ----------
    feature: Feature
        Feature to make sequential.
    kwargs: Any
        Keyword arguments to pass on as sequential properties of `feature`.

    Returns
    -------
    Feature
        The modified feature with sequential behavior.

    """

    import warnings

    warnings.warn(
        "The `Sequential()` function is deprecated and will be removed in a "
        "future release. Please use `Feature.to_sequence()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
            sequence_index=prop.sequence_index,
        )

        for key, val in feature.properties.items():
            if key == property_name:
                continue

            if isinstance(val, SequentialProperty):
                all_kwargs[key] = val
                all_kwargs["previous_" + key] = val.previous_values
            else:
                all_kwargs[key] = val
        if not prop.initial_sampling_rule:
            prop.initial_sampling_rule = prop.create_action(
                sampling_rule,
                **{
                    k:all_kwargs[k]
                    for k
                    in all_kwargs
                    if k != "previous_value"
                }
            )

        prop.current = prop.create_action(sampling_rule, **all_kwargs)

    return feature
