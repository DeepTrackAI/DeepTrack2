"""Tools for evaluating and propagating sequences of features.

This module provides functionality for sequentially evaluating DeepTrack2
features over multiple time steps. It enables the propagation of sequential
context—such as `sequence_index` and `sequence_length`—to all dependent
`SequentialProperty` attributes in a feature graph.

By injecting this contextual information before each evaluation, the module
supports simulations of dynamic, time-dependent systems, such as microscopy
videos, animations, and temporal data generation pipelines.

Key Features
------------
- **Temporal simulation via `SequentialProperty`**

    Features can be annotated with sampling rules that evolve across discrete
    time steps. These rules may depend on the current step index, the total
    sequence length, or values from previous steps.

- **Graph-wide sequential data propagation**

    Sequential context is propagated to all relevant nodes in the feature
    dependency graph, ensuring consistent and synchronized updates across
    composed and nested features.

Module Structure
----------------
Classes:

- `Sequence`

    Resolves a feature repeatedly over a specified number of time steps
    (`sequence_length`). Before each evaluation, sequential context is
    propagated to all dependent `SequentialProperty` attributes.

Examples
--------
>>> import deeptrack as dt

**Sequential evaluation**

In this example, a feature is evaluated repeatedly while one of its properties
evolves over time. No optics or image formation is involved.

Define a simple feature with a time-dependent property:

>>> feature = dt.Value(value=0)

Define a sampling rule that increments the value at each step:

>>> def increment(sequence_length, previous_value):
...     return previous_value + 1

Convert the feature to a sequential feature:

>>> sequential_feature = feature.to_sequential(value=increment)

Wrap the feature in a `Sequence` and evaluate it:

>>> sequence = dt.Sequence(sequential_feature, sequence_length=5)
>>> sequence()
[0, 1, 2, 3, 4]

**Simulating a spinning ellipsoid.**

Define an imaging system:

>>> optics = dt.Fluorescence(output_region=(0, 0, 32, 32))

Define a static ellipse:

>>> ellipse = dt.Ellipse(
...     radius=(1e-6, 0.5e-6),
...     position=(16, 16),
...     rotation=0.78,  # Initial rotation
...     intensity=1,
... )

Define a rotation function that increments the previous angle:

>>> def rotate(sequence_length, previous_value):
...     return previous_value + 6.28 / sequence_length

Convert the ellipse to a sequential feature:

>>> rotating_ellipse = ellipse.to_sequential(rotation=rotate)

Compose with the optics:

>>> imaged_rotating_ellipse = optics(rotating_ellipse)

Wrap the composed feature in a `Sequence`:

>>> imaged_rotating_ellipse_sequence = dt.Sequence(
...     imaged_rotating_ellipse,
...     sequence_length=50,
... )

Generate and display the result:

>>> imaged_rotating_ellipse_sequence.update().plot();

"""


from __future__ import annotations

from typing import Any

from deeptrack.features import Feature
from deeptrack.properties import SequentialProperty
from deeptrack.types import PropertyLike


__all__ = ["Sequence"]


class Sequence(Feature):
    """Resolve a feature repeatedly as a sequence.

    The `Sequence` class evaluates a wrapped feature multiple times in
    succession, producing a sequence of outputs. Before each evaluation, the
    sequential context (`sequence_index` and `sequence_length`) is propagated
    to all dependent `SequentialProperty` attributes in the feature graph.

    This enables temporal simulations and animations in which feature
    properties evolve over discrete time steps according to user-defined
    sampling rules. The wrapped feature itself may be a single feature or a
    composed feature graph.

    Parameters
    ----------
    feature: Feature
        The feature to be evaluated repeatedly.
    sequence_length: int
        The number of sequential evaluations to perform. Defaults to 1.
    **kwargs: Any
        Additional keyword arguments passed to the base `Feature` constructor.

    Attributes
    ----------
    feature: Feature
        The wrapped feature that is evaluated at each step.
    __distributed__: bool
        Indicates whether this feature is distributed across processes or
        devices. Always set to `False` for `Sequence`, as sequential evaluation
        requires ordered execution.

    Methods
    -------
    `get(input_list, sequence_length, _ID, **kwargs) -> list[Any] | tuple[...]`
        Evaluate the wrapped feature `sequence_length` times. The outputs are
        returned as a list. If the wrapped feature returns a tuple or list, the
        result is transposed into a tuple of lists.

    Examples
    --------
    >>> import deeptrack as dt

    **Sequential evaluation**

    In this example, a feature is evaluated repeatedly while one of its
    properties evolves over time. No optics or image formation is involved.

    Define a simple feature with a time-dependent property:

    >>> feature = dt.Value(value=0)

    Define a sampling rule that increments the value at each step:

    >>> def increment(sequence_length, previous_value):
    ...     return previous_value + 1

    Convert the feature to a sequential feature:

    >>> sequential_feature = feature.to_sequential(value=increment)

    Wrap the feature in a `Sequence` and evaluate it:

    >>> sequence = dt.Sequence(sequential_feature, sequence_length=5)
    >>> sequence()
    [0, 1, 2, 3, 4]

    **Simulating a spinning ellipsoid.**

    Define an imaging system:

    >>> optics = dt.Fluorescence(output_region=(0, 0, 32, 32))

    Define a static ellipse:

    >>> ellipse = dt.Ellipse(
    ...     radius=(1e-6, 0.5e-6),
    ...     position=(16, 16),
    ...     rotation=0.78,  # Initial rotation
    ...     intensity=1,
    ... )

    Define a rotation function that increments the previous angle:

    >>> def rotate(sequence_length, previous_value):
    ...     return previous_value + 6.28 / sequence_length

    Convert the ellipse to a sequential feature:

    >>> rotating_ellipse = ellipse.to_sequential(rotation=rotate)

    Compose with the optics:

    >>> imaged_rotating_ellipse = optics(rotating_ellipse)

    Wrap the composed feature in a `Sequence`:

    >>> imaged_rotating_ellipse_sequence = dt.Sequence(
    ...     imaged_rotating_ellipse,
    ...     sequence_length=50,
    ... )

    Generate and display the result:

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
        """Initialize a `Sequence` instance.

        This constructor wraps a feature so that it can be evaluated repeatedly
        as a sequence. The wrapped feature is added to the feature graph, and
        the `sequence_length` parameter is registered as a property of the
        `Sequence` node.

        Sequential context (`sequence_index` and `sequence_length`) is
        propagated to dependent `SequentialProperty` attributes during
        evaluation, not during initialization.

        Parameters
        ----------
        feature: Feature
            The feature to be evaluated sequentially.
        sequence_length: PropertyLike[int], optional
            The number of steps in the sequence. Defaults to 1.
        **kwargs: Any
            Additional keyword arguments passed to the base `Feature`
            constructor.

        """

        super().__init__(sequence_length=sequence_length, **kwargs)

        self.feature = self.add_feature(feature)

    def get(
        self: Sequence,
        input_list: list[Any] | None,
        sequence_length: int,
        _ID: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> list[Any] | tuple[list[Any], ...]:
        """Resolve the wrapped feature as a sequence of outputs.

        This method evaluates the wrapped feature `sequence_length` times.
        Before each evaluation, the sequential context (`sequence_index`
        and `sequence_length`) is propagated to all dependent
        `SequentialProperty` attributes in the feature graph.

        The outputs of each evaluation are collected and returned as a
        sequence. If the wrapped feature returns multiple values (as a tuple or
        list), the result is transposed into a tuple of lists, one per output
        component.

        Parameters
        ----------
        input_list: list[Any] or None
            Previously resolved outputs to extend. If `None`, a new output list
            is initialized.
        sequence_length: int
            Number of sequential evaluations to perform.
        _ID: tuple[int, ...], optional
            Evaluation identifier used to store and retrieve sequential state.
        **kwargs: Any
            Unused. Present for compatibility with the `Feature` interface.

        Returns
        -------
        list[Any] | tuple[list[Any], ...]
            The sequence of resolved outputs. If the wrapped feature returns a
            tuple or list, the result is a tuple of lists.

        """

        if sequence_length < 0:
            raise ValueError(
                "`sequence_length` must be non-negative, "
                f"got {sequence_length}."
            )

        output_list: list[Any] = list(input_list) if input_list else []

        for sequence_index in range(sequence_length):
            _propagate_sequential_data(
                self.feature,
                sequence_index=sequence_index,
                sequence_length=sequence_length,
                _ID=_ID,
            )
            out = self.feature(_ID=_ID)

            output_list.append(out)

        if not output_list:
            return output_list

        if isinstance(output_list[0], (tuple, list)):
            return tuple(list(x) for x in zip(*output_list))

        return output_list


def _propagate_sequential_data(
    feature: Feature,
    _ID: tuple[int, ...] = (),
    **kwargs: Any,
) -> None:
    """Propagate sequential context through a feature graph.

    This function propagates sequential context—such as `sequence_index` and
    `sequence_length`—to all dependent `SequentialProperty` attributes in the
    feature graph rooted at the given feature. The propagation is performed by
    traversing the feature’s dependency graph and updating matching attributes
    on each encountered `SequentialProperty`.

    Parameters
    ----------
    feature: Feature
        The root feature whose dependent sequential properties will be updated.
    _ID: tuple[int, ...], optional
        Evaluation identifier used to store propagated values.
    **kwargs: Any
        Sequential context to propagate, provided as attribute–value pairs.

    """

    for dep in feature.recurse_dependencies():
        if isinstance(dep, SequentialProperty):
            for key, value in kwargs.items():
                if hasattr(dep, key):
                    attr = getattr(dep, key, None)
                    set_value = getattr(attr, "set_value", None)
                    if callable(set_value):
                        set_value(value, _ID=_ID)



def Sequential(feature: Feature, **kwargs: Any) -> Feature:  # DEPRECATED
    """Convert a feature to be resolved sequentially.

    .. deprecated:: 2.0
        Use `Feature.to_sequential()` instead. This function will be removed in
        a future release.

    This function modifies a feature so that selected properties evolve over a
    sequence of evaluations. It should be applied to individual features rather
    than composed feature graphs.

    All keyword arguments are interpreted as sequential properties and attached
    to the feature. If a property with the same name already exists on the
    feature, its current value is used to initialize the sequential property at
    the first time step.

    Parameters
    ----------
    feature: Feature
        The feature to be converted to sequential behavior.
    **kwargs: Any
        Keyword arguments defining sequential properties of `feature`.

    Returns
    -------
    Feature
        The modified feature with sequential behavior.

    """

    import warnings

    warnings.warn(
        "The `Sequential()` function is deprecated and will be removed in a "
        "future release. Please use `Feature.to_sequential()` instead.",
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
