"""Tools to manage the properties of a feature

Classes
-------
Property
    The class `Property`, which represents the values of a property of a feature.
    A Property be:
    * A constant (initialization with, e.g., a number, a tuple)
    * A sequence of variables (initialization with, e.g., an iterator)
    * A random variable (initialization with, e.g., a function)
SequentialProperty
    The class `SequentialProperty`, which extends `Property` to sample one value
    for each step in a sequence.
PropertyDict
    The class `PropertyDict`, which is a dictionary with each element a Property.
    The class provides utility functions to update, sample, clear and retrieve
    properties.
"""

import numpy as np
from .utils import (
    isiterable,
    get_kwarg_names,
)


from .backend.core import DeepTrackNode
from . import features


class Property(DeepTrackNode):
    def __init__(self, sampling_rule, **kwargs):
        super().__init__()
        self.action = self.create_action(sampling_rule, **kwargs)

    def create_action(self, sampling_rule, **dependencies):

        if isinstance(sampling_rule, DeepTrackNode):
            sampling_rule.add_child(self)
            self.add_dependency(sampling_rule)
            return sampling_rule

        if isinstance(sampling_rule, dict):
            dict_of_actions = dict(
                (key, self.create_action(val, **dependencies))
                for key, val in sampling_rule.items()
            )

            return lambda _ID=(): dict(
                (key, val(_ID=_ID)) for key, val in dict_of_actions.items()
            )

        if isinstance(sampling_rule, list):
            list_of_actions = [
                self.create_action(val, **dependencies) for val in sampling_rule
            ]

            return lambda _ID=(): [val(_ID=_ID) for val in list_of_actions]

        if isinstance(sampling_rule, (tuple, np.ndarray)):
            return lambda _ID=(): sampling_rule

        if isiterable(sampling_rule):
            # If it's iterable, return the next value

            def wrapped_iterator():

                while True:
                    try:
                        next_value = next(sampling_rule)
                    except StopIteration:
                        pass

                    yield next_value

            iterator = wrapped_iterator()

            def action(_ID=()):
                return next(iterator)

            return action

        if isinstance(sampling_rule, slice):

            start = self.create_action(sampling_rule.start, **dependencies)
            stop = self.create_action(sampling_rule.stop, **dependencies)
            step = self.create_action(sampling_rule.step, **dependencies)

            return lambda _ID=(): slice(
                start(_ID=_ID),
                stop(_ID=_ID),
                step(_ID=_ID),
            )

        if callable(sampling_rule):

            knames = get_kwarg_names(sampling_rule)

            used_dependencies = dict(
                (key, dep) for key, dep in dependencies.items() if key in knames
            )

            for dep in used_dependencies.values():
                dep.add_child(self)
                self.add_dependency(dep)

            return lambda _ID=(): sampling_rule(
                **{
                    **({"_ID": _ID} if "_ID" in knames else {}),
                    **dict(
                        (key, dep(_ID=_ID)) for key, dep in used_dependencies.items()
                    ),
                }
            )

        return lambda _ID=(): sampling_rule


class PropertyDict(DeepTrackNode, dict):
    """Dictionary with Property elements

    A dictionary of properties. It provides utility functions to update,
    sample, reset and retrieve properties.

    Parameters
    ----------
    *args, **kwargs
        Arguments used to initialize a dict

    """

    def __init__(self, **kwargs):

        dependencies = {}

        while kwargs:

            for key, val in list(kwargs.items()):
                try:
                    dependencies[key] = Property(val, **{**dependencies, **kwargs})
                    kwargs.pop(key)
                except AttributeError:
                    pass

        def action(_ID=()):
            return dict((key, val(_ID=_ID)) for key, val in self.items())

        super().__init__(action, **dependencies)

        for val in dependencies.values():
            val.add_child(self)
            self.add_dependency(val)


def propagate_data_to_dependencies(X, **kwargs):
    for dep in X.recurse_dependencies():
        if isinstance(dep, PropertyDict):
            for key, value in kwargs.items():
                if key in dep:
                    dep[key].set_value(value)


class SequentialProperty(Property):
    """Property that has multiple sequential values

    Extends standard `Property` to resolve one value for each step
    in a sequence of images. This is often used when creating movies.

    Parameters
    ----------
    initializer : any
        Sampling rule for the first step of the sequence.
    sampling_rule : any
        Sampling rule for each step after the first.

    Attributes
    ----------
    initializer : any
        Sampling rule for the first step of the sequence.
    sampling_rule : any
        Sampling rule for each step after the first.
    has_updated_since_last_resolve : bool
        Whether the property has been updated since the last resolve.

    """

    def __init__(self, initialization=None, **kwargs):

        super().__init__(None)

        # Create extra dependencies
        self.sequence_length = Property(0)
        self.add_dependency(self.sequence_length)
        self.sequence_length.add_child(self)

        self.sequence_step = Property(0)
        self.add_dependency(self.sequence_step)
        self.sequence_step.add_child(self)

        self.previous_values = Property(
            lambda _ID=(): self.previous(_ID=_ID)[: self.sequence_step() - 1]
            if self.sequence_step(_ID=_ID)
            else []
        )
        self.add_dependency(self.previous_values)
        self.previous_values.add_child(self)
        self.previous_values.add_dependency(self.sequence_step)
        self.sequence_step.add_child(self.previous_values)

        self.previous_value = Property(
            lambda _ID=(): self.previous(_ID=_ID)[self.sequence_step() - 1]
            if self.previous(_ID=_ID)
            else None
        )
        self.add_dependency(self.previous_value)
        self.previous_value.add_child(self)
        self.previous_value.add_dependency(self.sequence_step)
        self.sequence_step.add_child(self.previous_value)

        if initialization:
            self.initialization = self.create_action(initialization, **kwargs)
        else:
            self.initialization = None

        self.current = lambda: None
        self.action = self._action

    def _action(self, _ID=()):
        return (
            self.initialization(_ID=_ID)
            if self.sequence_step(_ID=_ID) == 0
            else self.current(_ID=_ID)
        )

    def store(self, value, _ID=()):
        try:
            current_data = self.data[_ID].current_value()
        except KeyError:
            current_data = []

        super().store(current_data + [value], _ID=_ID)

    def current_value(self, _ID):
        return super().current_value(_ID=_ID)[self.sequence_step(_ID=_ID)]

    def __call__(self, _ID=()):
        return super().__call__(_ID=_ID)

    def update(self, sequence_length=0, **kwargs):
        """Updates current_value

        For each step in the sequence, sample `self.sampling_rule`.
        `self.initializer` is used for the first step. These rules
        should output one value per step. Sampling rules
        that are functions can optionally accept the following keyword
        arguments:

        * sequence_step : the current position in the sequence.
        * sequence_length : the length of the sequence.
        * previous_value : the value of the property at the previous
          step in the sequence.
        * previous_values : the value of the property at all previous
          steps in the sequence.

        Parameters
        ----------
        sequence_length : int, optional
            length of the sequence

        Returns
        -------
        self
            returns self
        """
        my_id = id(self)
        if (
            features.UPDATE_LOCK.locked()
            and my_id in features.UPDATE_MEMO["memoization"]
        ):
            return self

        kwargs.update(features.UPDATE_MEMO["user_arguments"])

        new_current_value = []

        for step in range(sequence_length):
            # Use initializer for first time step
            ruleset = self.sampling_rule

            # Elements inserted here can be passed to property functions
            kwargs.update(
                sequence_step=step,
                sequence_length=sequence_length,
                previous_values=new_current_value,
            )
            if step == 0:

                kwargs.update(previous_value=self.sample(self.initializer, **kwargs))
            else:
                kwargs.update(previous_value=new_current_value[-1])

            next_value = self.sample(ruleset, **kwargs)

            new_current_value.append(next_value)

        self.current_value = new_current_value
        features.UPDATE_MEMO["memoization"][my_id] = new_current_value
        return self
