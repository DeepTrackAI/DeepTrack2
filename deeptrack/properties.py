"""Tools to manage the properties of a feature
"""

import numpy as np
from .utils import (
    isiterable,
    get_kwarg_names,
)


from .backend.core import DeepTrackNode
from . import features


class Property(DeepTrackNode):
    """Class that defines a property of a feature.

    A property contains one argument for evaluating feature. It can be a constant, a function, a list,
    a tuple, a numpy array, a slice, a dictionary, a DeepTrackNode (such as another property or feature),
    or a generator. If it is a function, it can have the names of other properties of the same feature as arguments.
    The output of the function is the value of the property. If it is a list, each element of the list will be sampled individually.
    If it is a dictionary, each value of the dictionary will be sampled individually. If it is a tuple or an array, it will be treated as a constant.
    If it is a slice, the start, stop and step will be sampled individually. If it is a generator, the next value will be sampled.
    When the generator is exhausted, it will yield the final value infinitely.

    Parameters
    ----------
    sampling_rule : function, list, tuple, numpy array, slice, dictionary, DeepTrackNode, or generator
        The rule to sample the property.
    """

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
                    # Yield the final value infinitely
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

            # Extract the arguments that are also properties
            used_dependencies = dict(
                (key, dep) for key, dep in dependencies.items() if key in knames
            )

            # Add the dependencies of the function as dependencies.
            for dep in used_dependencies.values():
                dep.add_child(self)
                self.add_dependency(dep)

            # Create the action.
            return lambda _ID=(): sampling_rule(
                **{key: dep(_ID=_ID) for key, dep in used_dependencies.items()},
                **({"_ID": _ID} if "_ID" in knames else {}),
            )

        return lambda _ID=(): sampling_rule


class SequentialProperty(Property):
    """Property that has multiple sequential values

    Extends standard `Property` to resolve one value for each step
    in a sequence of images. This is often used when creating movies.

    Parameters
    ----------
    initialization : any
        Sampling rule for the first step of the sequence.


    """

    def __init__(self, initialization=None, **kwargs):

        super().__init__(None)

        # Create extra dependencies
        self.sequence_length = Property(0)
        self.add_dependency(self.sequence_length)
        self.sequence_length.add_child(self)

        # The current index of the sequence
        self.sequence_step = Property(0)
        self.add_dependency(self.sequence_step)
        self.sequence_step.add_child(self)

        # Stores all previous values
        self.previous_values = Property(
            lambda _ID=(): self.previous(_ID=_ID)[: self.sequence_step() - 1]
            if self.sequence_step(_ID=_ID)
            else []
        )
        self.add_dependency(self.previous_values)
        self.previous_values.add_child(self)
        self.previous_values.add_dependency(self.sequence_step)
        self.sequence_step.add_child(self.previous_values)

        # Stores the previous value
        self.previous_value = Property(
            lambda _ID=(): self.previous(_ID=_ID)[self.sequence_step() - 1]
            if self.previous(_ID=_ID)
            else None
        )
        self.add_dependency(self.previous_value)
        self.previous_value.add_child(self)
        self.previous_value.add_dependency(self.sequence_step)
        self.sequence_step.add_child(self.previous_value)

        # Creates an action for initializing the sequence
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
    """Iterates the dependencies of a feature and sets the value of their properties to the values in kwargs

    Parameters
    ----------
    X : features.Feature
        The feature whose dependencies are to be updated
    kwargs : dict
        The values to be set for the properties of the dependencies.
    """
    for dep in X.recurse_dependencies():
        if isinstance(dep, PropertyDict):
            for key, value in kwargs.items():
                if key in dep:
                    dep[key].set_value(value)