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

import typing


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

            return lambda replicate_index=None: dict(
                (key, val(replicate_index=replicate_index))
                for key, val in dict_of_actions.items()
            )

        if isinstance(sampling_rule, list):
            list_of_actions = [
                self.create_action(val, **dependencies) for val in sampling_rule
            ]

            return lambda replicate_index=None: [
                val(replicate_index=replicate_index) for val in list_of_actions
            ]

        if isinstance(sampling_rule, (tuple, np.ndarray)):
            return lambda replicate_index=None: sampling_rule

        if isiterable(sampling_rule):
            # If it's iterable, return the next value
            def action():
                try:
                    return next(sampling_rule)
                except StopIteration:
                    return self.previous()

            return action

        if isinstance(sampling_rule, slice):

            start = self.create_action(sampling_rule.start, **dependencies)
            stop = self.create_action(sampling_rule.stop, **dependencies)
            step = self.create_action(sampling_rule.step, **dependencies)

            return lambda replicate_index=None: slice(
                start(replicate_index=replicate_index),
                stop(replicate_index=replicate_index),
                step(replicate_index=replicate_index),
            )

        if callable(sampling_rule):

            knames = get_kwarg_names(sampling_rule)

            used_dependencies = dict(
                (key, dep) for key, dep in dependencies.items() if key in knames
            )

            for dep in used_dependencies.values():
                dep.add_child(self)
                self.add_dependency(dep)

            return lambda replicate_index=None: sampling_rule(
                **dict(
                    (key, dep(replicate_index=replicate_index))
                    for key, dep in used_dependencies.items()
                )
            )

        return lambda replicate_index=None: sampling_rule


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
                except AttributeError as e:
                    pass

        def action(replicate_index=None):
            return dict(
                (key, val(replicate_index=replicate_index))
                for key, val in dependencies.items()
            )

        super().__init__(action, **dependencies)

        for val in dependencies.values():
            val.add_child(self)
            self.add_dependency(val)


# class SequentialProperty(Property):
#     """Property that has multiple sequential values

#     Extends standard `Property` to resolve one value for each step
#     in a sequence of images. This is often used when creating movies.

#     Parameters
#     ----------
#     initializer : any
#         Sampling rule for the first step of the sequence.
#     sampling_rule : any
#         Sampling rule for each step after the first.

#     Attributes
#     ----------
#     initializer : any
#         Sampling rule for the first step of the sequence.
#     sampling_rule : any
#         Sampling rule for each step after the first.
#     has_updated_since_last_resolve : bool
#         Whether the property has been updated since the last resolve.

#     """

#     def __init__(self, sampling_rule, initializer=None):
#         super().__init__(sampling_rule)
#         if initializer is None:
#             self.initializer = sampling_rule
#         else:
#             self.initializer = initializer

#         self.sampling_rule = sampling_rule

#         # Deprecated
#         self.has_updated_since_last_resolve = False

#     def update(self, sequence_length=0, **kwargs):
#         """Updates current_value

#         For each step in the sequence, sample `self.sampling_rule`.
#         `self.initializer` is used for the first step. These rules
#         should output one value per step. Sampling rules
#         that are functions can optionally accept the following keyword
#         arguments:

#         * sequence_step : the current position in the sequence.
#         * sequence_length : the length of the sequence.
#         * previous_value : the value of the property at the previous
#           step in the sequence.
#         * previous_values : the value of the property at all previous
#           steps in the sequence.

#         Parameters
#         ----------
#         sequence_length : int, optional
#             length of the sequence

#         Returns
#         -------
#         self
#             returns self
#         """
#         my_id = id(self)
#         if (
#             features.UPDATE_LOCK.locked()
#             and my_id in features.UPDATE_MEMO["memoization"]
#         ):
#             return self

#         kwargs.update(features.UPDATE_MEMO["user_arguments"])

#         new_current_value = []

#         for step in range(sequence_length):
#             # Use initializer for first time step
#             ruleset = self.sampling_rule

#             # Elements inserted here can be passed to property functions
#             kwargs.update(
#                 sequence_step=step,
#                 sequence_length=sequence_length,
#                 previous_values=new_current_value,
#             )
#             if step == 0:
#                 kwargs.update(previous_value=self.sample(self.initializer, **kwargs))
#             else:
#                 kwargs.update(previous_value=new_current_value[-1])

#             next_value = self.sample(ruleset, **kwargs)

#             new_current_value.append(next_value)

#         self.current_value = new_current_value
#         features.UPDATE_MEMO["memoization"][my_id] = new_current_value
#         return self
