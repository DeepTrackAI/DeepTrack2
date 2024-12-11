"""Tools to manage the properties of a feature

"""

from typing import Any, Callable, Dict, List, Union

import numpy as np

from .utils import get_kwarg_names
from .backend.core import DeepTrackNode


class Property(DeepTrackNode):
    """Property of a feature in the DeepTrack2 framework.

    A `Property` defines a rule for sampling values used to evaluate features. 
    It supports various data types and structures, such as constants, 
    functions, lists, iterators, dictionaries, tuples, NumPy arrays, slices, 
    and DeepTrackNodes.

    The behavior of a `Property` depends on the type of the sampling rule:
    - Constant values (including tuples and NumPy arrays): Always returns the 
      same value.
    - Functions: Evaluates dynamically, potentially using other properties as 
      arguments.
    - Lists or dictionaries: Evaluates and samples each member individually.
    - Iterators: Returns the next value in the sequence, repeating the final 
      value indefinitely.
    - Slices: Samples the `start`, `stop`, and `step` values individually.
    - DeepTrackNodes (e.g., other properties or features): Uses the value 
      computed by the node.

    Dependencies between properties are tracked automatically, enabling 
    efficient recomputation when dependencies change.

    Parameters
    ----------
    sampling_rule : Any
        The rule for sampling values. Can be a constant, function, list, 
        dictionary, iterator, tuple, NumPy array, slice, or DeepTrackNode.
    **kwargs : dict
        Additional dependencies passed as named arguments. These dependencies 
        can be used as inputs to functions or other dynamic components of the 
        sampling rule.

    Methods
    -------
    create_action(sampling_rule, **dependencies)
        Creates an action that defines how the property is evaluated. The 
        behavior of the action depends on the type of `sampling_rule`.

    Examples
    --------
    Constant property:
    >>> const_prop = Property(42)
    >>> const_prop()  # Returns 42

    Dynamic property using a function:
    >>> dynamic_prop = Property(lambda x: x * 2, x=Property(5))
    >>> dynamic_prop()  # Returns 10

    Property with a dictionary rule:
    >>> dict_prop = Property({"a": Property(1), "b": lambda: 2})
    >>> dict_prop()  # Returns {"a": 1, "b": 2}

    Property with a generator:
    >>> gen = (i for i in range(3))
    >>> gen_prop = Property(gen)
    >>> gen_prop()  # Returns the next value from the generator
    >>> gen_prop()  # Returns the next value
    
    """

    def __init__(
        self,
        sampling_rule: Union[
            Callable[..., Any],
            List[Any],
            Dict[str, Any],
            tuple,
            np.ndarray,
            slice,
            DeepTrackNode,
            Any
        ],
        **kwargs: 'Property',
    ):
        """Initializes a Property object with a given sampling rule.

        Parameters
        ----------
        sampling_rule : Union[Callable[..., Any], List[Any], Dict[str, Any], 
                              tuple, np.ndarray, slice, Generator, 
                              DeepTrackNode, Any]
            The rule to sample values for the property.
        **kwargs : Property
            Additional named dependencies used in the sampling rule.
        
        """

        super().__init__()

        self.action = self.create_action(sampling_rule, **kwargs)

    def create_action(
        self,
        sampling_rule: Union[
            Callable[..., Any],
            List[Any],
            Dict[str, Any],
            tuple,
            np.ndarray,
            slice,
            DeepTrackNode,
            Any
        ],
        **dependencies: 'Property',
    ) -> Callable[..., Any]:
        """Creates an action defining how the property is evaluated.

        Parameters
        ----------
        sampling_rule : Union[Callable[..., Any], List[Any], Dict[str, Any], 
                              tuple, np.ndarray, slice, Generator, 
                              DeepTrackNode, Any]
            The rule to sample values for the property.
        **dependencies : Property
            Dependencies to be used in the sampling rule.

        Returns
        -------
        Callable[..., Any]
            A callable that defines the evaluation behavior of the property.

        """

        # DeepTrackNode (e.g., another property or feature).
        # Return the value sampled by the DeepTrackNode.
        if isinstance(sampling_rule, DeepTrackNode):
            sampling_rule.add_child(self)
            self.add_dependency(sampling_rule)
            return sampling_rule

        # Dictionary.
        # Return a dictionary with each each member sampled individually.
        if isinstance(sampling_rule, dict):
            dict_of_actions = dict(
                (key, self.create_action(value, **dependencies))
                for key, value in sampling_rule.items()
            )
            return lambda _ID=(): dict(
                (key, value(_ID=_ID)) for key, value in dict_of_actions.items()
            )

        # List.
        # Return a list with each each member sampled individually.
        if isinstance(sampling_rule, list):
            list_of_actions = [
                self.create_action(value, **dependencies)
                for value in sampling_rule
            ]
            return lambda _ID=(): [value(_ID=_ID) for value in list_of_actions]

        # Iterable.
        # Return the next value. The last value is returned indefinetely.
        if hasattr(sampling_rule, "__next__"):

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

        # Slice.
        # Sample individually the start, stop and step.
        if isinstance(sampling_rule, slice):

            start = self.create_action(sampling_rule.start, **dependencies)
            stop = self.create_action(sampling_rule.stop, **dependencies)
            step = self.create_action(sampling_rule.step, **dependencies)

            return lambda _ID=(): slice(
                start(_ID=_ID),
                stop(_ID=_ID),
                step(_ID=_ID),
            )

        # Function.
        # Return the result of the function. It accepts the names of other 
        # properties of the same feature as arguments.
        if callable(sampling_rule):

            knames = get_kwarg_names(sampling_rule)

            # Extract the arguments that are also properties
            used_dependencies = dict(
                (key, dependency) for key, dependency
                in dependencies.items() if key in knames
            )

            # Add the dependencies of the function as children.
            for dependency in used_dependencies.values():
                dependency.add_child(self)
                self.add_dependency(dependency)

            # Create the action.
            return lambda _ID=(): sampling_rule(
                **{key: dependency(_ID=_ID) for key, dependency 
                   in used_dependencies.items()},
                **({"_ID": _ID} if "_ID" in knames else {}),
            )

        # Constant, tuple or numpy array.
        # Return always the same constant value.
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
        self.action = self._action_override

    def _action_override(self, _ID=()):
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
            # SLOW
            return dict((key, val(_ID=_ID)) for key, val in self.items())

        super().__init__(action, **dependencies)

        for val in dependencies.values():
            val.add_child(self)
            self.add_dependency(val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

def propagate_data_to_dependencies(X, **kwargs):
    """Iterates the dependencies of a feature and sets the value of their properties to the values in kwargs.

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