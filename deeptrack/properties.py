"""Tools to manage feature properties in DeepTrack2.

This package provides classes and functions for managing, sampling, and 
evaluating properties of features within the DeepTrack2 framework. It offers 
flexibility in defining and handling properties with various data types, 
dependencies, and sampling rules.

Main Features
-------------
Property Management: Classes like `Property` and `PropertyDict` provide tools 
for defining, sampling, and evaluating properties. These properties can be 
constants, functions, lists, dictionaries, iterators, or slices, allowing for 
dynamic and context-dependent evaluations.

Sequential Sampling: The `SequentialProperty` class enables the creation of 
properties that evolve over a sequence, useful for applications like creating 
dynamic features in videos or time-series data.

Package Structure
-----------------
Property Classes:
- `Property`: Defines a single property of a feature, supporting various data 
              types and dynamic evaluations.
- `SequentialProperty`: Extends `Property` to support sequential sampling 
                        across steps.
- `PropertyDict`: A dictionary of properties with utilities for dependency 
                  management and sampling.

Example
-------
Create and use a constant property:

>>> const_prop = Property(42)
>>> const_prop()  # Returns 42

Define a dynamic property dependent on another:

>>> dynamic_prop = Property(lambda x: x * 2, x=Property(5))
>>> dynamic_prop()  # Returns 10

Create a dictionary of properties:

>>> prop_dict = PropertyDict(
...     constant=42,
...     dependent=lambda constant: constant + 10,
...     random=lambda: np.random.rand(),
... )
>>> print(prop_dict["constant"]())  # Returns 42
>>> print(prop_dict["dependent"]())  # Returns 52

Handle sequential properties:

>>> seq_prop = SequentialProperty(initialization=lambda: np.random.rand())
>>> seq_prop.sequence_length.store(5)
>>> for i in range(5):
...     seq_prop.sequence_step.store(i)
...     print(seq_prop())  # Returns different values for each step

"""

from typing import Any, Callable, Dict, List, Tuple, Union

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


class PropertyDict(DeepTrackNode, dict):
    """Dictionary with Property elements

    A `PropertyDict` is a specialized dictionary where values are instances of 
    `Property`. It provides additional utility functions to update, sample, 
    reset, and retrieve properties. This is particularly useful for managing 
    feature-specific properties in a structured manner.

    Parameters
    ----------
    **kwargs : dict
        Key-value pairs used to initialize the dictionary, where values are 
        either directly used to create `Property` instances or are dependent 
        on other `Property` values.

    Methods
    -------
    __init__(**kwargs: Any)
        Initializes the `PropertyDict`, resolving `Property` dependencies.
    __getitem__(key: str) -> Any
        Retrieves a value from the dictionary using a key.

    Examples
    --------
    Initialize a `PropertyDict` with different types of properties:

    >>> prop_dict = PropertyDict(
    ...     constant=42,
    ...     dependent=lambda constant: constant + 10,
    ...     random=lambda: np.random.rand(),
    ... )

    Access constant and dependent properties:

    >>> print(prop_dict["constant"]())
    42
    
    >>> print(prop_dict["dependent"]())
    52
    
    """

    def __init__(self, **kwargs: Any):
        """Initialize a PropertyDict with properties and dependencies.

        Iteratively converts the input dictionary's values into `Property` 
        instances while resolving dependencies between the properties.

        It resolves dependencies between the properties iteratively.
        
        An `action` is created to evaluate and return the dictionary with 
        sampled values.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs used to initialize the dictionary. Values can be 
            constants, functions, or other `Property`-compatible types.

        """

        dependencies = {}  # To store the resolved Property instances.
        while kwargs:
            for key, value in list(kwargs.items()):
                try:
                    # Create a Property instance for the key, 
                    # resolving dependencies.
                    dependencies[key] = Property(value,
                                                 **{**dependencies, **kwargs})
                    # Remove the key from the input dictionary once resolved.
                    kwargs.pop(key)
                except AttributeError:
                    # Catch unresolved dependencies and continue iterating.
                    pass

        def action(_ID: Tuple[int, ...] = ()) -> Dict[str, Any]:
            """Evaluate and return the dictionary with sampled Property values.

            Parameters
            ----------
            _ID : Tuple[int, ...], optional
                A unique identifier for sampling properties.

            Returns
            -------
            dict
                A dictionary where each value is sampled from its respective 
                `Property`.
            
            """

            return dict((key, value(_ID=_ID)) for key, value in self.items())  #TODO SLOW - why??

        super().__init__(action, **dependencies)

        for value in dependencies.values():
            value.add_child(self)
            # self.add_dependency(value)  # Already executed by add_child.

    def __getitem__(self, key: str) -> Any:
        """Retrieve a value from the dictionary.

        Overrides the default `__getitem__` to ensure dictionary functionality.

        Parameters
        ----------
        key : str
            The key to retrieve the value for.

        Returns
        -------
        Any
            The value associated with the specified key.

        Notes
        -----
        This method explicitly calls the `__getitem__` method of the built-in 
        `dict` class. This ensures that the standard dictionary behavior is 
        used to retrieve values, bypassing any custom logic in `PropertyDict` 
        that might otherwise cause infinite recursion or unexpected results.
        
        """

        # Directly invoke the built-in dictionary method to retrieve the value.
        # This avoids potential recursion by bypassing any overridden behavior
        # in the current class or its parents.
        return dict.__getitem__(self, key)


class SequentialProperty(Property):  #TODO comment.
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
