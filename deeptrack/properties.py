"""Tools to manage feature properties in DeepTrack2.

This module provides classes for managing, sampling, and evaluating properties 
of features within the DeepTrack2 framework. It offers flexibility in defining 
and handling properties with various data types, dependencies, and sampling 
rules.

Main Features
-------------
- **Property Management**

    Classes like `Property` and `PropertyDict` provide tools 
    for defining, sampling, and evaluating properties. These properties can be 
    constants, functions, lists, dictionaries, iterators, or slices, allowing 
    for dynamic and context-dependent evaluations.

**Sequential Sampling** 
    
    The `SequentialProperty` class enables the creation of properties that 
    evolve over a sequence, useful for applications like creating dynamic 
    features in videos or time-series data.

Model Structure
-----------------
Property Classes:

- `Property`: Property of a feature.

    Defines a single property of a feature, supporting various data types and 
    dynamic evaluations.
    
- `SequentialProperty`: Property for sequential sampling.

    Extends `Property` to support sequential sampling across steps.

- `PropertyDict`: Property dictionary.

    A dictionary of properties with utilities for dependency management and 
    sampling.

Example
-------
Create and use a constant property:

>>> import deeptrack as dt

>>> const_prop = dt.Property(42)
>>> const_prop()  # Returns 42

Define a dynamic property dependent on another:

>>> const_prop = dt.Property(5)
>>> dynamic_prop = dt.Property(lambda x: x * 2, x=const_prop)
>>> dynamic_prop()  # Returns 10

Create a dictionary of properties:

>>> prop_dict = dt.PropertyDict(
...     constant=42,
...     dependent=lambda constant: constant + 10,
...     random=lambda dependent: np.random.rand() + dependent,
... )
>>> print(prop_dict["constant"]())  # Returns 42
>>> print(prop_dict["dependent"]())  # Returns 52
>>> print(prop_dict["random"]())

Handle sequential properties:

>>> seq_prop = dt.SequentialProperty()
>>> seq_prop.sequence_length.store(5)
>>> seq_prop.current = lambda _ID=(): seq_prop.sequence_step() + 1
>>> for step in range(seq_prop.sequence_length()):
...     seq_prop.sequence_step.store(step)
...     seq_prop.store(seq_prop.current())
...     print(seq_prop.data[()].current_value())

"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    
    - **Constant values** (including tuples and NumPy arrays): Always returns the 
      same value.
    - **Functions**: Evaluates dynamically, potentially using other properties as 
      arguments.
    - **Lists or dictionaries**: Evaluates and samples each member individually.
    - **Iterators**: Returns the next value in the sequence, repeating the final 
      value indefinitely.
    - **Slices**: Samples the `start`, `stop`, and `step` values individually.
    - **DeepTrackNodes** (e.g., other properties or features): Uses the value 
      computed by the node.

    Dependencies between properties are tracked automatically, enabling 
    efficient recomputation when dependencies change.

    Parameters
    ----------
    sampling_rule : Any
        The rule for sampling values. Can be a constant, function, list, 
        dictionary, iterator, tuple, NumPy array, slice, or DeepTrackNode.
    **kwargs : Dict['Property']
        Additional dependencies passed as named arguments. These dependencies 
        can be used as inputs to functions or other dynamic components of the 
        sampling rule.

    Methods
    -------
    create_action(sampling_rule: Any, **dependencies: Dict[str, Property]) -> Callable[..., Any]
        Creates an action that defines how the property is evaluated. The 
        behavior of the action depends on the type of `sampling_rule`.

    Examples
    --------
    Constant property:
    
    >>> import deeptrack as dt
    
    >>> const_prop = dt.Property(42)
    >>> const_prop()  # Returns 42

    Dynamic property using a function:
    
    >>> const_prop = dt.Property(5)
    >>> dynamic_prop = dt.Property(lambda x: x * 2, x=const_prop)
    >>> dynamic_prop()  # Returns 10

    Property with a dictionary rule:
    
    >>> dict_prop = dt.Property({"a": Property(1), "b": lambda: 2})
    >>> dict_prop()  # Returns {"a": 1, "b": 2}

    Property with an iterable:
    
    >>> gen = (i for i in range(3))
    >>> gen_prop = dt.Property(gen)
    >>> gen_prop()  # Returns the next value from the generator
    >>> gen_prop.update()
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
        sampling_rule : Callable[..., Any] or List[Any] or Dict[str, Any] or 
                        tuple or np.ndarray or slice or DeepTrackNode or Any
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
        **dependencies: Dict[str, 'Property'],
    ) -> Callable[..., Any]:
        """Creates an action defining how the property is evaluated.

        Parameters
        ----------
        sampling_rule : Union[Callable[..., Any], List[Any], Dict[str, Any], 
                              tuple, np.ndarray, slice, Generator, 
                              DeepTrackNode, Any]
            The rule to sample values for the property.
        **dependencies : Dict[str, Property]
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
            # self.add_dependency(sampling_rule)  # Already done by add_child.
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
                        pass  # Yield the final value infinitely.
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

            # Extract the arguments that are also properties.
            used_dependencies = dict(
                (key, dependency) for key, dependency
                in dependencies.items() if key in knames
            )

            # Add the dependencies of the function as children.
            for dependency in used_dependencies.values():
                dependency.add_child(self)
                # self.add_dependency(dependency)  # Already done by add_child.

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
    """Dictionary with Property elements.

    A `PropertyDict` is a specialized dictionary where values are instances of 
    `Property`. It provides additional utility functions to update, sample, 
    reset, and retrieve properties. This is particularly useful for managing 
    feature-specific properties in a structured manner.

    Parameters
    ----------
    **kwargs : Dict[str, Any]
        Key-value pairs used to initialize the dictionary, where values are 
        either directly used to create `Property` instances or are dependent 
        on other `Property` values.

    Methods
    -------
    __init__(**kwargs: Dict[str, Any])
        Initializes the `PropertyDict`, resolving `Property` dependencies.
    __getitem__(key: str) -> Any
        Retrieves a value from the dictionary using a key.

    Examples
    --------
    Initialize a `PropertyDict` with different types of properties:

    >>> import deeptrack as dt

    >>> prop_dict = dt.PropertyDict(
    ...     constant=42,
    ...     dependent=lambda constant: constant + 10,
    ...     random=lambda: np.random.rand(),
    ... )

    Access the properties:

    >>> print(prop_dict["constant"]())  # Returns 42
    >>> print(prop_dict["dependent"]())  # Returns 52
    >>> print(prop_dict["random"]())
    
    """

    def __init__(self, **kwargs: Dict[str, Any]):
        """Initialize a PropertyDict with properties and dependencies.

        Iteratively converts the input dictionary's values into `Property` 
        instances while resolving dependencies between the properties.

        It resolves dependencies between the properties iteratively.
        
        An `action` is created to evaluate and return the dictionary with 
        sampled values.

        Parameters
        ----------
        **kwargs : Dict[str, Any]
            Key-value pairs used to initialize the dictionary. Values can be 
            constants, functions, or other `Property`-compatible types.

        """

        dependencies = {}  # To store the resolved Property instances.

        while kwargs:
            # Multiple passes over the data until everything that can be 
            # resolved is resolved.
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
            Dict[str, Any]
                A dictionary where each value is sampled from its respective 
                `Property`.
            
            """

            return dict((key, value(_ID=_ID)) for key, value in self.items())

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


class SequentialProperty(Property):
    """Property that yields different values for sequential steps.

    The `SequentialProperty` extends the standard `Property` to handle 
    scenarios where the property’s value evolves over discrete steps, such as 
    frames in a video, time-series data, or any sequential process. At each 
    step, it selects whether to use the `initialization` function (step = 0) or 
    the `current` function (steps >= 1). It also keeps track of all previously 
    generated values, allowing to refer back to them if needed.

    Parameters
    ----------
    initialization : Any, optional
        A sampling rule for the first step of the sequence (step=0). 
        Can be any value or callable that is acceptable to `Property`. 
        If not provided, the initial value is `None`.
    **kwargs : Dict[str, Property]
        Additional dependencies that might be required if `initialization` 
        is a callable. These dependencies are injected when evaluating
        `initialization`.

    Attributes
    ----------
    sequence_length : Property
        A `Property` holding the total number of steps in the sequence. 
        Initialized to 0 by default.
    sequence_step : Property
        A `Property` holding the index of the current step (starting at 0).
    previous_values : Property
        A `Property` returning all previously stored values up to, but not
        including, the current value and the previous value.
    previous_value : Property
        A `Property` returning the most recently stored value, or `None` 
        if there is no history yet.
    initialization : Callable[..., Any], optional
        A function to compute the value at step=0. If `None`, the property 
        returns `None` at the first step.
    current : Callable[..., Any]
        A function to compute the value at steps >= 1. By default,  it returns 
        `None`.
    action : Callable[..., Any]
        Overrides the default `Property.action` to select between 
        `initialization` (if `sequence_step` is 0) or `current` (otherwise).

    Methods
    -------
    _action_override(_ID: Tuple[int, ...]) -> Any
        Internal logic to pick which function (`initialization` or `current`) 
        to call based on the `sequence_step`.
    store(value: Any, _ID: Tuple[int, ...] = ()) -> None
        Store a newly computed `value` in the property’s internal list of 
        previously generated values.
    current_value(_ID: Tuple[int, ...] = ()) -> Any
        Retrieve the value associated with the current step index.
    __call__(_ID: Tuple[int, ...] = ()) -> Any
        Evaluate the property at the current step, returning either the 
        initialization (if step=0) or current value (if step>0).

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    >>> seq_prop = dt.SequentialProperty()
    >>> seq_prop.sequence_length.store(5)
    >>> seq_prop.current = lambda _ID=(): seq_prop.sequence_step() + 1
    >>> for step in range(seq_prop.sequence_length()):
    ...     seq_prop.sequence_step.store(step)
    ...     current_value = seq_prop.current()
    ...     seq_prop.store(current_value)
    ...     print(seq_prop.data[()].current_value())
    [1]
    [1, 2]
    [1, 2, 3]
    [1, 2, 3, 4]
    [1, 2, 3, 4, 5]
    
    """


    # Attributes.
    sequence_length: Property
    sequence_step: Property
    previous_values: Property
    previous_value: Property
    initialization: Optional[Callable[..., Any]]
    current: Callable[..., Any]
    action: Callable[..., Any]

    def __init__(
        self,
        initialization: Optional[Any] = None,
        **kwargs: Dict[str, 'Property'],
    ):
        """Create a SequentialProperty with optional initialization.
        
        Parameters
        ----------
        initialization : Any, optional
            The sampling rule (value or callable) for step=0. Defaults to None.
        **kwargs : Dict[str, Property]
            Additional named dependencies for `initialization`.
        
        """

        # Set sampling_rule=None to the base constructor, as it overrides 
        # action below with _action_override.
        super().__init__(sampling_rule=None)

        # 1) Initialize sequence length to 0.
        self.sequence_length = Property(0)
        self.sequence_length.add_child(self)
        # self.add_dependency(self.sequence_length)  # Done by add_child.

        # 2) Current index of the sequence (0).
        self.sequence_step = Property(0)
        self.sequence_step.add_child(self)
        # self.add_dependency(self.sequence_step)  # Done by add_child.

        # 3) Store all previous values.
        self.previous_values = Property(
            lambda _ID=(): self.previous(_ID=_ID)[: self.sequence_step() - 1]
                           if self.sequence_step(_ID=_ID)
                           else []
        )
        self.previous_values.add_child(self)
        # self.add_dependency(self.previous_values)  # Done by add_child.
        self.sequence_step.add_child(self.previous_values)
        # self.previous_values.add_dependency(self.sequence_step)  # Done.

        # 4) Store the previous value.
        self.previous_value = Property(
            lambda _ID=(): self.previous(_ID=_ID)[self.sequence_step() - 1]
                           if self.previous(_ID=_ID)
                           else None
        )
        self.previous_value.add_child(self)
        # self.add_dependency(self.previous_value)  # Done by add_child.
        self.sequence_step.add_child(self.previous_value)
        # self.previous_value.add_dependency(self.sequence_step)  # Done.

        # 5) Create an action for initializing the sequence.
        if initialization is not None:
            self.initialization = self.create_action(initialization, **kwargs)
        else:
            self.initialization = None

        # 6) Define a default current function for steps >= 1.
        self.current = lambda _ID=(): None

        # 7) Override the default action with our custom logic.
        self.action = self._action_override

    def _action_override(self, _ID: Tuple[int, ...] = ()) -> Any:
        """Decide which function to call based on the current step.

        For step=0, call `initialization`. Otherwise, call `self.current`.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            A unique identifier that differentiates parallel evaluations.

        Returns
        -------
        Any
            The result of the `initialization` (step=0) or the `current` 
            function (step>0).
        
        """

        if self.sequence_step(_ID=_ID) == 0:
            return (self.initialization(_ID=_ID) 
                    if self.initialization else None)
        else:
            return self.current(_ID=_ID)

    def store(self, value: Any, _ID: Tuple[int, ...] = ()) -> None:
        """Append value to the internal list of previously generated values.

        It retrieves the existing list of values for this _ID. If this _ID has 
        never been used, it starts an empty list

        Parameters
        ----------
        value : Any
            The value to store, e.g., the output from calling `self()`.
        _ID : Tuple[int, ...], optional
            A unique identifier that allows the property to keep separate 
            histories for different parallel evaluations.

        Raises
        ------
        KeyError
            If no existing data for this _ID, it initializes an empty list.

        """

        try:
            current_data = self.data[_ID].current_value()
        except KeyError:
            current_data = []

        super().store(current_data + [value], _ID=_ID)

    def current_value(self, _ID: Tuple[int, ...] = ()) -> Any:
        """Retrieve the value corresponding to the current step.

        It expects that each step's value has been stored. If no value has been 
        stored for this step, it thorws an IndexError.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            A unique identifier for separate parallel evaluations.

        Returns
        -------
        Any
            The value stored at the index = `self.sequence_step(_ID=_ID)`.
        
        """

        return super().current_value(_ID=_ID)[self.sequence_step(_ID=_ID)]

    def __call__(self, _ID: Tuple[int, ...] = ()) -> Any:
        """Evaluate the property at the current step.
        
        It returns either the initialization (if step=0) or the result of 
        `self.current`.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            A unique identifier for parallel evaluations.

        Returns
        -------
        Any
            The computed value for this step.
        
        """

        return super().__call__(_ID=_ID)
