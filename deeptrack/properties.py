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

Module Structure
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

Examples
--------
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
>>> seq_prop.sampling_rule = lambda _ID=(): seq_prop.sequence_index() + 1
>>> for step in range(seq_prop.sequence_length()):
...     seq_prop.sequence_index.store(step)
...     seq_prop.store(seq_prop.current())
...     print(seq_prop.data[()].current_value())

"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import numpy as np

from deeptrack.backend.core import DeepTrackNode
from deeptrack.utils import get_kwarg_names


__all__ = [
    "Property",
    "PropertyDict",
    "SequentialProperty",
]


if TYPE_CHECKING:
    import torch


class Property(DeepTrackNode):
    """Property of a feature in the DeepTrack2 framework.

    A `Property` defines a rule for sampling values used to evaluate features. 
    It supports various data types and structures, such as constants, 
    functions, lists, iterators, dictionaries, tuples, NumPy arrays, PyTorch
    tensors, slices, and `DeepTrackNode`s.

    The behavior of a `Property` depends on the type of the sampling rule:
    
    - **Constant values** (including tuples, NumPy arrays, and PyTorch
        tensors) always return the same value.
    - **Functions** are evaluated dynamically, potentially using other
        properties as arguments.
    - **Lists or dictionaries** evaluate and sample each member individually.
    - **Iterators** return the next value in the sequence, repeating the final
        value indefinitely.
    - **Slices** sample the `start`, `stop`, and `step` values individually.
    - **DeepTrackNode's** (e.g., other properties or features) use the value 
      computed by the node.

    Dependencies between properties are tracked automatically, enabling 
    efficient recomputation when dependencies change.

    Parameters
    ----------
    sampling_rule: Any
        The rule for sampling values. Can be a constant, function, list, 
        dictionary, iterator, tuple, NumPy array, torch Tensor, slice,
        or DeepTrackNode.
    **kwargs: dict[str, Property]
        Additional dependencies passed as named arguments. These dependencies 
        can be used as inputs to functions or other dynamic components of the 
        sampling rule.

    Methods
    -------
    create_action(sampling_rule: Any, **dependencies: dict[str, Property]) -> Callable[..., Any]
        Creates an action that defines how the property is evaluated. The 
        behavior of the action depends on the type of `sampling_rule`.

    Examples
    --------
    >>> import deeptrack as dt

    Constant properties are returned forever:

    >>> const_prop = dt.Property(42)  # Number
    >>> const_prop()  # Returns 42

    >>> const_prop = dt.Property([1, 2, 3])  # List
    >>> const_prop()  # Returns [1, 2, 3]

    >>> const_prop = dt.Property((1, 2, 3))  # Tuple
    >>> const_prop()  # Returns (1, 2, 3)

    >>> import numpy as np
    >>> 
    >>> const_prop = dt.Property(np.array([1, 2, 3]))  # NumPy array
    >>> const_prop()  # Returns array([1, 2, 3])

    >>> import torch
    >>> 
    >>> const_prop = dt.Property(torch.Tensor([1, 2, 3]))  # PyTorch tensor
    >>> const_prop()  # Returns tensor([1., 2., 3.])

    Dynamic property using functions, which can also depend on other
    properties:
    
    >>> dynamic_prop = dt.Property(lambda: np.random.rand())
    >>> dynamic_prop()  # Returns random value
    >>> dynamic_prop()  # Returns same random value
    >>> dynamic_prop.update()  # Updates the value
    >>> dynamic_prop()  # Returns different random value

    >>> const_prop = dt.Property(5)
    >>> dynamic_prop = dt.Property(lambda x: 2 * x, x=const_prop)
    >>> dynamic_prop()  # Returns 10

    >>> def func(x):
    ...     return 2 * x
    >>> 
    >>> const_prop = dt.Property(5)
    >>> dynamic_prop = dt.Property(func, x=const_prop)
    >>> dynamic_prop()  # Returns 10

    Slices can be constructed from dynamic or static components:

    >>> slice_prop = dt.Property(slice(1, lambda: 10, dt.Property(2)))
    >>> s = slice_prop()
    >>> s.start, s.stop, s.step  # Returns (1, 10, 2)

    Iterators return their next value each time, repeating the last
    indefinitely:

    >>> iter_prop = dt.Property(iter([1, 2, 3]))
    >>> iter_prop()  # Returns 1
    >>> iter_prop.update()
    >>> iter_prop()  # Returns 2
    >>> iter_prop.update()
    >>> iter_prop()  # Returns 3
    >>> iter_prop.update()
    >>> iter_prop()  # Returns 3 (last value repeats)

    Lists and dictionaries can contain properties, functions, or constants:

    >>> list_prop = dt.Property([
    ...     1,
    ...     lambda: 2,
    ...     dt.Property(3),
    ... ])
    >>> list_prop()  # Returns [1, 2, 3]

    >>> dict_prop = dt.Property({
    ...     "a": 1,
    ...     "b": lambda: 2,
    ...     "c": dt.Property(3),
    ... })
    >>> dict_prop()  # Returns {'a': 1, 'b': 2, 'c': 3}

    Property can wrap a DeepTrackNode, such as another feature node:

    >>> node = dt.DeepTrackNode(100)
    >>> node_prop = dt.Property(node)
    >>> node_prop()  # Returns 100

    >>> node = dt.DeepTrackNode(lambda _ID=(): np.random.rand())
    >>> node_prop = Property(node)
    >>> node_prop()  # Returns a random float in [0, 1]

    The ID mechanism allows parameterizing evaluation:

    >>> id_prop0 = dt.Property(lambda _ID: _ID)
    >>> id_prop0()           # Returns ()
    >>> id_prop0((1,))       # Returns ()
    >>> id_prop0((1, 2, 3))  # Returns ()

    >>> id_prop1 = dt.Property(lambda _ID: _ID)
    >>> id_prop1((1,))       # Returns (1,)
    >>> id_prop1((1, 2, 3))  # Returns (1,)

    >>> id_prop2 = dt.Property(lambda _ID: _ID)
    >>> id_prop2((1, 2, 3))  # Returns (1, 2, 3)

    Properties can be combined in complex nested structures:

    >>> P = dt.Property(
    ...     {
    ...         "constant": 42,
    ...         "list": [1, lambda: 2, dt.Property(3)],
    ...         "dict": {"a": dt.Property(1), "b": lambda: 2},
    ...         "function": lambda x, y: x * y,
    ...         "slice": slice(1, lambda: 10, dt.Property(2)),
    ...     },
    ...     x=dt.Property(5),
    ...     y=dt.Property(3),
    ... )
    >>> result = P()
    >>> result["constant"]         # Returns 42
    >>> result["list"]             # Returns [1, 2, 3]
    >>> result["dict"]             # Returns {'a': 1, 'b': 2}
    >>> result["function"]         # Returns 15
    >>> result["slice"].start      # Returns 1
    >>> result["slice"].stop       # Returns 10
    >>> result["slice"].step       # Returns 2

    """

    def __init__(
        self: Property,
        sampling_rule: (
            Callable[..., Any] |
            list[Any] |
            dict[str, Any] |
            tuple |
            np.ndarray |
            torch.Tensor |
            slice |
            DeepTrackNode |
            Any
        ),
        **dependencies: Property,
    ):
        """Initialize a `Property` object with a given sampling rule.

        Parameters
        ----------
        sampling_rule: Callable[..., Any] or list[Any] or dict[str, Any]
                       or tuple or np.ndarray or torch.Tensor or slice
                       or DeepTrackNode or Any
            The rule to sample values for the property.
        **dependencies: dict[str, Property]
            Additional named dependencies used in the sampling rule.
        
        """

        super().__init__()

        self.action = self.create_action(sampling_rule, **dependencies)

    def create_action(
        self: Property,
        sampling_rule: (
            Callable[..., Any] |
            list[Any] |
            dict[str, Any] |
            tuple |
            np.ndarray |
            torch.Tensor |
            slice |
            DeepTrackNode |
            Any
        ),
        **dependencies: Property,
    ) -> Callable[..., Any]:
        """Create an action defining how the property is evaluated.

        Parameters
        ----------
        sampling_rule: Callable[..., Any] or list[Any] or dict[str, Any]
                       or tuple or np.ndarray or torch.Tensor or slice
                       or DeepTrackNode or Any
            The rule to sample values for the property.
        **dependencies: dict[str, Property]
            Dependencies to be used in the sampling rule.

        Returns
        -------
        Callable[..., Any]
            A callable that defines the evaluation behavior of the property.

        """

        # DeepTrackNode (e.g., another property or feature)
        # Return the value sampled by the DeepTrackNode.
        if isinstance(sampling_rule, DeepTrackNode):
            sampling_rule.add_child(self)
            # self.add_dependency(sampling_rule)  # Already done by add_child.
            return sampling_rule

        # Dictionary
        # Return a dictionary with each each member sampled individually.
        if isinstance(sampling_rule, dict):
            dict_of_actions = dict(
                (key, self.create_action(value, **dependencies))
                for key, value in sampling_rule.items()
            )
            return lambda _ID=(): dict(
                (key, value(_ID=_ID)) for key, value in dict_of_actions.items()
            )

        # List
        # Return a list with each each member sampled individually.
        if isinstance(sampling_rule, list):
            list_of_actions = [
                self.create_action(value, **dependencies)
                for value in sampling_rule
            ]
            return lambda _ID=(): [value(_ID=_ID) for value in list_of_actions]

        # Iterable
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

        # Slice
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

        # Function
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

        # Constant, tuple, numpy array, or torch Tensor
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
    **kwargs: dict[str, Any]
        Key-value pairs used to initialize the dictionary, where values are 
        either directly used to create `Property` instances or are dependent 
        on other `Property` values.

    Methods
    -------
    __init__(**kwargs: dict[str, Any])
        Initializes the `PropertyDict`, resolving `Property` dependencies.
    __getitem__(key: str) -> Any
        Retrieves a value from the dictionary using a key.

    Examples
    --------
    Initialize a `PropertyDict` with different types of properties:

    >>> import deeptrack as dt
    >>> import numpy as np

    >>> prop_dict = dt.PropertyDict(
    ...     constant=42,
    ...     dependent=lambda constant: constant + 10,
    ...     random=lambda: np.random.rand(),
    ... )

    Access the properties:

    >>> prop_dict["constant"]()  # Returns 42
    >>> prop_dict["dependent"]()  # Returns 52
    >>> prop_dict["random"]()  # Returns random number
    
    """

    def __init__(
        self: PropertyDict,
        **kwargs: Any,
    ):
        """Initialize a PropertyDict with properties and dependencies.

        Iteratively converts the input dictionary's values into `Property` 
        instances while resolving dependencies between the properties.

        It resolves dependencies between the properties iteratively.
        
        An `action` is created to evaluate and return the dictionary with 
        sampled values.

        Parameters
        ----------
        **kwargs: dict[str, Any]
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

        def action(_ID: tuple[int, ...] = ()) -> dict[str, Any]:
            """Evaluate and return the dictionary with sampled Property values.

            Parameters
            ----------
            _ID: tuple[int, ...], optional
                A unique identifier for sampling properties.

            Returns
            -------
            dict[str, Any]
                A dictionary where each value is sampled from its respective 
                `Property`.
            
            """

            return dict((key, value(_ID=_ID)) for key, value in self.items())

        super().__init__(action, **dependencies)

        for value in dependencies.values():
            value.add_child(self)
            # self.add_dependency(value)  # Already executed by add_child.

    def __getitem__(
        self: PropertyDict,
        key: str,
    ) -> Any:
        """Retrieve a value from the dictionary.

        Overrides the default `__getitem__` to ensure dictionary functionality.

        Parameters
        ----------
        key: str
            The key to retrieve the value for.

        Returns
        -------
        Any
            The value associated with the specified key.

        Notes
        -----
        This method directly calls the `__getitem__` method of the built-in 
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
    SequentialProperty lets the user encapsulate feature sampling rules and
    iterator logic in a single object to evaluate them sequentially.
    
    The `SequentialProperty` class extends the standard `Property` to handle 
    scenarios where the property’s value evolves over discrete steps, such as 
    frames in a video, time-series data, or any sequential process. At each 
    step, it selects whether to use the `initialization` function (step = 0) or 
    the `current` function (steps >= 1). It also keeps track of all previously 
    generated values, allowing to refer back to them if needed.


    Parameters
    ----------
    initial_sampling_rule: Any, optional
        A sampling rule for the first step of the sequence (step=0). 
        Can be any value or callable that is acceptable to `Property`. 
        If not provided, the initial value is `None`.
        
    current_value: Any, optional
        The sampling rule (value or callable) for steps > 0. Defaults to None.
    sequence_length: int, optional
        The length of the sequence.
    sequence_index: int, optional
        The current index of the sequence. 
        
    **kwargs: dict[str, Property]
        Additional dependencies that might be required if `initialization` 
        is a callable. These dependencies are injected when evaluating
        `initialization`.

    Attributes
    ----------
    sequence_length: Property
        A `Property` holding the total number of steps in the sequence. 
        Initialized to 0 by default.
    sequence_index: Property
        A `Property` holding the index of the current step (starting at 0).
    previous_values: Property
        A `Property` returning all previously stored values up to, but not
        including, the current value and the previous value.
    previous_value: Property
        A `Property` returning the most recently stored value, or `None` 
        if there is no history yet.
    initial_sampling_rule: Callable[..., Any], optional
        A function to compute the value at step=0. If `None`, the property 
        returns `None` at the first step.
    sample: Callable[..., Any]
        Computes the value at steps >= 1 with the given sampling rule.
        By default, it returns `None`.
    action: Callable[..., Any]
        Overrides the default `Property.action` to select between 
        `initialization` (if `sequence_index` is 0) or `current` (otherwise).

    Methods
    -------
    _action_override(_ID: tuple[int, ...]) -> Any
        Internal logic to pick which function (`initialization` or `current`) 
        to call based on the `sequence_index`.
    store(value: Any, _ID: tuple[int, ...] = ()) -> None
        Store a newly computed `value` in the property’s internal list of 
        previously generated values.
    current_value(_ID: tuple[int, ...] = ()) -> Any
        Retrieve the value associated with the current step index.
    __call__(_ID: tuple[int, ...] = ()) -> Any
        Evaluate the property at the current step, returning either the 
        initialization (if step=0) or current value (if step>0).
    set_sequence_length(self, value, ID) -> None:
        Stores the value for the length of the sequence,
        analagous to SequentialProperty.sequence_length.store()        
    set_current_index(self, value, ID) -> None:
        Stores the value for the current step of the sequence,
        analagous to SequentialProperty.current_step.store()
        
    

    Examples
    --------
    >>> import deeptrack as dt
    >>> import numpy as np

    >>> seq_prop = dt.SequentialProperty()
    >>> seq_prop.sequence_length.store(5)
    >>> seq_prop.sampling_rule = lambda _ID=(): seq_prop.sequence_index() + 1
    >>> for step in range(seq_prop.sequence_length()):
    ...     seq_prop.sequence_index.store(step)
    ...     current_value = seq_prop.sampling_rule()
    ...     seq_prop.store(current_value)
    ...     print(seq_prop.data[()].current_value())
    [1]
    [1, 2]
    [1, 2, 3]
    [1, 2, 3, 4]
    [1, 2, 3, 4, 5]
    
    """
    
    sequence_length: Property
    sequence_index: Property
    previous_values: Property
    previous_value: Property
    initial_sampling_rule: Optional[Callable[..., Any]]
    sample: Optional[Callable[..., Any]]
    action: Callable[..., Any]

    def __init__(
        self: SequentialProperty,
        initial_sampling_rule: Optional[Any] = None,
        sampling_rule: Optional[Any] = None,
        sequence_length: Optional[int] = None,
        sequence_index: Optional[int] = None,
        **kwargs: Dict[str, Property],
    ) -> None:
        """Create a SequentialProperty with optional initialization sampling rule.
        
        Parameters
        ----------
        initial_sampling_rule : Any, optional
            The sampling rule (value or callable) for step=0. Defaults to None.
        sampling_rule: Any, optional
            The sampling rule (value or callable) for the current step.
            Defaults to None.
        sequence_length: int, optional
            The length of the sequence. 
        sequence_index: int, optional
            The current index of the sequence. 
        **kwargs: dict[str, Property]
            Additional named dependencies for `initialization` and `current`.
        
        """

        # Set sampling_rule=None to the base constructor.
        # It overrides action below with _action_override().
        super().__init__(sampling_rule=None)


        # 1) Initialize sequence length.
        if isinstance(sequence_length, int):
            self.sequence_length = Property(sequence_length)
        else:   
            self.sequence_length = Property(0)
        self.sequence_length.add_child(self)
        # self.add_dependency(self.sequence_length)  # Done by add_child.

        # 2) Initialize sequence index.
        if isinstance(sequence_index, int):
            self.sequence_index = Property(sequence_index)
        else:
            self.sequence_index = Property(0)
        self.sequence_index.add_child(self)
        # self.add_dependency(self.sequence_index)  # Done by add_child.

        # 3) Store all previous values if sequence step > 0.
        self.previous_values = Property(
            lambda _ID=(): self.previous(_ID=_ID)[: self.sequence_index() - 1]
                           if self.sequence_index(_ID=_ID)
                           else []
        )
        self.previous_values.add_child(self)
        
        # self.add_dependency(self.previous_values)  # Done by add_child
        self.sequence_index.add_child(self.previous_values)
        
        # self.previous_values.add_dependency(self.sequence_index)  # Done
        
        # 4) Store the previous value.
        self.previous_value = Property(
            lambda _ID=(): self.previous(_ID=_ID)[self.sequence_index() - 1]
                           if self.previous(_ID=_ID)
                           else None
        )
        self.previous_value.add_child(self)
        # self.add_dependency(self.previous_value)  # Done by add_child
        self.sequence_index.add_child(self.previous_value)
        # self.previous_value.add_dependency(self.sequence_index)  # Done

        # 5) Create an action for initializing the sequence.
        if initial_sampling_rule is not None:
            self.initial_sampling_rule = self.create_action(initial_sampling_rule, **kwargs)
        else:
            self.initial_sampling_rule = None

        # 6) Define a default current function for steps >= 1.
        if sampling_rule is not None:
            self.sample = self.create_action(sampling_rule, **kwargs)
        else:
            self.sample = lambda _ID=(): None

        # 7) Override the default action with our custom logic.
        self.action = self._action_override

    def _action_override(
        self: SequentialProperty,
        _ID: tuple[int, ...] = (),
    ) -> Any:
        """Decide which function to call based on the current step.

        For step=0, call `self.initial_sampling_rule`. Otherwise, call `self.sampling_rule`.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            A unique identifier that differentiates parallel evaluations.

        Returns
        -------
        Any
            The result of the `self.initial_sampling_rule` function (if step == 0)
            or the result of the `self.sampling_rule` function (if step > 0).
        
        """

        if self.sequence_index(_ID=_ID) == 0:
            if self.initial_sampling_rule:
                return self.initial_sampling_rule(_ID=_ID)
            return None

        return self.sample(_ID=_ID)

    def store(
        self: SequentialProperty,
        value: Any,
        _ID: tuple[int, ...] = (),
    ) -> None:
        """Append value to the internal list of previously generated values.

        It retrieves the existing list of values for this _ID. If this _ID has 
        never been used, it starts an empty list.

        Parameters
        ----------
        value: Any
            The value to store, e.g., the output from calling `self()`.
        _ID: tuple[int, ...], optional
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

    def current_value(
        self: SequentialProperty,
        _ID: tuple[int, ...] = (),
    ) -> Any:
        """Retrieve the value corresponding to the current sequence step.

        It expects that each step's value has been stored. If no value has been 
        stored for this step, it thorws an IndexError.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            A unique identifier for separate parallel evaluations.

        Returns
        -------
        Any
            The value stored at the index = `self.sequence_index(_ID=_ID)`.

        Raises
        ------
        IndexError
            If no value has been stored for this step, it thorws an IndexError.

        """

        return super().current_value(_ID=_ID)[self.sequence_index(_ID=_ID)]

    def __call__(
        self: SequentialProperty,
        _ID: tuple[int, ...] = (),
    ) -> Any:
        """Evaluate the property at the current sequence step.
        
        It returns the result of `self.initial_sampling_rule` (if step == 0) or the
        result of `self.sampling_rule` (if step > 0).

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            A unique identifier for parallel evaluations.

        Returns
        -------
        Any
            The computed value for this step.

        """

        return super().__call__(_ID=_ID)

    def set_sequence_length(
        self: SequentialProperty,
        value: Any,
        _ID: tuple[int, ...] = (),
    ) -> None:
        """Sets the `sequence_length` attribute of a sequence to be resolved.

        It supports dependencies if `value` is a `Property`.

        Parameters
        ----------
        value: Any
            The value to store in `self.sequence_length`.
        _ID: tuple[int, ...], optional
            A unique identifier that allows the property to keep separate 
            histories for different parallel evaluations.

        """

        if isinstance(value, Property):  # For dependencies
            self.sequence_length = Property(lambda _ID: value(_ID))
            self.sequence_length.add_dependency(value)
        else:
            self.sequence_length = Property(value, _ID=_ID)

    def set_current_index(
        self: SequentialProperty,
        value: Any,
        _ID: tuple[int, ...] = (),
    ) -> None:
        """Sets the `sequence_index` attribute of a sequence to be resolved.

        It supports dependencies if `value` is a `Property`.

        Parameters
        ----------
        value: Any
            The value to store in `sequence_index`.
        _ID: tuple[int, ...], optional
            A unique identifier that allows the property to keep separate 
            histories for different parallel evaluations.
    
        """

        if isinstance(value, Property):  # For dependencies
            self.sequence_index = Property(lambda _ID: value(_ID))
            self.sequence_index.add_dependency(value)
        else:
            self.sequence_index = Property(value, _ID=_ID)
