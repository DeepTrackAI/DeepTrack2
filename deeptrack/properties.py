"""Tools to manage feature properties in DeepTrack2.

This module provides classes for managing, sampling, and evaluating properties
of features within the DeepTrack2 framework. It offers flexibility in defining
and handling properties with various data types, dependencies, and sampling
rules.

Key Features
------------
- **Property Management**

    The `Property` and `PropertyDict` classes  provide tools for defining,
    sampling, and evaluating properties. These properties can be constants,
    functions, lists, dictionaries, iterators, or slices, allowing for dynamic
    and context-dependent evaluations.

- **Sequential Sampling** 
    
    The `SequentialProperty` class enables the creation of properties that
    evolve over a sequence, useful for applications like creating dynamic
    features in videos or time-series data.

Module Structure
----------------
Classes:

- `Property`: Property of a feature.

    Defines a single property of a feature, supporting various data types and
    dynamic evaluations.
    
- `PropertyDict`: Property dictionary.

    A dictionary of properties with utilities for dependency management and
    sampling.

- `SequentialProperty`: Property for sequential sampling.

    Extends `Property` to support sequential sampling across steps.

Examples
--------
>>> import deeptrack as dt

Create and use a constant property:

>>> const_prop = dt.Property(42)
>>> const_prop()
42

Define a dynamic property dependent on another:

>>> const_prop = dt.Property(5)
>>> dynamic_prop = dt.Property(lambda x: x * 2, x=const_prop)
>>> dynamic_prop()
10

Create a dictionary of properties:

>>> import numpy as np
>>>
>>> prop_dict = dt.PropertyDict(
...     constant=42,
...     dependent=lambda constant: constant + 10,
...     random=lambda dependent: np.random.rand() + dependent,
... )
>>> prop_dict["constant"]()
42

>>> prop_dict["dependent"]()
52

>>> prop_dict["random"]()
52.35065943710633

Handle sequential properties:

>>> seq_prop = dt.SequentialProperty(
...     sampling_rule=lambda: np.random.randint(10, 20),
...     sequence_length=5,
... )
>>> for step in range(seq_prop.sequence_length()):
...     seq_prop()
...     seq_prop.next_step()
...     print(f"Sequence at step {step}: {seq_prop.sequence()}")
Sequence at step 0: [19]
Sequence at step 1: [19, 10]
Sequence at step 2: [19, 10, 11]
Sequence at step 3: [19, 10, 11, 14]
Sequence at step 4: [19, 10, 11, 14, 12]

"""


from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

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
    tensors, slices, and `DeepTrackNode` objects.

    The behavior of a `Property` depends on the type of the sampling rule:
    
    - **Constant values** (including tuples, NumPy arrays, and PyTorch
        tensors) always return the same value.
    - **Functions** are evaluated dynamically, potentially using other
        properties as arguments.
    - **Lists, dictionaries, or tuples ** evaluate and sample each member
        individually.
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
        dictionary, iterator, tuple, NumPy array, PyTorch tensor, slice,
        or `DeepTrackNode`.
    node_name: str | None
        The name of this node. Defaults to None.
    **dependencies: Property
        Additional dependencies passed as named arguments. These dependencies 
        can be used as inputs to functions or other dynamic components of the 
        sampling rule.

    Methods
    -------
    `create_action(sampling_rule, **dependencies) -> Callable[..., Any]`
        Creates an action that defines how the property is evaluated. The
        behavior of the action depends on the type of `sampling_rule`.

    Examples
    --------
    >>> import deeptrack as dt

    Constant properties are returned forever:

    >>> const_prop = dt.Property(42)  # Number
    >>> const_prop()
    42

    >>> const_prop = dt.Property([1, 2, 3])  # List
    >>> const_prop()
    [1, 2, 3]

    >>> const_prop = dt.Property((1, 2, 3))  # Tuple
    >>> const_prop()
    (1, 2, 3)

    >>> import numpy as np
    >>> 
    >>> const_prop = dt.Property(np.array([1, 2, 3]))  # NumPy array
    >>> const_prop()
    array([1, 2, 3])

    >>> import torch
    >>> 
    >>> const_prop = dt.Property(torch.Tensor([1, 2, 3]))  # PyTorch tensor
    >>> const_prop()
    tensor([1., 2., 3.])

    Dynamic property typically use functions and can also depend on other
    properties:
    
    >>> dynamic_prop = dt.Property(lambda: np.random.rand())
    >>> dynamic_prop()  # Returns random value
    0.37700241766131415
    >>> dynamic_prop()  # Returns same random value
    0.37700241766131415
    >>> dynamic_prop.update()  # Updates the value
    >>> dynamic_prop()  # Returns different random value
    0.5862725216547282
    >>> dynamic_prop.new()  # Returns different random value
    0.36122033451938484

    >>> const_prop = dt.Property(5)
    >>> dynamic_prop = dt.Property(lambda x: 2 * x, x=const_prop)
    >>> dynamic_prop()
    10

    >>> def func(x):
    ...     return 2 * x
    >>> 
    >>> const_prop = dt.Property(5)
    >>> dynamic_prop = dt.Property(func, x=const_prop)
    >>> dynamic_prop()
    10

    Slices can be constructed from dynamic or static components:

    >>> slice_prop = dt.Property(slice(1, lambda: 10, dt.Property(2)))
    >>> s = slice_prop()
    >>> s.start, s.stop, s.step
    (1, 10, 2)

    Iterators return their next value each time, repeating the last
    indefinitely:

    >>> iter_prop = dt.Property(iter([1, 2, 3]))
    >>> iter_prop()
    1
    >>> iter_prop.new()  # equivalent to iter_prop.update()()
    2
    >>> iter_prop.new()
    3
    >>> iter_prop.new()  # Last value repeats
    3

    Lists, dictionaries, and tuples can contain properties, functions, or
    constants:

    >>> list_prop = dt.Property([
    ...     1,
    ...     lambda: 2,
    ...     dt.Property(3),
    ... ])
    >>> list_prop()
    [1, 2, 3]

    >>> dict_prop = dt.Property({
    ...     "a": 1,
    ...     "b": lambda: 2,
    ...     "c": dt.Property(3),
    ... })
    >>> dict_prop()
    {'a': 1, 'b': 2, 'c': 3}

    >>> tuple_prop = dt.Property((
    ...     1,
    ...     lambda: 2,
    ...     dt.Property(3),
    ... ))
    >>> tuple_prop()
    (1, 2, 3)

    Property can wrap a `DeepTrackNode`, such as another feature node:

    >>> node = dt.DeepTrackNode(100)
    >>> node_prop = dt.Property(node)
    >>> node_prop()
    100

    >>> node = dt.DeepTrackNode(lambda _ID=(): np.random.rand())
    >>> node_prop = dt.Property(node)
    >>> node_prop()
    0.5065650298607408

    The ID mechanism allows parameterizing evaluation:

    >>> id_prop0 = dt.Property(lambda _ID: _ID)
    >>> id_prop0()
    ()
    >>> id_prop0((1,))
    ()
    >>> id_prop0((1, 2, 3))
    ()

    >>> id_prop1 = dt.Property(lambda _ID: _ID)
    >>> id_prop1((1,))
    (1,)
    >>> id_prop1((1, 2, 3))
    (1,)

    >>> id_prop2 = dt.Property(lambda _ID: _ID)
    >>> id_prop2((1, 2, 3))
    (1, 2, 3)

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
    >>> result["constant"]
    42
    >>> result["list"]
    [1, 2, 3]
    >>> result["dict"]
    {'a': 1, 'b': 2}
    >>> result["function"]
    15
    >>> result["slice"].start
    1
    >>> result["slice"].stop
    10
    >>> result["slice"].step
    2

    """

    def __init__(
        self: Property,
        sampling_rule: (
            Callable[..., Any] |
            list[Any] |
            dict[Any, Any] |
            tuple[Any, ...] |
            np.ndarray |
            torch.Tensor |
            slice |
            DeepTrackNode |
            Any
        ),
        node_name: str | None = None,
        **dependencies: Property,
    ) -> None:
        """Initialize a `Property` object with a given sampling rule.

        Parameters
        ----------
        sampling_rule: Any
            The rule to sample values for the property. It can be essentially
            anything, most often:
            Callable[..., Any] or list[Any] or dict[Any, Any] or tuple
            or NumPy array or PyTorch tensor or slice or DeepTrackNode or Any
        node_name: str or None
            The name of this node. Defaults to None.
        **dependencies: Property
            Additional named dependencies used in the sampling rule.
        
        """

        super().__init__()

        self.action = self.create_action(sampling_rule, **dependencies)

        self.node_name = node_name

    def create_action(
        self: Property,
        sampling_rule: (
            Callable[..., Any] |
            list[Any] |
            dict[Any, Any] |
            tuple[Any, ...] |
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
        sampling_rule: Any
            The rule to sample values for the property. It can be essentially
            anything, most often:
            Callable[..., Any] or list[Any] or dict[Any, Any] or tuple
            or NumPy array or PyTorch tensor or slice or DeepTrackNode or Any
        **dependencies: Property
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
            return sampling_rule

        # Dictionary
        # Return a dictionary with each member sampled individually.
        if isinstance(sampling_rule, dict):
            dict_of_actions = dict(
                (key, self.create_action(rule, **dependencies))
                for key, rule in sampling_rule.items()
            )
            return lambda _ID=(): dict(
                (key, action(_ID=_ID))
                for key, action in dict_of_actions.items()
            )

        # List
        # Return a list with each member sampled individually.
        if isinstance(sampling_rule, list):
            list_of_actions = [
                self.create_action(rule, **dependencies)
                for rule in sampling_rule
            ]
            return lambda _ID=(): [
                action(_ID=_ID)
                for action in list_of_actions
            ]

        # Tuple
        # Return a tuple with each member sampled individually.
        if isinstance(sampling_rule, tuple):
            tuple_of_actions = tuple(
                self.create_action(rule, **dependencies)
                for rule in sampling_rule
            )
            return lambda _ID=(): tuple(
                action(_ID=_ID)
                for action in tuple_of_actions
            )

        # Iterable
        # Return the next value. The last value is returned indefinitely.
        if hasattr(sampling_rule, "__next__"):

            def wrapped_iterator():
                next_value = None
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
        # Sample start, stop, and step individually.
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
                (key, dependency)
                for key, dependency
                in dependencies.items()
                if key in knames
            )

            # Add the dependencies of the function as children.
            for dependency in used_dependencies.values():
                dependency.add_child(self)

            # Create the action.
            return lambda _ID=(): sampling_rule(
                **{key: dependency(_ID=_ID)
                   for key, dependency
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
    node_name: str | None, optional
        The name of this node. Defaults to `None`.
    **kwargs: Any
        Key-value pairs used to initialize the dictionary, where values are
        either directly used to create `Property` instances or are dependent
        on other `Property` values.

    Methods
    -------
    `__getitem__(key) -> Any`
        Retrieves a value from the dictionary using a key.

    Examples
    --------
    >>> import deeptrack as dt

    Initialize a `PropertyDict` with different types of properties:

    >>> import numpy as np
    >>>
    >>> prop_dict = dt.PropertyDict(
    ...     constant=42,
    ...     dependent=lambda constant: constant + 10,
    ...     random=lambda: np.random.rand(),
    ... )

    Access the properties:

    >>> prop_dict["constant"]()
    42

    >>> prop_dict["dependent"]()
    52

    >>> prop_dict["random"]()
    0.33112452108057056
    
    """

    def __init__(
        self: PropertyDict,
        node_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a PropertyDict with properties and dependencies.

        Iteratively converts the input dictionary's values into `Property`
        instances while iteratively resolving dependencies between the
        properties.
        
        An `action` is created to evaluate and return the dictionary with 
        sampled values.

        Parameters
        ----------
        node_name: str or None
            The name of this node. Defaults to `None`.
        **kwargs: Any
            Key-value pairs used to initialize the dictionary. Values can be 
            constants, functions, or other `Property`-compatible types.

        """

        dependencies: dict[str, Property] = {}  # Store resolved properties
        unresolved = dict(kwargs)

        while unresolved:
            # Multiple passes over the data until everything that can be
            # resolved is resolved.
            progressed = False  # Track whether any key resolved in this pass

            for key, rule in list(unresolved.items()):
                try:
                    # Create a Property instance for the key,
                    # resolving dependencies.
                    dependencies[key] = Property(
                        rule,
                        node_name=key,
                        **{**dependencies, **unresolved},
                    )
                    # Remove the key from the input dictionary once resolved.
                    unresolved.pop(key)

                    progressed = True  # Progress has been made

                except AttributeError:
                    # Catch unresolved dependencies and continue iterating.
                    continue

            if not progressed:
                raise ValueError(
                    "Could not resolve PropertyDict dependencies for keys: "
                    f"{', '.join(unresolved.keys())}."
                )

        def action(
            _ID: tuple[int, ...] = (),
        ) -> dict[str, Any]:
            """Evaluate and return the dictionary with sampled Property values.

            Parameters
            ----------
            _ID: tuple[int, ...], optional
                A unique identifier for sampling properties. Defaults to `()`.

            Returns
            -------
            dict[str, Any]
                A dictionary where each value is sampled from its respective
                `Property`.
            
            """

            return dict((key, prop(_ID=_ID)) for key, prop in self.items())

        super().__init__(action, **dependencies)

        self.node_name = node_name

        for prop in dependencies.values():
            prop.add_child(self)

    def __getitem__(
        self: PropertyDict,
        key: str,
    ) -> Any:
        """Retrieve a value from the dictionary.

        Overrides the default `.__getitem__()` to ensure dictionary
        functionality.

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
        This method directly calls the `.__getitem__()` method of the built-in
        `dict` class. This ensures that the standard dictionary behavior is
        used to retrieve values, bypassing any custom logic in `PropertyDict`
        that might otherwise cause infinite recursion or unexpected results.
        
        """

        # Directly invoke the built-in dictionary method to retrieve the value.
        # This avoids potential recursion by bypassing any overridden behavior
        # in the current class or its parents.
        return dict.__getitem__(self, key)


class SequentialProperty(Property):
    """Property that yields different values across sequential steps.

    A `SequentialProperty` encapsulates sampling rules and step management in a
    single object for sequential evaluation.

    This class extends `Property` to support scenarios where a property value
    evolves over discrete steps, such as frames in a video, time-series data,
    or other sequential processes. At each step, it selects whether to use the
    `initial_sampling_rule` (when step == 0 and it is provided) or the
    `sampling_rule` (otherwise). It also keeps track of previously generated
    values, allowing sampling rules to depend on history.

    Parameters
    ----------
    node_name: str | None, optional
        The name of this node. Defaults to `None`.
    initial_sampling_rule: Any, optional
        A sampling rule for the first step (step == 0). Can be any value or
        callable accepted by `Property`. Defaults to `None`.
    sampling_rule: Any, optional
        The sampling rule (value or callable) for steps > 0, and also for
        step == 0 when `initial_sampling_rule` is `None`. Defaults to `None`.
    sequence_length: int, optional
        The length of the sequence. Defaults to `None`.
    **kwargs: Property
        Additional dependencies injected when evaluating callable sampling
        rules.

    Attributes
    ----------
    sequence_length: Property
        A `Property` holding the total number of steps (`int`) in the sequence.
        Initialized to 0 by default.
    sequence_index: Property
        A `Property` holding the index (`int`) of the current step (starting
        at 0).
    previous_values: Property
        A `Property` returning all stored values strictly before the previous
        value (`list[Any]`).
    previous_value: Property
        A `Property` returning the most recently stored value (`Any`), or
        `None` if no values have been stored yet.
    initial_sampling_rule: Callable[..., Any] | None
        A function (or constant wrapped as an action) used to compute the value
        at step 0. If `None`, the property falls back to `sampling_rule` at
        step 0.
    sample: Callable[..., Any]
        The action used to compute the value at steps > 0 (and at step 0 if
        `initial_sampling_rule` is `None`). If no `sampling_rule` is provided,
        it returns `None`.
    action: Callable[..., Any]
        Overrides the default `Property.action` to select between
        `initial_sampling_rule` (when step is 0) and `sample` (otherwise).

    Methods
    -------
    `_action_override(_ID) -> Any`
        Select the appropriate sampling rule based on `sequence_index`.
    `sequence(_ID) -> list[Any]`
        Return the stored sequence for `_ID` without recomputing.
    `next_step(_ID) -> bool`
        Advance the sequence index by one step (if possible).
    `store(value, _ID) -> None`
        Append a newly computed value to the stored sequence for `_ID`.
    `current_value(_ID) -> Any`
        Return the stored value at the current step index.

    Examples
    --------
    To illustrate the use of `SequentialProperty`, we will implement a
    one-dimensional Brownian walker.

    >>> import deeptrack as dt

    Define the `SequentialProperty`:

    >>> import numpy as np
    >>>
    >>> seq_prop = dt.SequentialProperty(
    ...     initial_sampling_rule=0,  # Sampling rule for first time step
    ...     sampling_rule=(  # Sampl. rule for subsequent steps
    ...         lambda previous_value: previous_value + np.random.randn()
    ...     ),
    ...     sequence_length=10,  # Number of steps
    ... )

    Iteratively calculate the sequence:

    >>> for step in range(seq_prop.sequence_length()):
    ...     seq_prop()
    ...     seq_prop.next_step()  # Returns False at the final step

    Print all values of the sequence:

    >>> seq_prop.sequence()
    [0,
    -0.38200070551587934,
    0.4107493780458869,
    0.4168147820083061,
    -0.37943277485427523,
    -0.24658839362797394,
    0.6200008820895946,
    0.7763449126000742,
    1.9552313612982135,
    1.8016703270391572]

    """

    sequence_length: Property  # int
    sequence_index: Property  # int
    previous_values: Property  # list[Any]
    previous_value: Property  # Any
    initial_sampling_rule: Callable[..., Any] | None
    sample: Callable[..., Any]
    action: Callable[..., Any]

    def __init__(
        self: SequentialProperty,
        node_name: str | None = None,
        initial_sampling_rule: Any = None,
        sampling_rule: Any = None,
        sequence_length: int | None = None,
        **kwargs: Property,
    ) -> None:
        """Create a SequentialProperty.
        
        Parameters
        ----------
        node_name: str or None, optional
            The name of this node. Defaults to `None`.
        initial_sampling_rule: Any, optional
            The sampling rule (value or callable) for step == 0. If `None`,
            evaluation at step 0 falls back to `sampling_rule`.
            Defaults to `None`.
        sampling_rule: Any, optional
            The sampling rule (value or callable) for steps > 0, and also for
            step == 0 when `initial_sampling_rule` is `None`.
            Defaults to `None`.
        sequence_length: int, optional
            The length of the sequence. Defaults to `None`.
        **kwargs: Property
            Additional named dependencies for callable sampling rules.
        
        """

        # Set sampling_rule=None to the base constructor.
        # It overrides action below with _action_override().
        super().__init__(sampling_rule=None, node_name=node_name)

        # 1) Initialize sequence length.
        if isinstance(sequence_length, int):
            self.sequence_length = Property(
                sequence_length,
                node_name="sequence_length",
            )
        else:
            self.sequence_length = Property(0, node_name="sequence_length")
        self.sequence_length.add_child(self)

        # 2) Initialize sequence index.
        # Invariant: 0 <= sequence_index < sequence_length for valid sequence.
        self.sequence_index = Property(0, node_name="sequence_index")
        self.sequence_index.add_child(self)

        # 3) Store all previous values if sequence index > 0.
        self.previous_values = Property(
            lambda _ID=(): (
                self.sequence(_ID=_ID)[: self.sequence_index(_ID=_ID) - 1]
                if self.sequence_index(_ID=_ID) > 0
                else []
            ),
            node_name="previous_values",
        )
        self.previous_values.add_child(self)
        self.sequence_index.add_child(self.previous_values)

        # 4) Store the previous value.
        self.previous_value = Property(
            lambda _ID=(): (
                self.sequence(_ID=_ID)[self.sequence_index(_ID=_ID) - 1]
                if self.sequence_index(_ID=_ID) > 0
                else None
            ),
            node_name="previous_value",
        )
        self.previous_value.add_child(self)
        self.sequence_index.add_child(self.previous_value)

        # 5) Create an action for initializing the sequence.
        if initial_sampling_rule is not None:
            self.initial_sampling_rule = self.create_action(
                initial_sampling_rule,
                **kwargs,
            )
        else:
            self.initial_sampling_rule = None

        # 6) Define a default current function for steps >= 1.
        if sampling_rule is not None:
            self.sample = self.create_action(
                sampling_rule,
                sequence_index=self.sequence_index,
                sequence_length=self.sequence_length,
                previous_values=self.previous_values,
                previous_value=self.previous_value,
                **kwargs,
            )
        else:
            self.sample = lambda _ID=(): None

        # 7) Override the default action with our custom logic.
        self.action = self._action_override

    def _action_override(
        self: SequentialProperty,
        _ID: tuple[int, ...] = (),
    ) -> Any:
        """Select the appropriate sampling rule for the current step.

        At step 0, this calls `initial_sampling_rule` if it is not `None`.
        Otherwise, it calls `sample`.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            A unique identifier that differentiates parallel evaluations.

        Returns
        -------
        Any
            The sampled value for the current step.
        
        """

        if (
            self.sequence_index(_ID=_ID) == 0
            and self.initial_sampling_rule is not None
        ):
            return self.initial_sampling_rule(_ID=_ID)

        return self.sample(_ID=_ID)

    def store(
        self: SequentialProperty,
        value: Any,
        _ID: tuple[int, ...] = (),
    ) -> None:
        """Append a value to the stored sequence for _ID.

        Appends `value` to the stored sequence for `_ID`. If no values have
        been stored yet for `_ID`, it starts a new list.

        Parameters
        ----------
        value: Any
            The value to store, e.g., the output from calling `self()`.
        _ID: tuple[int, ...], optional
            A unique identifier that allows the property to keep separate 
            histories for different parallel evaluations.

        """

        current_data = self.sequence(_ID=_ID)
        super().store(current_data + [value], _ID=_ID)

    def current_value(
        self: SequentialProperty,
        _ID: tuple[int, ...] = (),
    ) -> Any:
        """Return the stored value at the current step index.

        It expects that each step's value has been stored. If no value has been
        stored for this step, it throws an IndexError.

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
            If no value has been stored for this step, it throws an IndexError.

        """

        sequence = self.sequence(_ID=_ID)
        index = self.sequence_index(_ID=_ID)

        if index >= len(sequence):
            raise IndexError(
                "No stored value for current step: index="
                f"{index}, stored_values={len(sequence)}."
            )

        return sequence[index]

    def sequence(self, _ID: tuple[int, ...] = ()) -> list[Any]:
        """Retrieve the stored sequence for _ID without recomputing.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            The ID for which to retrieve the previous value.

        Returns
        -------
        list[Any]
            The list of stored values for this `_ID`. Returns an empty list if
            no values have been stored yet.

        """

        if self.data.valid_index(_ID) and _ID in self.data.keys():
            return self.data[_ID].current_value()

        return []

    # Invariant:
    # For a sequence of length L = sequence_length(_ID),
    # the valid range of sequence_index(_ID) is:
    #
    #     0 <= sequence_index < L
    #
    # Each index corresponds to one stored value in the sequence.
    # Attempting to advance beyond L - 1 returns False.

    def next_step(
        self: SequentialProperty,
        _ID: tuple[int, ...] = (),
    ) -> bool:
        """Advance the sequence index by one step.

        This method increments `sequence_index` by one for the given `_ID` if
        the next index remains strictly less than `sequence_length`. It also
        invalidates cached properties that depend on the sequence index to
        ensure correct recomputation on subsequent access. If the sequence is
        already at its final step, the index is not changed.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            A unique identifier that allows the property to keep separate
            sequence states for different parallel evaluations.

        Returns
        -------
        bool
            True if the index was advanced, False if already at the final step.

        """

        current_index = self.sequence_index(_ID=_ID)
        sequence_length = self.sequence_length(_ID=_ID)

        if current_index + 1 >= sequence_length:
            return False

        self.sequence_index.store(current_index + 1, _ID=_ID)

        # Ensures updates when action is executed again
        self.previous_value.invalidate(_ID=_ID)
        self.previous_values.invalidate(_ID=_ID)

        return True
