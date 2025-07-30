"""Core data structures for DeepTrack2.

This module defines the foundational data structures used throughout
DeepTrack2 for constructing, managing, and evaluating computational graphs
with flexible data storage and dependency management.

Key Features
------------
- **Hierarchical Data Management**

    Provides validated, hierarchical data containers (`DeepTrackDataObject`
    and `DeepTrackDataDict`) for storing data and managing complex, nested
    data structures. Supports dependency tracking and flexible indexing.

- **Computation Graphs with Lazy Evaluation**

    Implements the `DeepTrackNode` class, the core abstraction for nodes in
    a computational graph. Supports lazy evaluation, caching, dependency
    tracking, and operator overloading for intuitive composition of complex
    computational pipelines.

- **Citation Support**

    Provides citation metadata to ensure proper academic attribution for work
    built on DeepTrack2.

Module Structure
----------------
Classes:

- `DeepTrackDataObject`: Basic container for data with validation status.

    Simple data container that stores data and tracks its validity
    (valid/invalid).

- `DeepTrackDataDict`: Hierarchical dictionary for multiple data objects.

    Stores multiple `DeepTrackDataObject` instances indexed by tuples of
    integers, enabling the creation of flexible, nested data hierarchies.

- `DeepTrackNode`: Node in a computation graph with operator overloading.

    Represents a node in a computation graph, capable of storing and
    computing values based on dependencies, with full support for lazy
    evaluation, dependency tracking, and operator overloading.

Functions:

- `_equivalent(a, b)`

      def _equivalent(a: Any, b: Any) -> bool

    Determines whether two objects should be considered equivalent,
    according to DeepTrack2's internal rules (identity, empty lists, etc).

- `_create_node_with_operator(op, a, b)`

      def _create_node_with_operator(
          op: Callable,
          a: Any,
          b: Any,
      ) -> DeepTrackNode

    Internal helper to create a new computation node by applying a
    specified operator to two operands, establishing correct graph
    relationships and supporting operator overloading.

Attributes:

- `CITATION_MIDTVEDT2021QUANTITATIVE`: str

    BibTeX citation for the original DeepTrack2 publication.

Examples
--------
>>> import deeptrack as dt

Create a simple computational pipeline using DeepTrack2 nodes:

>>> parent = dt.DeepTrackNode()
>>> child = dt.DeepTrackNode(lambda: 2 * parent())
>>> parent.add_child(child)
>>> parent.store(5)
>>> child()  # Compute child
10

Operator overloading for computation nodes:

>>> a = dt.DeepTrackNode(lambda: 3)
>>> b = dt.DeepTrackNode(lambda: 4)
>>> sum_node = a + b
>>> sum_node()
7

Create and use a hierarchical data dictionary:

>>> data_dict = dt.DeepTrackDataDict()
>>> data_dict.create_index((0, 1))
>>> data_dict[(0, 1)].store("Example data")
>>> data_dict[(0, 1)].current_value()
'Example data'

Validate and invalidate a data object:

>>> data_obj = dt.DeepTrackDataObject()
>>> data_obj.is_valid()
False

>>> data_obj.store(42)
>>> data_obj.is_valid()
True

>>> data_obj.invalidate()
>>> data_obj.is_valid()
False

"""


from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView
import operator  # Operator overloading for computation nodes.
from weakref import WeakSet  # Manages relationships between nodes without
                             # creating circular dependencies.
from typing import Any, Callable, Iterator

from deeptrack.utils import get_kwarg_names


__all__ = [
    "DeepTrackDataDict",
    "DeepTrackDataObject",
    "DeepTrackNode",
]


CITATION_MIDTVEDT2021QUANTITATIVE = """
@article{Midtvet2021Quantitative,
    author  = {Midtvedt, Benjamin and Helgadottir, Saga and Argun, Aykut and 
               Pineda, Jesús and Midtvedt, Daniel and Volpe, Giovanni},
    title   = {Quantitative digital microscopy with deep learning},
    journal = {Applied Physics Reviews},
    volume  = {8},
    number  = {1},
    pages   = {011310},
    year    = {2021},
    doi     = {10.1063/5.0034891}
}
"""


class DeepTrackDataObject:
    """Basic data container for DeepTrack2.

    `DeepTrackDataObject` is a simple data container to store some data and 
    track its validity.

    Attributes
    ----------
    _data: Any
        The stored data. Defaults to `None`.
    _valid: bool
        Flag indicating whether the stored data is valid. Defaults to `False`.

    Methods
    -------
    `store(data) -> None`
        Store data in the container and mark it as valid.
    `current_value() -> Any`
        Return the currently stored data.
    `is_valid() -> bool`
        Return whether the stored data is valid.
    `invalidate() -> None`
        Mark the data as invalid.
    `validate() -> None`
        Mark the data as valid.
    `__repr__() -> str`
        Return the string representation of the object.

    Example
    -------
    >>> import deeptrack as dt

    Create a `DeepTrackDataObject`:
    >>> data_obj = dt.DeepTrackDataObject()
    >>> data_obj

    Store a value in this container:
    >>> data_obj.store(42)
    >>> data_obj
    DeepTrackDataObject(data=42, valid=True)

    Access the currently stored value:
    >>> data_obj.current_value()
    42

    Check if the stored data is valid:
    >>> data_obj.is_valid()
    True

    Invalidate the stored data:
    >>> data_obj.invalidate()
    >>> data_obj
    DeepTrackDataObject(data=42, valid=False)

    >>> data_obj.is_valid()
    False

    Validate the data to restore its valid status:
    >>> data_obj.validate()
    >>> data_obj
    DeepTrackDataObject(data=42, valid=True)

    >>> data_obj.is_valid()
    True

    """

    _data: Any
    _valid: bool

    def __init__(self: DeepTrackDataObject):
        """Initialize the container without data.

        Initializes `_data` to `None` and `_valid` to `False`.

        """

        self._data = None
        self._valid = False

    def store(
        self: DeepTrackDataObject,
        data: Any,
    ) -> None:
        """Store data and mark it as valid.

        Parameters
        ----------
        data: Any
            The data to be stored in the container.

        """

        self._data = data
        self._valid = True

    def current_value(self: DeepTrackDataObject) -> Any:
        """Retrieve the stored data.

        Returns
        -------
        Any
            The data stored in the container.

        """

        return self._data

    def is_valid(self: DeepTrackDataObject) -> bool:
        """Return whether the stored data is valid.

        Returns
        -------
        bool
            `True` if the data is valid, `False` otherwise.
        
        """

        return self._valid

    def invalidate(self: DeepTrackDataObject) -> None:
        """Mark the stored data as invalid."""

        self._valid = False

    def validate(self: DeepTrackDataObject) -> None:
        """Mark the stored data as valid."""

        self._valid = True

    def __repr__(self: DeepTrackDataObject) -> str:
        """Return the string representation of the object.

        Provides a concise representation of the data object, including the
        stored data and its validity flag. It is useful for debugging and
        logging purposes.

        Returns
        -------
        str
            A string in the format:
            "DeepTrackDataObject(data=<data>, valid=<valid>)".

        """

        return (
            f"{self.__class__.__name__}(data={self._data!r}, "
            f"valid={self._valid})"
        )


class DeepTrackDataDict:
    """Store multiple data objects indexed by tuples of integers (_ID).

    `DeepTrackDataDict` can store multiple `DeepTrackDataObject` instances, 
    each associated with a unique tuple of integers (its `_ID`).

    **Use of _IDs**

    The default `_ID` is an empty tuple, `_ID = ()`.

    Once the first entry is created, all `_ID`s must match the set key length.

    When retrieving the data associated to an `_ID`:
    -   If an `_ID` longer than the set key length is requested, it is trimmed. 
    -   If an `_ID` shorter than the set key length is requested, a dictionary
        slice containing all matching entries is returned.

    NOTE: The `_ID`s are specifically used in the `Repeat` feature to allow it
    to return different values without changing the input.

    Attributes
    ----------
    keylength: int or None
        The length of the `_ID`s set when the first entry is created.
        If `None`, no entries have been created, and any `_ID` length is valid.
    dict: dict[tuple[int, ...], DeepTrackDataObject] or {}
        Read-only property exposing the internal dictionary of stored data,
        `_dict`. This is a dictionary mapping tuples of integers (`_ID`s) to
        `DeepTrackDataObject` instances.

    Methods
    -------
    `create_index(_ID) -> None`
        Create an entry for the given `_ID` if it does not exist.
    `invalidate() -> None`
        Mark all stored data objects as invalid.
    `validate() -> None`
        Mark all stored data objects as valid.
    `valid_index(_ID) -> bool`
        Check if the given _ID is valid for the current configuration.
    `__getitem__(_ID) -> DeepTrackDataObject or dict[_ID, DeepTrackDataObject]`
        Retrieve data associated with the `_ID`. Can return a
        `DeepTrackDataObject` or a dict of `DeepTrackDataObject`s if `_ID` is
        shorter than `keylength`.
    `__contains__(_ID) -> bool`
        Check whether the given `_ID` exists in the dictionary.
    `__len__() -> int`
        Return the number of stored entries.
    `__iter__() -> Iterator`
        Iterate over the keys of the dictionary.
    `items() -> ItemsView[tuple[int, ...], DeepTrackDataObject]`
        Return a view of the dictionary’s (key, value) pairs.
    `keys() -> KeysView[tuple[int, ...]]`
        Return a view of the dictionary’s keys.
    `values() -> ValuesView[DeepTrackDataObject]`
        Return a view of the dictionary’s values.
    `__repr__() -> str`
        Return a string representation of the data dictionary.

    Example
    -------
    >>> import deeptrack as dt

    Create a structure to store multiple, indexed instances of data:
    >>> data_dict = dt.DeepTrackDataDict()
    >>> data_dict
    DeepTrackDataDict(0 entries, keylength=None)

    Create the entries:    
    >>> data_dict.create_index((0, 0))
    >>> data_dict.create_index((0, 1))
    >>> data_dict.create_index((1, 0))
    >>> data_dict.create_index((1, 1))
    data_dict
    DeepTrackDataDict(4 entries, keylength=2)

    Store the values associated with each `_ID`:
    >>> data_dict[(0, 0)].store("Data at (0, 0)")
    >>> data_dict[(0, 1)].store("Data at (0, 1)")
    >>> data_dict[(1, 0)].store("Data at (1, 0)")
    >>> data_dict[(1, 1)].store("Data at (1, 1)")
    >>> data_dict
    DeepTrackDataDict(4 entries, keylength=2)

    Retrieve values based on their `_ID`s:
    >>> data_dict[(0, 0)]
    DeepTrackDataObject(data='Data at (0, 0)', valid=True)

    >>> data_dict[(0, 0)].current_value()
    'Data at (0, 0)'

    >>> data_dict[(1, 1)]
    DeepTrackDataObject(data='Data at (1, 1)', valid=True)

    >>> data_dict[(1, 1)].current_value()
    'Data at (1, 1)'

    If requesting a shorter `_ID`, it returns all matching nested entries:
    >>> data_dict[(0,)]
    {(0, 0): DeepTrackDataObject(data='Data at (0, 0)', valid=True),
    (0, 1): DeepTrackDataObject(data='Data at (0, 1)', valid=True)}
 
    Validate and invalidate all entries at once:
    >>> data_dict.invalidate()
    >>> data_dict[(0, 0)].is_valid()
    False

    >>> data_dict[(1, 1)].is_valid()
    False

    >>> data_dict.validate()
    >>> data_dict[(0, 0)].is_valid()
    True

    >>> data_dict[(1, 1)].is_valid()
    True

    Invalidate and validate a single entry:
    >>> data_dict[(0, 1)].invalidate()
    >>> data_dict[(0, 1)].is_valid()
    False

    >>> data_dict[(0, 1)].validate()
    >>> data_dict[(0, 1)].is_valid()
    True

    Check if a given `_ID` exists:
    >>> (1, 0) in data_dict
    True

    >>> (2, 2) in data_dict
    False

    Iterate over all entries:
    >>> for key, value in data_dict.items():
    ...     print(key, value.current_value())
    (0, 0) DeepTrackDataObject(data='Data at (0, 0)', valid=True)
    (0, 1) DeepTrackDataObject(data='Data at (0, 1)', valid=True)
    (1, 0) DeepTrackDataObject(data='Data at (1, 0)', valid=True)
    (1, 1) DeepTrackDataObject(data='Data at (1, 1)', valid=True)

    >>> for key in data_dict.keys():
    ...     print(key)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)

    >>> for value in data_dict.values():
    ...     print(value)
    DeepTrackDataObject(data='Data at (0, 0)', valid=True)
    DeepTrackDataObject(data='Data at (0, 1)', valid=True)
    DeepTrackDataObject(data='Data at (1, 0)', valid=True)
    DeepTrackDataObject(data='Data at (1, 1)', valid=True)

    Check if an _ID is valid according to current keylength:
    >>> data_dict.valid_index((0, 1))
    True

    >>> data_dict.valid_index((0,))  # Shorter than keylength
    False

    >>> data_dict.valid_index((0, 1, 2))  # Longer than keylength
    False

    >>> data_dict.valid_index((2, 2))  # Valid length, even if not created yet
    True

    """

    keylength: int | None
    _dict: dict[tuple[int, ...], DeepTrackDataObject]

    def __init__(self: DeepTrackDataDict):
        """Initialize the data dictionary.

        It initializes `keylength` to `None` and `dict` to an empty dictionary,
        indicating no data objects are currently stored.
        
        """

        self.keylength = None
        self._dict = {}

    def invalidate(self: DeepTrackDataDict) -> None:
        """Mark all stored data objects as invalid.

        It calls `invalidate()` on every `DeepTrackDataObject` in the
        dictionary.

        """

        for dataobject in self._dict.values():
            dataobject.invalidate()

    def validate(self: DeepTrackDataDict) -> None:
        """Mark all stored data objects as valid.

        It calls `validate()` on every `DeepTrackDataObject` in the dictionary.

        """

        for dataobject in self._dict.values():
            dataobject.validate()

    def valid_index(
        self: DeepTrackDataDict,
        _ID: tuple[int, ...],
    ) -> bool:
        """Check if a given _ID is valid for this data dictionary.

        If `keylength` is `None`, any tuple `_ID` is considered valid since no 
        entries have been created yet.

        If `_ID` already exists in `dict`, it is automatically valid.
        
        Otherwise, `_ID` must have the same length as `keylength` to be
        considered valid.
        
        Parameters
        ----------
        _ID: tuple[int, ...]
            The index to check, consisting of a tuple of integers.

        Returns
        -------
        bool
            `True` if the _ID is valid given the current configuration, `False` 
            otherwise.

        Raises
        ------
        AssertionError
            If `_ID` is not a tuple of integers.
        
        """

        # Ensure _ID is a tuple of integers.
        assert isinstance(_ID, tuple), (
            f"Data index {_ID} is not a tuple. Got: {type(_ID).__name__}."
        )
        assert all(isinstance(i, int) for i in _ID), (
            f"Data index {_ID} is not a tuple of integers. "
            f"Got a tuple of types: {[type(i).__name__ for i in _ID]}."
        )

        # If keylength has not yet been set, all indexes are valid.
        if self.keylength is None:
            return True

        # If index is already stored, always valid.
        if _ID in self._dict:
            return True

        # Otherwise, the _ID length must match the established keylength
        # for _ID to be valid.
        return len(_ID) == self.keylength

    def create_index(
        self: DeepTrackDataDict,
        _ID: tuple[int, ...] = (),
    ) -> None:
        """Create a new data entry for the given _ID if not already existing.

        Each newly created index is associated with a new
        `DeepTrackDataObject`.

        If `_ID` is already in `dict`, no new entry is created.
        
        If `keylength` is `None`, it is set to the length of `_ID`. Once 
        established, all subsequently created _IDs must have this same length.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            A tuple of integers representing the _ID for the data entry. 
            Default is `()`, which represents a root-level data entry with no 
            nesting.
        
        Raises
        ------
        AssertionError
            - If `_ID` is not a tuple of integers.
            - If `_ID` is not valid for the current configuration.
            
        """

        # Check if the given _ID is valid.
        # (Also: Ensure _ID is a tuple of integers.)
        assert self.valid_index(_ID), (
            f"{_ID} is not a valid index for current dictionary configuration."
        )

        # If `_ID` already exists, do nothing.
        if _ID in self._dict:
            return

        # Create a new DeepTrackDataObject for this _ID.
        self._dict[_ID] = DeepTrackDataObject()

        # If `keylength` is not set, initialize it with current _IDs length.
        if self.keylength is None:
            self.keylength = len(_ID)

    def __getitem__(
        self: DeepTrackDataDict,
        _ID: tuple[int, ...],
    ) -> DeepTrackDataObject | dict[tuple[int, ...], DeepTrackDataObject]:
        """Retrieve data associated with a given _ID.

        Parameters
        ----------
        _ID: tuple[int, ...]
            The _ID for the requested data.

        Returns
        -------
        DeepTrackDataObject or Dict[tuple[int, ...], DeepTrackDataObject]
            If `_ID` matches `keylength`, it returns the corresponding 
            `DeepTrackDataObject`.
            If `_ID` is longer than `keylength`, the request is trimmed to 
            match `keylength` and it returns the corresponding
            `DeepTrackDataObject`.
            If `_ID` is shorter than `keylength`, it returns a dict of all
            entries whose _IDs match the given `_ID` prefix.

        Raises
        ------
        AssertionError
            If `_ID` is not a tuple of integers.
        KeyError
            If the dictionary is empty (`keylength` is `None`).

        """

        # Ensure `_ID` is a tuple of integers.
        assert isinstance(_ID, tuple), (
            f"Data index {_ID} is not a tuple. Got: {type(_ID).__name__}."
        )
        assert all(isinstance(i, int) for i in _ID), (
            f"Data index {_ID} is not a tuple of integers. "
            f"Got a tuple of types: {[type(i).__name__ for i in _ID]}."
        )

        if self.keylength is None:
            raise KeyError("Attempting to index an empty dict.")

        # If _ID matches keylength, return corresponding DeepTrackDataObject.
        if len(_ID) == self.keylength:
            if _ID not in self._dict:
                raise KeyError(
                    f"The _ID {_ID} does not exist in this DeepTrackDataDict. "
                    f"Available keys: {list(self._dict.keys())}"
                )
            return self._dict[_ID]

        # If _ID longer than keylength, trim the requested _ID
        # and return corresponding DeepTrackDataObject.
        if len(_ID) > self.keylength:
            return self[_ID[: self.keylength]]

        # If _ID shorter than keylength, return a slice of all matching items.
        return {k: v for k, v in self._dict.items() if k[: len(_ID)] == _ID}

    def __contains__(
        self: DeepTrackDataDict,
        _ID: tuple[int, ...],
    ) -> bool:
        """Check if a given _ID exists in the dictionary.

        Parameters
        ----------
        _ID: tuple[int, ...]
            The _ID to check.

        Returns
        -------
        bool
            `True` if the _ID exists, `False` otherwise.

        Raises
        ------
        AssertionError
            If `_ID` is not a tuple of integers.

        """

        # Ensure _ID is a tuple of integers.
        assert isinstance(_ID, tuple), (
            f"Data index {_ID} is not a tuple. Got: {type(_ID).__name__}."
        )
        assert all(isinstance(i, int) for i in _ID), (
            f"Data index {_ID} is not a tuple of integers. "
            f"Got a tuple of types: {[type(i).__name__ for i in _ID]}."
        )

        return _ID in self._dict

    def __len__(self: DeepTrackDataDict) -> int:
        """Return the number of stored entries.

        Returns
        -------
        int
            The number of `_ID` entries in the dictionary.

        """

        return len(self._dict)

    def __iter__(self: DeepTrackDataDict) -> Iterator[tuple[int, ...]]:
        """Iterate over the keys of the dictionary.

        Returns
        -------
        Iterator[tuple[int, ...]]
            An iterator over the dictionary's keys.

        """

        return iter(self._dict)

    def items(
        self: DeepTrackDataDict,
    ) -> ItemsView[tuple[int, ...], DeepTrackDataObject]:
        """Return a view of the dictionary’s (key, value) pairs.

        Returns
        -------
        ItemsView[tuple[int, ...], DeepTrackDataObject]
            A dynamic view of the internal dictionary’s entries.

        """

        return self._dict.items()

    def keys(self: DeepTrackDataDict) -> KeysView[tuple[int, ...]]:
        """Return a view of the dictionary’s keys.

        Returns
        -------
        KeysView[tuple[int, ...]]
            A dynamic view of the internal dictionary’s keys.

        """

        return self._dict.keys()

    def values(self: DeepTrackDataDict) -> ValuesView[DeepTrackDataObject]:
        """Return a view of the dictionary’s values.

        Returns
        -------
        ValuesView[DeepTrackDataObject]
            A dynamic view of the internal dictionary’s values.

        """
        return self._dict.values()

    def __repr__(self: DeepTrackDataDict) -> str:
        """Return a string representation of the data dictionary.

        Provides a concise summary of the current `DeepTrackDataDict` instance,
        including the number of stored entries and the current `keylength`. It
        is useful for debugging and logging.

        Returns
        -------
        str
            A string in the format:
            "DeepTrackDataDict(<number> entries, keylength=<keylength>)".

        """

        return (
            f"{self.__class__.__name__}("
            f"{len(self)} entries, keylength={self.keylength})"
        )

    @property
    def dict(self: DeepTrackDataDict) -> dict[tuple[int, ...], DeepTrackDataObject]:
        """Access the internal data dictionary (read-only).

        This property exposes the internal `_dict` attribute as a public
        read-only interface. It allows access to all stored data objects
        indexed by their `_ID`.

        Returns
        -------
        dict[tuple[int, ...], DeepTrackDataObject]
            The mapping of `_ID`s to `DeepTrackDataObject` instances.

        """

        return self._dict


class DeepTrackNode:
    """Node in a DeepTrack2 computation graph, supporting operator overloading.

    `DeepTrackNode` represents a node within a DeepTrack2 computation graph. 
    Each node can store data and compute new values based on its dependencies.
    The value of a node is computed by calling its `action`.

    `DeepTrackNode` supports operator overloading, enabling intuitive
    construction of computation graphs using standard Python operators.
    For example, nodes can be added, multiplied, subtracted, or compared
    directly (e.g., `node1 + node2`, `node1 * 3`, `node1 > node2`), and the
    resulting node will represent the composed operation.

    Parameters
    ----------
    action: Callable or Any, optional
        Action to compute this node's value. If not provided, uses a no-op 
        action (lambda: None).
    node_name: str or None, optional
        Optional name assigned to the node. Defaults to `None`.
    **kwargs: Any
        Additional arguments for subclasses or extended functionality.

    Attributes
    ----------
    node_name: str or None
        Optional name assigned to the node. Defaults to `None`.
    data: DeepTrackDataDict
        Dictionary-like object for storing data, indexed by tuples of integers.
    children: WeakSet[DeepTrackNode]
        Nodes that depend on this node (its children, grandchildren, etc.).
        This is a weakref.WeakSet, so references are weak and do not prevent
        garbage collection of nodes that are no longer used.
    dependencies: WeakSet[DeepTrackNode]
        Nodes on which this node depends (its parents, grandparents, etc.).
        This is a weakref.WeakSet, for efficient memory management.
    _action: Callable
        The function or lambda-function to compute the node value.
    _accepts_ID: bool
        Whether `action` accepts an input _ID.
    _all_children: set[DeepTrackNode]
        All nodes in the subtree rooted at the node, including the node itself.
    _citations: list[str]
        Citations associated with this node.
    
    Methods
    -------
    `action: property`
        Get or set the computation function for the node (stored as `_action`).
    `add_child(child) -> DeepTrackNode`
        Add a child node that depends on this node.
        Also add the dependency on this node in the child node.
    `add_dependency(parent) -> DeepTrackNode`
        Add a dependency, making this node depend on the parent node.
        Also set this node as a child of the parent node.
    `store(data, _ID) -> DeepTrackNode`
        Store computed data for the given `_ID`.
    `is_valid(_ID) -> bool`
        Check whether the data for the given `_ID` is valid.
    `valid_index(_ID) -> bool`
        Check whether the given `_ID` is valid for this node.
    `invalidate(_ID) -> DeepTrackNode`
        Invalidate the data for the given `_ID` and all child nodes.
    `validate(_ID) -> DeepTrackNode`
        Validate the data for the given `_ID`, marking it as up-to-date, but 
        not its children.
    `update() -> DeepTrackNode`
        Reset the data.
    `set_value(value, _ID) -> DeepTrackNode`
        Set a value for the given `_ID`. If the new value differs from the 
        current value, the node is invalidated to ensure dependencies are 
        recomputed.
    `recurse_children(memory) -> set[DeepTrackNode]`
        Return all child nodes in the dependency tree rooted at this node.
    `recurse_dependencies(memory) -> Iterator[DeepTrackNode]`
        Yield all nodes that this node depends on, traversing dependencies.
    `get_citations() -> set[str]`
        Return a set of citations for this node and its dependencies.
    `__call__(_ID) -> Any`
        Evaluate the node's computation for the given `_ID`, recomputing if 
        necessary.
    `current_value(_ID) -> Any`
        Return the currently stored value for the given `_ID` without 
        recomputation.
    `__hash__() -> int`
        Return a unique hash for this node.
    `__getitem__(idx) -> DeepTrackNode`
        Creates a new node that indexes into this node's computed data.
    `__repr__(self) -> str:`
        Return a string representation of the node.

    Supported Operators
    -------------------
    `DeepTrackNode` supports the following Python operators:

    Arithmetic:
        +   Addition (__add__, __radd__)
        -   Subtraction (__sub__, __rsub__)
        *   Multiplication (__mul__, __rmul__)
        /   True division (__truediv__, __rtruediv__)
        //  Floor division (__floordiv__, __rfloordiv__)

    Comparison:
        <   Less than (__lt__, __gt__)
        >   Greater than (__gt__, __lt__)
        <=  Less than or equal (__le__, __ge__)
        >=  Greater than or equal (__ge__, __le__)

    Each operation returns a new `DeepTrackNode` representing the result of the
    corresponding operation in the computation graph.

    Examples
    --------
    >>> from deeptrack.backend.core import DeepTrackNode

    Create two `DeepTrackNode` objects, one as a parent and one as a child:
    >>> parent = DeepTrackNode(action=lambda: 10)
    >>> child = DeepTrackNode(action=lambda _ID=None: parent(_ID) * 2)
    >>> parent.add_child(child)

    Store and retrieve data for specific _IDs:
    >>> parent.store(15, _ID=(0,))
    >>> parent.store(20, _ID=(1,))
    >>> parent.current_value((0,))
    15
    >>> parent.current_value((1,))
    20

    Compute and retrieve the value for the child node:

    >>> child(_ID=(0,))
    30
    >>> child(_ID=(1,))
    40

    Validation and invalidation:

    >>> parent.is_valid((0,))
    True
    >>> child.is_valid((0,))
    True

    >>> parent.invalidate((0,))
    >>> parent.is_valid((0,))
    False
    >>> child.is_valid((0,))
    False

    >>> parent.validate((0,))
    >>> parent.is_valid((0,))
    True
    >>> child.is_valid((0,))
    False

    Setting a value and automatic invalidation:

    >>> parent.current_value((0,))
    15
    >>> child((1,))  # Computes and stores the value in child
    >>> child.current_value((0,))
    30

    >>> parent.set_value(42, _ID=(0,))
    >>> parent.current_value((0,))
    42
    >>> child((0,))  # Recomputes and stores the value in child
    >>> child.current_value((0,))
    84

    Resetting all data in the dependency tree (recomputation required):

    >>> parent.update()

    Dependency graph traversal (children and dependencies):

    >>> all_children = parent.recurse_children()
    >>> all_dependencies = list(child.recurse_dependencies())

    Operator overloading—arithmetic and comparison:

    >>> node_a = DeepTrackNode(lambda: 5)
    >>> node_b = DeepTrackNode(lambda: 3)

    >>> sum_node = node_a + node_b
    >>> sum_node()
    8

    >>> diff_node = node_a - node_b
    >>> diff_node()
    2

    >>> prod_node = node_a * 2
    >>> prod_node()
    10

    >>> div_node = node_a / node_b
    >>> div_node()
    1.666...

    >>> floordiv_node = node_a // node_b
    >>> floordiv_node()
    1

    >>> lt_node = node_a < node_b
    >>> lt_node()
    False

    >>> ge_node = node_a >= node_b
    >>> ge_node()
    True

    Indexing into computed data:

    >>> vector_node = DeepTrackNode(lambda: [10, 20, 30])
    >>> first_element = vector_node[0]
    >>> first_element()
    10

    Citations for a node and its dependencies:

    >>> parent.get_citations()  # Set of citation strings
    {...} 

    """

    node_name: str | None
    data: DeepTrackDataDict
    children: WeakSet[DeepTrackNode]
    dependencies: WeakSet[DeepTrackNode]

    _action: Callable[..., Any]
    _accepts_ID: bool

    _all_children: set[DeepTrackNode]

    # Citations associated with DeepTrack2.
    _citations: list[str] = [CITATION_MIDTVEDT2021QUANTITATIVE]

    @property
    def action(self: DeepTrackNode) -> Callable[..., Any]:
        """Get the function used to compute this node's value.

        When accessed, it returns the current action. This is often a function
        or lambda-function that takes `_ID` as an optional parameter if 
        `_accepts_ID` is True.

        Returns
        -------
        Callable[..., Any]
            The function used to compute this node's value.

        """

        return self._action

    @action.setter
    def action(
        self: DeepTrackNode,
        _action: Callable[..., Any],
    ) -> None:
        """Set the action used to compute this node's value.

        Parameters
        ----------
        _action: Callable[..., Any]
            A function or lambda-function used for computing the node's value.
            If the function's signature includes `_ID`, this node will pass
            `_ID` when calling `action`.

        """

        self._action = _action
        self._accepts_ID = "_ID" in get_kwarg_names(_action)

    def __init__(
        self: DeepTrackNode,
        action: Callable[..., Any] | Any = None,
        node_name: str | None = None,
        **kwargs: Any,
    ):
        """Initialize a new DeepTrackNode.

        Parameters
        ----------
        action: Callable or Any, optional
            Action to compute this node's value. If not provided, uses a no-op 
            action (lambda: None).
        name: str or None, optional
            Optional name for the node. Defaults to `None`.
        **kwargs: Any
            Additional arguments for subclasses or extended functionality.
            
        """

        # Call super init in case of multiple inheritance.
        super().__init__(**kwargs)

        # Initialize attributes.
        self.node_name = node_name
        self.data = DeepTrackDataDict()
        self.children = WeakSet()
        self.dependencies = WeakSet()

        # If action is provided, set it.
        # If it's callable, use it directly;
        # otherwise, wrap it in a lambda.
        if callable(action):
            self._action = action
        else:
            self._action = lambda: action

        # Check if action accepts `_ID`.
        self._accepts_ID = "_ID" in get_kwarg_names(self.action)

        # Keep track of all children, including this node.
        self._all_children = set()
        self._all_children.add(self)

    def add_child(
        self: DeepTrackNode,
        child: DeepTrackNode,
    ) -> DeepTrackNode:
        """Add a child node to the current node.

        Adding a child also updates `_all_children` for this node and all 
        its dependencies. It also ensures that dependency and child 
        relationships remain consistent.

        Parameters
        ----------
        child: DeepTrackNode
            The child node that depends on this node.
        
        Returns
        -------
        self: DeepTrackNode
            Return the current node for chaining.

        """

        self.children.add(child)
        if self not in child.dependencies:
            child.add_dependency(self)  # Ensure bidirectional relationship.

        # Get all children of `child` and add `child` itself.
        children = child._all_children.copy()
        children.add(child)

        # Merge all these children into this node's subtree.
        self._all_children = self._all_children.union(children)
        for parent in self.recurse_dependencies():
            parent._all_children = parent._all_children.union(children)

        return self

    def add_dependency(
        self: DeepTrackNode,
        parent: DeepTrackNode,
    ) -> DeepTrackNode:
        """Adds a dependency, making this node depend on a parent node.

        Parameters
        ----------
        parent: DeepTrackNode
            The parent node that this node depends on. If `parent` changes, 
            this node's data may become invalid.

        Returns
        -------
        self: DeepTrackNode
            Return the current node for chaining.
        
        """

        self.dependencies.add(parent)

        parent.add_child(self)  # Ensure the child relationship is also set.

        return self

    def store(
        self: DeepTrackNode,
        data: Any,
        _ID: tuple[int, ...] = (),
    ) -> DeepTrackNode:
        """Store computed data in this node.

        Parameters
        ----------
        data: Any
            The data to be stored.
        _ID: tuple[int, ...], optional
            The index for this data. If `_ID` does not exist, it creates it.
            Defaults to (), indicating a root-level entry.

        Returns
        -------
        self: DeepTrackNode
            Return the current node for chaining.
        
        """

        # Create the index if necessary, then store data in it.
        self.data.create_index(_ID)

        self.data[_ID].store(data)

        return self

    def is_valid(
        self: DeepTrackNode,
        _ID: tuple[int, ...] = (),
    ) -> bool:
        """Check whether data for the given _ID is valid.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            The _ID to check validity for.

        Returns
        -------
        bool
            `True` if data at `_ID` is valid, otherwise `False`.
        
        """

        try:
            return self.data[_ID].is_valid()
        except (KeyError, AttributeError):
            return False

    def valid_index(
        self: DeepTrackNode,
        _ID: tuple[int, ...],
    ) -> bool:
        """Check if _ID is a valid index for this node's data.

        Parameters
        ----------
        _ID: tuple[int, ...]
            The _ID to validate.

        Returns
        -------
        bool
            `True` if `_ID` is valid, otherwise `False`.
        
        """

        return self.data.valid_index(_ID)

    def invalidate(
        self: DeepTrackNode,
        _ID: tuple[int, ...] = (),
    ) -> DeepTrackNode:
        """Mark this node's data and all its children's data as invalid.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            The _ID to invalidate. Default is empty tuple, indicating 
            potentially the full dataset.

        Returns
        -------
        self: DeepTrackNode
            Return the current node for chaining.
        
        Note
        ----
        At the moment, the code to invalidate specific _IDs is not implemented, 
        so the _ID parameter is not effectively used.

        """

        # Invalidate data for all children of this node.

        for child in self.recurse_children():
            child.data.invalidate()

        return self

    def validate(
        self: DeepTrackNode,
        _ID: tuple[int, ...] = (),
    ) -> DeepTrackNode:
        """Mark this node's data as valid.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            The _ID to validate. Default is empty tuple.

        Returns
        -------
        self: DeepTrackNode

        """

        self.data[_ID].validate()

        return self

    def update(self: DeepTrackNode) -> DeepTrackNode:
        """Reset data in all children.

        This method resets `data` for all children of each dependency, 
        effectively clearing cached values to force a recomputation on the next 
        evaluation.
        
        Returns
        -------
        self: DeepTrackNode
            Return the current node for chaining.
        
        """

        # Pre-instantiate memory for optimization,
        # used to avoid repeated processing of the same nodes.
        child_memory = []

        # For each dependency, reset data in all of its children.
        for dependency in self.recurse_dependencies():
            for dep_child in dependency.recurse_children(memory=child_memory):
                dep_child.data = DeepTrackDataDict()

        return self

    def set_value(
        self: DeepTrackNode,
        value: Any,
        _ID: tuple[int, ...] = (),
    ) -> DeepTrackNode:
        """Set a value for this node's data at _ID.

        If the value is different from the currently stored one (or if it is 
        invalid), it will invalidate the old data before storing the new one.

        Parameters
        ----------
        value: Any
            The value to store.
        _ID: tuple[int, ...], optional
            The _ID at which to store the value.

        Returns
        -------
        self: DeepTrackNode
            Return the current node for chaining.
        
        """

        # Check if current value is equivalent. If not, invalidate and store
        # the new value. If set to same value, no need to invalidate.
        if not (
            self.is_valid(_ID=_ID)
            and _equivalent(value, self.data[_ID].current_value())
        ):
            self.invalidate(_ID=_ID)
            self.store(value, _ID=_ID)

        return self

    def recurse_children(
        self: DeepTrackNode,
        memory: set[DeepTrackNode] | None = None,
    ) -> set[DeepTrackNode]:
        """Return all children of this node.

        Parameters
        ----------
        memory: set, optional
            Set of nodes that have already been visited (not used directly
            here).

        Returns
        -------
        set
            All nodes in the subtree rooted at this node, including itself.

        """

        # Simply return `_all_children` since it's maintained incrementally.
        return self._all_children

    def old_recurse_children(
        self: DeepTrackNode,
        memory: list[DeepTrackNode] | None = None,
    ) -> Iterator[DeepTrackNode]:
        """Legacy recursive method for traversing children.

        Parameters
        ----------
        memory: list, optional
            A list to remember visited nodes, ensuring that each node is 
            yielded only once.

        Yields
        ------
        DeepTrackNode
            Yields each node in a depth-first traversal.

        Notes
        -----
        This method is kept for backward compatibility or debugging purposes.

        """

        # On first call, instantiate memory.
        if memory is None:
            memory = []

        # Make sure each DeepTrackNode is only yielded once.
        if self in memory:
            return

        # Remember self.
        memory.append(self)

        # Yield self and recurse children.
        yield self

        # Recursively traverse children.
        for child in self.children:
            yield from child.recurse_children(memory=memory)

    def recurse_dependencies(
        self: DeepTrackNode,
        memory: list[DeepTrackNode] | None = None,
    ) -> Iterator[DeepTrackNode]:
        """Yield all dependencies of this node, ensuring each is visited once.

        Parameters
        ----------
        memory: list, optional
            A list of visited nodes to avoid repeated visits or infinite loops.

        Yields
        ------
        DeepTrackNode
            Yields this node and all nodes it depends on.
        
        """

        # On first call, instantiate memory.
        if memory is None:
            memory = []

        # Make sure each DeepTrackNode is only yielded once.
        if self in memory:
            return

        # Remember self.
        memory.append(self)

        # Yield self and recurse dependencies.
        yield self

        # Recursively yield dependencies.
        for dependency in self.dependencies:
            yield from dependency.recurse_dependencies(memory=memory)

    def get_citations(self: DeepTrackNode) -> set[str]:
        """Get citations from this node and all its dependencies.

        It gathers citations from this node and all nodes that it depends on. 
        Citations are stored as the class attribute `_citations`.

        Returns
        -------
        set[str]
            Set of all citations relevant to this node and its dependency tree.
        
        """

        # Initialize citations as a set of elements from self.citations.
        citations = set(self._citations) if self._citations else set()

        # Recurse through dependencies to collect all citations.
        for dependency in self.recurse_dependencies():
            for obj in type(dependency).mro():
                if hasattr(obj, "citations"):
                    # Add the citations of the current object.
                    citations.update(
                        obj.citations if isinstance(obj.citations, list)
                        else [obj.citations]
                    )

        return citations

    def __call__(
        self: DeepTrackNode,
        _ID: tuple[int, ...] = (),
    ) -> Any:
        """Evaluate this node at _ID.

        If valid data is already stored at `_ID`, it is returned. Otherwise,
        the node's `action` function is called to compute the value, which is
        then stored and returned. The `_ID` is passed to `action` only if it
        is declared to accept it.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            The `_ID` at which to evaluate the node's action. Defaults to `()`.

        Returns
        -------
        Any
            The computed or retrieved data for the given `_ID`.

        """

        # First try to return the already stored value, if it's valid.
        if self.is_valid(_ID):
            try:
                return self.current_value(_ID)
            except KeyError:
                pass  # Data might have been invalidated or removed.

        # Call action with or without `_ID` depending on `_accepts_ID`.
        if self._accepts_ID:
            new_value = self.action(_ID=_ID)
        else:
            new_value = self.action()

        # Store the newly computed value.
        self.store(new_value, _ID=_ID)

        # Return the newly stored value.
        return self.current_value(_ID)

    def current_value(
        self: DeepTrackNode,
        _ID: tuple[int, ...] = (),
    ) -> Any:
        """Retrieve the currently stored value at _ID.

        Parameters
        ----------
        _ID: tuple[int, ...], optional
            The `_ID` at which to retrieve the current value. Defaults to `()`.

        Returns
        -------
        Any
            The currently stored value for `_ID`.
        
        """

        return self.data[_ID].current_value()

    def __hash__(self: DeepTrackNode) -> int:
        """Return a unique hash for this node.

        Uses the node's `id` to ensure uniqueness.
        
        """

        return id(self)

    def __getitem__(
        self: DeepTrackNode,
        idx: Any,
    ) -> DeepTrackNode:
        """Allow indexing into the node's computed data.

        Parameters
        ----------
        idx: Any
            The index applied to the result of evaluating this node.

        Returns
        -------
        DeepTrackNode
            A new node that, when evaluated, applies `idx` to the result of 
            `self`.

        Notes
        -----
        This effectively creates a node that corresponds to `self(...)[idx]`, 
        allowing you to select parts of the computed data dynamically.

        """

        # Create a new node whose action indexes into this node's result.
        node = DeepTrackNode(lambda _ID=None: self(_ID=_ID)[idx])

        self.add_child(node)
        # node.add_dependency(self)  # Already executed by add_child.

        return node

    def __repr__(self: DeepTrackNode) -> str:
        """Return a string representation of the node.

        This method returns a concise textual description of the node for
        debugging and introspection. The string includes:
        
        - The node's class name (`DeepTrackNode`)
        - Its `name`, if provided
        - The number of stored data entries (`len`)
        - The name of the action function or type (`action`)
        - The list of stored `_ID`s (excluding the root `()`), if any exist

        Returns
        -------
        str
            A string in the format: "DeepTrackNode(name='<name>', len=<N>,
            action=<action_name>, IDs=[...])" Fields `name=...` and `IDs=[...]`
            are included only if applicable.

        """

        action_name = getattr(
            self._action,
            "__name__",
            type(self._action).__name__,
        )

        ID_list = [_ID for _ID in self.data.dict if _ID != tuple()]

        parts = [
            f"name='{self.node_name}'" if self.node_name else None,
            f"len={len(self.data)}",
            f"action={action_name}",
            f"IDs={ID_list}" if ID_list else None,
        ]

        return f"{self.__class__.__name__}({', '.join(p for p in parts if p)})"

    # Node-node operators.
    # These methods define arithmetic and comparison operations for
    # DeepTrackNode objects. Each operation creates a new DeepTrackNode that
    # represents the result of applying the corresponding operator to `self`
    # and `other`. The operators are applied lazily and will be computed only
    # when the resulting node is evaluated.

    def __add__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Add node to another node or value.

        Creates a new `DeepTrackNode` representing the addition of the values
        produced by the `self` node and the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to add.

        Returns
        -------
        DeepTrackNode
            A new node that represents the addition operation `self + other`.
        
        """

        return _create_node_with_operator(operator.__add__, self, other)

    def __radd__(
        self: DeepTrackNode,
        other: Any,
    ) -> DeepTrackNode:
        """Add other value to node (right-hand).

        Creates a new `DeepTrackNode` representing the addition of the `other`
        value and the `self` node.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The value or node to add.

        Returns
        -------
        DeepTrackNode
            A new node that represents the addition operation `other + self`.
        
        """

        return _create_node_with_operator(operator.__add__, other, self)

    def __sub__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Subtract a node from another node or value.

        Creates a new `DeepTrackNode` representing the subtraction of the 
        values produced by the `self`node and the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to subtract.

        Returns
        -------
        DeepTrackNode
            A new node that represents the subtraction operation
            `self - other`.
        
        """

        return _create_node_with_operator(operator.__sub__, self, other)

    def __rsub__(
        self: DeepTrackNode,
        other: Any,
    ) -> DeepTrackNode:
        """Subtract node from other value (right-hand).

        Creates a new `DeepTrackNode` representing the subtraction of the value
        produced by the `other` value from the `self` node.

        Parameters
        ----------
        other: Any
            The value or node to subtract from.

        Returns
        -------
        DeepTrackNode
            A new node that represents the subtraction operation
            `other - self`.
        
        """

        return _create_node_with_operator(operator.__sub__, other, self)

    def __mul__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Multiply node by another node or value.

        Creates a new `DeepTrackNode` representing the multiplication of the 
        values produced by the `self` node and the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to multiply by.

        Returns
        -------
        DeepTrackNode
            A new node that represents the multiplication operation 
            `self * other`.
        
        """

        return _create_node_with_operator(operator.__mul__, self, other)

    def __rmul__(
        self: DeepTrackNode,
        other: Any,
    ) -> DeepTrackNode:
        """Multiply other value by node (right-hand).

        Creates a new `DeepTrackNode` representing the multiplication of the
        `other` value by the self node.

        Parameters
        ----------
        other: Any
            The value or node to multiply.

        Returns
        -------
        DeepTrackNode
            A new node that represents the multiplication operation
            `other * self`.

        """

        return _create_node_with_operator(operator.__mul__, other, self)

    def __truediv__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Divide node by another node or value.

        Creates a new `DeepTrackNode` representing the division of the value
        produced by the `self` node by the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to divide by.

        Returns
        -------
        DeepTrackNode
            A new node that represents the division operation (`self / other`).
        
        """

        return _create_node_with_operator(operator.__truediv__, self, other)

    def __rtruediv__(
        self: DeepTrackNode,
        other: Any,
    ) -> DeepTrackNode:
        """Divide other value by node (right-hand).

        Creates a new `DeepTrackNode` representing the division of the `other`
        value by the `self` node.

        Parameters
        ----------
        other: Any
            The value or node to divide.

        Returns
        -------
        DeepTrackNode
            A new node that represents the division operation `other / self`.
        
        """

        return _create_node_with_operator(operator.__truediv__, other, self)

    def __floordiv__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Perform floor division of node by another node or value.

        Creates a new `DeepTrackNode` representing the floor division of the
        value produced by the `self` node by the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to divide by.

        Returns
        -------
        DeepTrackNode
            A new node that represents the floor division operation 
            `self // other`.
        
        """

        return _create_node_with_operator(operator.__floordiv__, self, other)

    def __rfloordiv__(
        self: DeepTrackNode,
        other: Any,
    ) -> DeepTrackNode:
        """Perform floor division of other value by node (right-hand).

        Creates a new `DeepTrackNode` representing the floor division of the
        other value by the `self` node.

        Parameters
        ----------
        other: Any
            The value or node to divide.

        Returns
        -------
        DeepTrackNode
            A new node that represents the floor division operation
            `other // self`.
        
        """

        return _create_node_with_operator(operator.__floordiv__, other, self)

    def __lt__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Check whether node is less than other node or value.

        Creates a new `DeepTrackNode` representing whether the `self` node is
        less than the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison `self < other`.
        
        """

        return _create_node_with_operator(operator.__lt__, self, other)

    def __gt__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Check whether node is greater than other node or value.

        Creates a new `DeepTrackNode` representing whether the `self` node is
        greater than the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison `self > other`.
        
        """

        return _create_node_with_operator(operator.__gt__, self, other)

    def __le__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Check whether node is less than or equal to other node or value.

        Creates a new `DeepTrackNode` representing whether the `self` node is
        less than or equal to the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison `self <= other`.
        
        """

        return _create_node_with_operator(operator.__le__, self, other)

    def __ge__(
        self: DeepTrackNode,
        other: DeepTrackNode | Any,
    ) -> DeepTrackNode:
        """Check whether node is greater than or equal to other node or value.

        Creates a new `DeepTrackNode` representing whether the `self` node is
        greater than or equal to the `other` node or value.

        Parameters
        ----------
        other: DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison `self >= other`.

        """

        return _create_node_with_operator(operator.__ge__, self, other)


def _equivalent(
    a: Any,
    b: Any,
) -> bool:
    """Check if two objects are equivalent.

    This internal helper function provides a basic implementation to determine 
    equivalence between two objects:
    - If `a` and `b` are the same object (identity check), they are considered 
      equivalent.
    - If both `a` and `b` are empty lists, they are considered equivalent.

    Additional cases can be implemented as needed to refine this behavior.

    NOTE: For immutable built-in types like empty tuples, integers, and `None`,
    Python may reuse the same object in memory. Thus, `a is b` may return
    `True` even if the objects are created separately.

    Parameters
    ----------
    a: Any
        The first object to compare.
    b: Any
        The second object to compare.

    Returns
    -------
    bool
        `True` if the objects are equivalent, `False` otherwise.

    Examples
    --------
    >>> from deeptrack.backend.core import _equivalent

    >>> _equivalent([], [])
    True

    >>> a = [1, 2]
    >>> _equivalent(a, a)
    True

    >>> _equivalent([1], [1])
    False

    >>> _equivalent([], ())
    False

    >>> _equivalent(None, None)
    True

    """

    # If a and b are the same object, return True.
    if a is b:
        return True

    # If a and b are empty lists, consider them identical.
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == 0 and len(b) == 0

    # Otherwise, return False.
    return False


def _create_node_with_operator(
    op: Callable,
    a: Any,
    b: Any,
) -> DeepTrackNode:
    """Create a new computation node using the given operator and operands.

    This internal helper function constructs a `DeepTrackNode` obtained from 
    the  application of the specified operator to two operands. If the operands 
    are not already `DeepTrackNode` instances, they are converted to nodes.

    This function also establishes bidirectional relationships between the new 
    node and its operands:
    
    - The new node is added as a child of the operands `a` and `b`.
    - The operands `a` and `b` are added as dependencies of the new node.
    - The operator `op` is applied lazily, meaning it will be evaluated when 
      the new node is called, for computational efficiency.

    Parameters
    ----------
    op: Callable
        The operator function.
    a: Any
        First operand. If not a `DeepTrackNode`, it will be wrapped in one.
    b: Any
        Second operand. If not a `DeepTrackNode`, it will be wrapped in one.

    Returns
    -------
    DeepTrackNode
        A new `DeepTrackNode` containing the result of applying the operator 
        `op` to the values of nodes `a` and `b`.

    """

    # Ensure `a` is a `DeepTrackNode`. Wrap it if necessary.
    if not isinstance(a, DeepTrackNode):
        a = DeepTrackNode(a)

    # Ensure `b` is a `DeepTrackNode`. Wrap it if necessary.
    if not isinstance(b, DeepTrackNode):
        b = DeepTrackNode(b)

    # New node that applies the operator `op` to the values of `a` and `b`.
    new_node = DeepTrackNode(lambda _ID=(): op(a(_ID=_ID), b(_ID=_ID)))

    # Set the new node as a child of both `a` and `b`.
    # (Also: Establish dependency relationships between the nodes.)
    a.add_child(new_node)
    b.add_child(new_node)

    # Establish dependency relationships between the nodes.
    # (Not needed because already done implicitly above.)
    # new_node.add_dependency(a)
    # new_node.add_dependency(b)

    return new_node
