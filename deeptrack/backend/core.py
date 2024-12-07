"""DeepTrack2 core package.

This package provides the core classes and functionality required for managing, 
processing, and evaluating data within the DeepTrack framework. It is designed 
to support building flexible pipelines for scientific data analysis and machine 
learning applications.

This package is the core component of the DeepTrack2 framework. It enables 
users to:
- Construct flexible and efficient computational pipelines.
- Manage data and dependencies in a hierarchical structure.
- Perform lazy evaluations for performance optimization.

Main Features
-------------
Data Management:
- `DeepTrackDataObject` and `DeepTrackDataDict` provide tools to store, 
  validate, and manage data with dependency tracking.
- Enables nested structures and flexible indexing for complex data 
  hierarchies.

Computational Graphs:
- `DeepTrackNode` forms the backbone of computation pipelines, representing 
  nodes in a computation graph.
- Nodes support lazy evaluation, dependency tracking, and caching to 
  optimize performance.
- Implements mathematical operators for easy composition of computational 
  graphs.

Citations:
- Supports citing the relevant publication (`Midtvedt et al., 2021`) to 
  ensure proper attribution.

Utilities:
- Includes helper functions like `_equivalent` and 
  `_create_node_with_operator` to streamline graph operations.

Package Structure
-----------------
Data Containers:
- `DeepTrackDataObject`: A basic container for data with validation status.
- `DeepTrackDataDict`: Stores multiple data objects indexed by unique access 
                       IDs, enabling nested data storage.

Computation Nodes:
- `DeepTrackNode`: Represents a node in a computation graph, capable of lazy 
                   evaluation, caching, and dependency management.

Citation Management:
- Provides support for including citations in pipelines for academic and 
  scientific use.

Example
-------
Create a DeepTrackNode with an action:

>>> node = DeepTrackNode(lambda x: x**2)
>>> node.store(5)

Retrieve the stored value:

>>> print(node.current_value())  # Output: 25

"""

import operator  # Operator overloading for computation nodes.
from weakref import WeakSet  # Manages relationships between nodes without 
                             # creating circular dependencies.

import numpy as np
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
)

from .. import utils


citation_Midtvet2021Quantitative = """
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

    This class serves as a simple container for storing data and managing its 
    validity. It is designed to track whether the data remains valid, based on 
    changes in dependencies or other external factors that might invalidate the 
    data since it was last updated.

    Attributes
    ----------
    data : Any
        The stored data. Default is `None`. Can hold any data type.
    valid : bool
        A flag indicating whether the stored data is valid. Default is `False`.

    Methods
    -------
    store(data : Any)
        Stores data in the container and marks it as valid.
    current_value() -> Any
        Returns the currently stored data.
    is_valid() -> bool
        Checks if the stored data is valid.
    invalidate()
        Marks the data as invalid.
    validate()
        Marks the data as valid.
    """

    # Attributes.
    data: Any
    valid: bool

    def __init__(self):
        """Initialize the container without data.

        The `data` attribute is set to `None`, and the `valid` attribute is set 
        to `False` by default.
        """
        self.data = None
        self.valid = False

    def store(self, data: Any) -> None:
        """Store data in the container and mark it as valid.

        Parameters
        ----------
        data : Any
            The data to be stored in the container.
        """
        self.data = data
        self.valid = True

    def current_value(self) -> Any:
        """Retrieve the currently stored data.

        Returns
        -------
        Any
            The data stored in the container.
        """
        return self.data

    def is_valid(self) -> bool:
        """Check if the stored data is valid.

        Returns
        -------
        bool
            `True` if the data is valid, `False` otherwise.
        """
        return self.valid

    def invalidate(self) -> None:
        """Mark the stored data as invalid."""
        self.valid = False

    def validate(self) -> None:
        """Mark the stored data as valid."""
        self.valid = True


class DeepTrackDataDict:
    """Stores multiple data objects indexed by a tuple of integers (access ID).

    This class allows a single object to store multiple `DeepTrackDataObject` 
    instances concurrently, each associated with a unique tuple of integers 
    (the access ID). This is particularly useful for handling sequences of data 
    or nested structures, as required by features like `Repeat`.

    The default access ID is an empty tuple `()`. Once an entry is created, all 
    IDs must match the established key length. If an ID longer than the 
    stored length is requested, the request is trimmed. If an ID shorter than 
    what is stored is requested, a dictionary slice containing all matching 
    entries is returned. This mechanism supports flexible indexing of nested 
    data and ensures that dependencies at various nesting depths can be 
    correctly handled.

    Example
    -------
    Consider the following structure, where `Repeat` is a feature that creates
    multiple instances of another feature:

    >>> F = Repeat(Repeat(DummyFeature(prop=np.random.rand), 2), 2)

    Here, `F` contains 2 * 2 = 4 instances of the feature `prop`. 
    These can be accessed using the IDs:
    (0, 0), (0, 1), (1, 0), and (1, 1).

    In this nested structure:
    - (0, 0) refers to the first repeat of the outer feature and the first 
        repeat of the inner feature.
    - (0, 1) refers to the first repeat of the outer feature and the second 
        repeat of the inner feature.
    And so forth, resolving nested structures via tuples of indices.
    
    Attributes
    ----------
    keylength : int or None
        The length of the IDs currently stored. Set when the first entry is 
        created. If `None`, no entries have been created yet, and any ID length 
        is valid.
    dict : Dict[Tuple[int, ...], DeepTrackDataObject]
        A dictionary mapping tuples of integers (IDs) to `DeepTrackDataObject` 
        instances.

    Methods
    -------
    invalidate()
        Marks all stored data objects as invalid.
    validate()
        Marks all stored data objects as valid.
    valid_index(_ID : Tuple[int, ...]) -> bool
        Checks if the given ID is valid for the current configuration.
    create_index(_ID : Tuple[int, ...] = ())
        Creates an entry for the given ID if it does not exist.
    __getitem__(_ID : tuple) -> Union[
            DeepTrackDataObject, 
            Dict[Tuple[int, ...], DeepTrackDataObject]
        ]
        Retrieves data associated with the ID. Can return a 
        `DeepTrackDataObject` or a dict of matching entries if `_ID` is shorter 
        than `keylength`.
    __contains__(_ID : Tuple[int, ...]) -> bool
        Checks if the given ID exists in the dictionary.
    
    """

    # Attributes.
    keylength: Optional[int]
    dict: Dict[Tuple[int, ...], DeepTrackDataObject]

    def __init__(self):
        """Initialize the data dictionary.

        Initializes `keylength` to `None` and `dict` to an empty dictionary,
        indicating no data objects are currently stored.
        
        """
        
        self.keylength = None
        self.dict = {}

    def invalidate(self) -> None:
        """Mark all stored data objects as invalid.

        Calls `invalidate()` on every `DeepTrackDataObject` in the dictionary.
        
        """
        
        for dataobject in self.dict.values():
            dataobject.invalidate()

    def validate(self) -> None:
        """Mark all stored data objects as valid.

        Calls `validate()` on every `DeepTrackDataObject` in the dictionary.
        
        """
        
        for dataobject in self.dict.values():
            dataobject.validate()

    def valid_index(self, _ID: Tuple[int, ...]) -> bool:
        """Check if a given ID is valid for this data dictionary.

        Parameters
        ----------
        _ID : Tuple[int, ...]
            The index to check, consisting of a tuple of integers.

        Returns
        -------
        bool
            `True` if the ID is valid given the current configuration, `False` 
            otherwise.

        Raises
        ------
        AssertionError
            If `_ID` is not a tuple of integers. The error message includes the 
            actual `_ID` value and the types of its elements for easier 
            debugging.

        Notes
        -----
        - If `keylength` is `None`, any tuple ID is considered valid since no 
            entries have been created yet.
        - If `_ID` already exists in `dict`, it is automatically valid.
        - Otherwise, `_ID` must have the same length as `keylength` to be 
            considered valid.
        
        """
        
        assert (
            isinstance(_ID, tuple) and all(isinstance(i, int) for i in _ID)
        ), (
            f"Data index {_ID} is not a tuple of integers. "
            f"Got a tuple of types: {[type(i).__name__ for i in _ID]}."
        )

        # If keylength has not yet been set, all indexes are valid.
        if self.keylength is None:
            return True

        # If index is already stored, always valid.
        if _ID in self.dict:
            return True

        # Otherwise, the ID length must match the established keylength.
        return len(_ID) == self.keylength

    def create_index(self, _ID: Tuple[int, ...] = ()) -> None:
        """Create a new data entry for the given ID if not already existing.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            A tuple of integers representing the ID for the data entry. 
            Default is `()`, which represents a root-level data entry with no 
            nesting.
        
        Raises
        ------
        AssertionError
            - If `_ID` is not a tuple of integers. The error message includes 
                the value of `_ID` and the types of its elements.
            - If `_ID` is not a valid index according to the current 
                configuration.

        Notes
        -----
        - If `keylength` is `None`, it is set to the length of `_ID`. Once 
            established, all subsequently created IDs must have this same length.
        - If `_ID` is already in `dict`, no new entry is created.
        - Each newly created index is associated with a fresh 
            `DeepTrackDataObject`.
            
        """
        
        # Ensure `_ID` is a tuple of integers.
        assert (
            isinstance(_ID, tuple) and all(isinstance(i, int) for i in _ID)
        ), (
            f"Data index {_ID} is not a tuple of integers. "
            f"Got a tuple of types: {[type(i).__name__ for i in _ID]}."
        )

        # If `_ID` already exists, do nothing.
        if _ID in self.dict:
            return

        # Check if the given `_ID` is valid.
        assert self.valid_index(_ID), (
            f"{_ID} is not a valid index for current dictionary configuration."
        )

        # If `keylength` is not set, initialize it with current ID's length.
        if self.keylength is None:
            self.keylength = len(_ID)

        # Create a new DeepTrackDataObject for this ID.
        self.dict[_ID] = DeepTrackDataObject()

    def __getitem__(
        self, 
        _ID: Tuple[int, ...],
    ) -> Union[DeepTrackDataObject, 
               Dict[Tuple[int, ...], DeepTrackDataObject]]:
        """Retrieve data associated with a given ID.

        Parameters
        ----------
        _ID : Tuple[int, ...]
            The ID for the requested data.

        Returns
        -------
        DeepTrackDataObject or dict
            If `_ID` matches `keylength`, returns the corresponding 
            `DeepTrackDataObject`.
            If `_ID` is longer than `keylength`, the request is trimmed to 
            match `keylength`.
            If `_ID` is shorter than `keylength`, returns a dict of all entries 
            whose IDs match the given `_ID` prefix.

        Raises
        ------
        AssertionError
            If `_ID` is not a tuple of integers.
        KeyError
            If the dictionary is empty (`keylength` is `None`).
        """
        assert (
            isinstance(_ID, tuple) and all(isinstance(i, int) for i in _ID)
        ), (
            f"Data index {_ID} is not a tuple of integers. "
            f"Got a tuple of types: {[type(i).__name__ for i in _ID]}."
        )

        if self.keylength is None:
            raise KeyError("Indexing an empty dict.")

        if len(_ID) == self.keylength:
            return self.dict[_ID]
        elif len(_ID) > self.keylength:
            # Trim the requested ID to match keylength.
            return self[_ID[: self.keylength]]
        else:
            # Return a slice of all items matching the shorter ID prefix.
            return {k: v for k, v in self.dict.items() if k[: len(_ID)] == _ID}

    def __contains__(self, _ID: Tuple[int, ...]) -> bool:
        """Check if a given ID exists in the dictionary.

        Parameters
        ----------
        _ID : Tuple[int, ...]
            The ID to check.

        Returns
        -------
        bool
            `True` if the ID exists, `False` otherwise.
        """
        return _ID in self.dict


class DeepTrackNode:
    """Object corresponding to a node in a computation graph.

    This class represents a node within a computation graph, where each node 
    can store data and compute new values based on its dependencies. The value 
    of a node is computed by calling its `action` method.

    Attributes
    ----------
    data : DeepTrackDataDict
        A dictionary-like object for storing data, indexed by tuples of ints.
    children : WeakSet[DeepTrackNode]
        Set of nodes that depend on this node.
    dependencies : WeakSet[DeepTrackNode]
        Set of nodes on which this node depends.
    _action : Callable
        The function or lambda to compute new values.
    _accepts_ID : bool
        Whether `action` accepts `_ID`.
    _all_subchildren : Set[DeepTrackNode]
        All nodes in the subtree rooted at this node, including itself.
    citations : List[str]
        Citations associated with this node.
    
 Methods
    -------
    action : property
        Get/set the computation function.
    add_child(other: DeepTrackNode) -> DeepTrackNode
        Adds a child node.
    add_dependency(other: DeepTrackNode) -> DeepTrackNode
        Adds a dependency node.
    store(data: Any, _ID: Tuple[int, ...] = ()) -> DeepTrackNode
        Stores computed data.
    is_valid(_ID: Tuple[int, ...] = ()) -> bool
        Checks if data is valid.
    valid_index(_ID: Tuple[int, ...]) -> bool
        Checks if `_ID` is valid.
    invalidate(_ID: Tuple[int, ...] = ()) -> DeepTrackNode
        Invalidates this node’s and its children's data.
    validate(_ID: Tuple[int, ...] = ()) -> DeepTrackNode
        Validates this node’s and its children's data.
    _update() -> DeepTrackNode
        Internal method to reset data.
    set_value(value: Any, _ID: Tuple[int, ...] = ()) -> DeepTrackNode
        Sets a value, invalidating if necessary.
    previous(_ID: Tuple[int, ...] = ()) -> Any
        Returns previously stored value.
    recurse_children(memory: Optional[Set[DeepTrackNode]] = None) 
        -> Set[DeepTrackNode]
        Returns all subchildren.
    old_recurse_children(memory: Optional[List[DeepTrackNode]] = None) 
        -> Iterator[DeepTrackNode]
        Legacy depth-first traversal.
    recurse_dependencies(memory: Optional[List[DeepTrackNode]] = None) 
        -> Iterator[DeepTrackNode]
        Yields dependencies.
    get_citations() -> Set[str]
        Gathers citations.
    __call__(_ID: Tuple[int, ...] = ()) -> Any
        Evaluates the node.
    current_value(_ID: Tuple[int, ...] = ()) -> Any
        Returns currently stored value.
    __hash__() -> int
        Returns unique hash.
    __getitem__(idx: Any) -> DeepTrackNode
        Index into the node’s computed data.

    """

    # Attributes.
    data: DeepTrackDataDict
    children: WeakSet  #TODO 
                       # From Python 3.9, change to WeakSet['DeepTrackNode']
    dependencies: WeakSet #TODO
                          # From Python 3.9, change to WeakSet['DeepTrackNode']
    _action: Callable[..., Any]
    _accepts_ID: bool
    _all_subchildren: Set['DeepTrackNode']

    # Citations associated with this DeepTrack2.
    citations: List[str] = [citation_Midtvet2021Quantitative]

    @property
    def action(self) -> Callable[..., Any]:
        """Callable: The function that computes this node’s value.

        When accessed, returns the current action. This is often a lambda or 
        function that takes `_ID` as an optional parameter if `_accepts_ID` is 
        True.
        
        """
        
        return self._action
    
    @action.setter
    def action(self, value: Callable[..., Any]) -> None:
        """Set the action used to compute this node’s value.

        Parameters
        ----------
        value : Callable[..., Any]
            A function or lambda to be used for computing the node’s value. If 
            the function’s signature includes `_ID`, this node will pass `_ID` 
            when calling `action`.
        """
        self._action = value
        self._accepts_ID = utils.get_kwarg_names(value).__contains__("_ID")

    def __init__(
        self, 
        action: Optional[Callable[..., Any]] = None, 
        **kwargs: Any,
    ):
        """Initialize a new DeepTrackNode.

        Parameters
        ----------
        action : Callable or Any, optional
            Action to compute this node’s value. If not provided, uses a no-op.
        
        **kwargs : dict
            Additional arguments for subclasses or extended functionality.
            
        """
        
        self.data = DeepTrackDataDict()
        self.children = WeakSet()
        self.dependencies = WeakSet()
        self._action = lambda: None  # Default no-op action.

        # If action is provided, set it. If it's callable, use it directly; 
        # otherwise, wrap it in a lambda.
        if action is not None:
            if callable(action):
                self.action = action
            else:
                self.action = lambda: action

        # Check if action accepts `_ID`.
        self._accepts_ID = "_ID" in utils.get_kwarg_names(self.action)
        
        # Call super init in case of multiple inheritance.
        super().__init__(**kwargs)

        # Keep track of all subchildren, including this node.
        self._all_subchildren = set()
        self._all_subchildren.add(self)

    def add_child(self, other: 'DeepTrackNode') -> 'DeepTrackNode':
        """Add a child node, indicating that `other` depends on this node.

        Parameters
        ----------
        other : DeepTrackNode
            The node that depends on this node.
        
        Returns
        -------
        self : DeepTrackNode
            Returns the current node for chaining.

        Notes
        -----
        Adding a child also updates `_all_subchildren` for this node and all 
        its dependencies. Ensures that dependency and child relationships 
        remain consistent.
        
        """
        
        self.children.add(other)
        if self not in other.dependencies:
            other.add_dependency(self)  # Ensure bidirectional relationship.
        
        # Get all subchildren of `other` and add `other` itself.
        subchildren = other._all_subchildren.copy()
        subchildren.add(other)

        # Merge all these subchildren into this node’s subtree.
        self._all_subchildren = self._all_subchildren.union(subchildren)
        for parent in self.recurse_dependencies():
            parent._all_subchildren = \
                parent._all_subchildren.union(subchildren)

        return self

    def add_dependency(self, other: 'DeepTrackNode') -> 'DeepTrackNode':
        """Add a dependency node indicating this node depends on `other`.

        Parameters
        ----------
        other : DeepTrackNode
            The node that this node depends on. If `other` changes, this node’s 
            data may become invalid.

        Returns
        -------
        self : DeepTrackNode
            Returns the current node for chaining.
        
        """
        
        self.dependencies.add(other)
        other.add_child(self)  # Ensure the child relationship is also set.

        return self

    def store(self, data: Any, _ID: Tuple[int, ...] = ()) -> 'DeepTrackNode':
        """Store computed data in this node.

        Parameters
        ----------
        data : Any
            The data to store.
        _ID : Tuple[int, ...], optional
            The index for this data. Default is the empty tuple, indicating a 
            root-level entry.

        Returns
        -------
        self : DeepTrackNode
        
        """
        
        # Create the index if necessary, then store data in it.
        self.data.create_index(_ID)
        self.data[_ID].store(data)

        return self

    def is_valid(self, _ID: Tuple[int, ...] = ()) -> bool:
        """Check if data for the given ID is valid.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            The ID to check validity for.

        Returns
        -------
        bool
            `True` if data at `_ID` is valid, otherwise `False`.
        
        """
        
        try:
            return self.data[_ID].is_valid()
        except (KeyError, AttributeError):
            return False

    def valid_index(self, _ID: Tuple[int, ...]) -> bool:
        """Check if `_ID` is a valid index for this node’s data.

        Parameters
        ----------
        _ID : Tuple[int, ...]
            The ID to validate.

        Returns
        -------
        bool
            `True` if `_ID` is valid, otherwise `False`.
        
        """

        return self.data.valid_index(_ID)

    def invalidate(self, _ID: Tuple[int, ...] = ()) -> 'DeepTrackNode':
        """Mark this node’s data and all its children’s data as invalid.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            The ID to invalidate. Default is empty tuple, indicating 
            potentially the full dataset.

        Returns
        -------
        self : DeepTrackNode

        """
        
        # Invalidate data for all children of this node.

        for child in self.recurse_children():
            child.data.invalidate()

        return self

    def validate(self, _ID: Tuple[int, ...] = ()) -> 'DeepTrackNode':
        """Mark this node’s data and all its children’s data as valid.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            The ID to validate. Default is empty tuple.

        Returns
        -------
        self : DeepTrackNode

        Notes
        -----
        If a child does not have the specified `_ID`, it is ignored.
        
        """
        
        for child in self.recurse_children():
            try:
                child.data[_ID].validate()
            except KeyError:
                pass

        return self

    def _update(self) -> 'DeepTrackNode':
        """Internal method to reset data in all dependent children.

        This method resets `data` for all children of each dependency, 
        effectively clearing cached values to force a recomputation on the next 
        evaluation.
        
        Returns
        -------
        self : DeepTrackNode
        
        """
        
        # Pre-instantiate memory for optimization used to avoid repeated 
        # processing of the same nodes.
        child_memory = []

        # For each dependency, reset data in all of its children.
        for dependency in self.recurse_dependencies():
            for dep_child in dependency.recurse_children(memory=child_memory):
                dep_child.data = DeepTrackDataDict()

        return self

    def set_value(self, value, _ID: Tuple[int, ...] = ()) -> 'DeepTrackNode':
        """Set a value for this node’s data at `_ID`.

        If the value is different from the currently stored one (or if it is 
        invalid), it will invalidate the old data before storing the new one.

        Parameters
        ----------
        value : Any
            The value to store.
        _ID : Tuple[int, ...], optional
            The ID at which to store the value.

        Returns
        -------
        self : DeepTrackNode
        
        """
        
        # Check if current value is equivalent. If not, invalidate and store 
        # the new value. If set to same value, no need to invalidate
        if not (
            self.is_valid(_ID=_ID) 
            and _equivalent(value, self.data[_ID].current_value())
        ):
            self.invalidate(_ID=_ID)
            self.store(value, _ID=_ID)

        return self

    def previous(self, _ID: Tuple[int, ...] = ()) -> Any:
        """Retrieve the previously stored value at `_ID` without recomputing.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            The ID for which to retrieve the previous value.

        Returns
        -------
        Any
            The previously stored value if `_ID` is valid.
            Returns `[]` if `_ID` is not a valid index.
        
        """
        
        if self.data.valid_index(_ID):
            return self.data[_ID].current_value()
        else:
            return []

    def recurse_children(
        self, 
        memory: Optional[Set['DeepTrackNode']] = None,
    ) -> Set['DeepTrackNode']:
        """Return all subchildren of this node.

        Parameters
        ----------
        memory : set, optional
            Memory set to track visited nodes (not used directly here).

        Returns
        -------
        set
            All nodes in the subtree rooted at this node, including itself.
        """
        
        # Simply return `_all_subchildren` since it's maintained incrementally.
        return self._all_subchildren

    def old_recurse_children(
        self, 
        memory: Optional[List['DeepTrackNode']] = None,
    ) -> Iterator['DeepTrackNode']:
        """Legacy recursive method for traversing children.

        Parameters
        ----------
        memory : list, optional
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
        self, 
        memory: Optional[List['DeepTrackNode']] = None,
    ) -> Iterator['DeepTrackNode']:
        """Yield all dependencies of this node, ensuring each is visited once.

        Parameters
        ----------
        memory : list, optional
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

    def get_citations(self) -> Set[str]:
        """Get citations from this node and all its dependencies.

        Gathers citations from this node and all nodes that it depends on. 
        Citations are stored as a class attribute `citations`.

        Returns
        -------
        set
            Set of all citations relevant to this node and its dependency tree.
        
        """

        # Initialize citations as a set of elements from self.citations.
        citations = set(self.citations) if self.citations else set()

        # Recurse through dependencies to collect all citations.
        for dep in self.recurse_dependencies():
            for obj in type(dep).mro():
                if hasattr(obj, "citations"):
                    # Add the citations of the current object.
                    citations.update(
                        obj.citations if isinstance(obj.citations, list)
                        else [obj.citations]
                    )

        return citations

    def __call__(self, _ID: Tuple[int, ...] = ()) -> Any:
        """Evaluate this node at `_ID`.

        If the data at `_ID` is valid, it returns the stored value. Otherwise, 
        it calls `action` to compute a new value, stores it, and returns it.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            The ID at which to evaluate the node’s action.

        Returns
        -------
        Any
            The computed or retrieved data for the given `_ID`.
        
        """

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
        
        return self.current_value(_ID)

    def current_value(self, _ID: Tuple[int, ...] = ()) -> Any:
        """Retrieve the currently stored value at `_ID`.

        Parameters
        ----------
        _ID : Tuple[int, ...], optional
            The ID at which to retrieve the current value.

        Returns
        -------
        Any
            The currently stored value for `_ID`.
        
        """
        
        return self.data[_ID].current_value()

    def __hash__(self) -> int:
        """Return a unique hash for this node.

        Uses the node’s `id` to ensure uniqueness.
        
        """
        
        return id(self)

    def __getitem__(self, idx: Any) -> 'DeepTrackNode':
        """Allow indexing into the node’s computed data.

        Parameters
        ----------
        idx : Any
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
        
        # Create a new node whose action indexes into this node’s result.
        node = DeepTrackNode(lambda _ID=None: self(_ID=_ID)[idx])
        
        node.add_dependency(self)
        
        self.add_child(node)
        
        return node

    # Node-node operators.
    # These methods define arithmetic and comparison operations for 
    # DeepTrackNode objects. Each operation creates a new DeepTrackNode that 
    # represents the result of applying the corresponding operator to `self` 
    # and `other`. The operators are applied lazily and will be computed only 
    # when the resulting node is evaluated.

    def __add__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Add node to another node or value.

        Creates a new `DeepTrackNode` representing the addition of the values
        produced by this node (`self`) and another node or value (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to add.

        Returns
        -------
        DeepTrackNode
            A new node that represents the addition operation (`self + other`).
        
        """
        
        return _create_node_with_operator(operator.__add__, self, other)

    def __radd__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Add other value to node (right-hand).

        Creates a new `DeepTrackNode` representing the addition of another
        node or value (`other`) to the value produced by this node (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to add.

        Returns
        -------
        DeepTrackNode
            A new node that represents the addition operation (`other + self`).
        
        """
        
        return _create_node_with_operator(operator.__add__, other, self)

    def __sub__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Subtract another node or value from node.

        Creates a new `DeepTrackNode` representing the subtraction of the 
        values produced by another node or value (`other`) from this node 
        (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to subtract.

        Returns
        -------
        DeepTrackNode
            A new node that represents the subtraction operation 
            (`self - other`).
        
        """
        
        return _create_node_with_operator(operator.__sub__, self, other)

    def __rsub__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Subtract node from other value (right-hand).

        Creates a new `DeepTrackNode` representing the subtraction of the value
        produced by this node (`self`) from another node or value (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to subtract from.

        Returns
        -------
        DeepTrackNode
            A new node that represents the subtraction operation 
                `other - self`).
        
        """
        
        return _create_node_with_operator(operator.__sub__, other, self)

    def __mul__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Multiply node by another node or value.

        Creates a new `DeepTrackNode` representing the multiplication of the 
        values produced by this node (`self`) and another node or value 
        (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to multiply by.

        Returns
        -------
        DeepTrackNode
            A new node that represents the multiplication operation 
            (`self * other`).
        
        """
        
        return _create_node_with_operator(operator.__mul__, self, other)

    def __rmul__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Multiply other value by node (right-hand).

        Creates a new `DeepTrackNode` representing the multiplication of 
        another node or value (`other`) by the value produced by this node 
        (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to multiply.

        Returns
        -------
        DeepTrackNode
            A new node that represents the multiplication operation 
            (`other * self`).
        """
        return _create_node_with_operator(operator.__mul__, other, self)

    def __truediv__(
        self, 
        other: Union['DeepTrackNode', Any],
    ) -> 'DeepTrackNode':
        """Divide node by another node or value.

        Creates a new `DeepTrackNode` representing the division of the value
        produced by this node (`self`) by another node or value (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to divide by.

        Returns
        -------
        DeepTrackNode
            A new node that represents the division operation (`self / other`).
        
        """
        
        return _create_node_with_operator(operator.__truediv__, self, other)

    def __rtruediv__(
        self, 
        other: Union['DeepTrackNode', Any],
    ) -> 'DeepTrackNode':
        """Divide other value by node (right-hand).

        Creates a new `DeepTrackNode` representing the division of another
        node or value (`other`) by the value produced by this node (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to divide.

        Returns
        -------
        DeepTrackNode
            A new node that represents the division operation (`other / self`).
        
        """
        
        return _create_node_with_operator(operator.__truediv__, other, self)

    def __floordiv__(
        self, 
        other: Union['DeepTrackNode', Any],
    ) -> 'DeepTrackNode':
        """Perform floor division of node by another node or value.

        Creates a new `DeepTrackNode` representing the floor division of the
        value produced by this node (`self`) by another node or value 
        (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to divide by.

        Returns
        -------
        DeepTrackNode
            A new node that represents the floor division operation 
            (`self // other`).
        
        """
        
        return _create_node_with_operator(operator.__floordiv__, self, other)

    def __rfloordiv__(
        self, 
        other: Union['DeepTrackNode', Any],
    ) -> 'DeepTrackNode':
        """Perform floor division of other value by node (right-hand).

        Creates a new `DeepTrackNode` representing the floor division of 
        another node or value (`other`) by the value produced by this node 
        (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to divide.

        Returns
        -------
        DeepTrackNode
            A new node that represents the floor division operation 
            (`other // self`).
        
        """
        
        return _create_node_with_operator(operator.__floordiv__, other, self)

    def __lt__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if node is less than another node or value.

        Creates a new `DeepTrackNode` representing the comparison of this node
        (`self`) being less than another node or value (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation (`self < other`).
        
        """
        
        return _create_node_with_operator(operator.__lt__, self, other)

    def __rlt__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if other value is less than node (right-hand).

        Creates a new `DeepTrackNode` representing the comparison of another
        node or value (`other`) being less than this node (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to compare.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation 
            (`other < self`).
        
        """
        
        return _create_node_with_operator(operator.__lt__, other, self)

    def __gt__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if node is greater than another node or value.

        Creates a new `DeepTrackNode` representing the comparison of this node
        (`self`) being greater than another node or value (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation 
            (`self > other`).
        
        """
        
        return _create_node_with_operator(operator.__gt__, self, other)

    def __rgt__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if other value is greater than node (right-hand).

        Creates a new `DeepTrackNode` representing the comparison of another
        node or value (`other`) being greater than this node (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to compare.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation 
            (`other > self`).
        
        """
        
        return _create_node_with_operator(operator.__gt__, other, self)

    def __le__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if node is less than or equal to another node or value.

        Creates a new `DeepTrackNode` representing the comparison of this node
        (`self`) being less than or equal to another node or value (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation 
            (`self <= other`).
        
        """
        
        return _create_node_with_operator(operator.__le__, self, other)

    def __rle__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if other value is less than or equal to node (right-hand).

        Creates a new `DeepTrackNode` representing the comparison of another
        node or value (`other`) being less than or equal to this node (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to compare.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation 
            (`other <= self`).
        
        """
        
        return _create_node_with_operator(operator.__le__, other, self)

    def __ge__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if node is greater than or equal to another node or value.

        Creates a new `DeepTrackNode` representing the comparison of this node
        (`self`) being greater than or equal to another node or value 
        (`other`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The node or value to compare with.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation 
            (`self >= other`).
        
        """
        
        return _create_node_with_operator(operator.__ge__, self, other)

    def __rge__(self, other: Union['DeepTrackNode', Any]) -> 'DeepTrackNode':
        """Check if other value is greater than or equal to node (right-hand).

        Creates a new `DeepTrackNode` representing the comparison of another
        node or value (`other`) being greater than or equal to this node 
        (`self`).

        Parameters
        ----------
        other : DeepTrackNode or Any
            The value or node to compare.

        Returns
        -------
        DeepTrackNode
            A new node that represents the comparison operation 
            (`other >= self`).
        
        """
        
        return _create_node_with_operator(operator.__ge__, other, self)


def _equivalent(a, b):
    """Check if two objects are equivalent (internal function).

    This is a basic implementation to determine equivalence between two 
    objects. Additional cases can be implemented as needed to refine this 
    behavior.

    Parameters
    ----------
    a : Any
        The first object to compare.
    b : Any
        The second object to compare.

    Returns
    -------
    bool
        `True` if the objects are equivalent, `False` otherwise.

    Notes
    -----
    - If `a` and `b` are the same object (identity check), they are considered 
      equivalent.
    - If both `a` and `b` are empty lists, they are considered equivalent.
    """
    
    if a is b:
        return True

    if isinstance(a, list) and isinstance(b, list):
        return len(a) == 0 and len(b) == 0

    return False


def _create_node_with_operator(op, a, b):
    """Create a new computation node using a given operator and operands.

    This internal helper function constructs a `DeepTrackNode` that represents 
    the application of the specified operator to two operands. If the operands 
    are not already `DeepTrackNode` instances, they are converted to nodes.

    Parameters
    ----------
    op : Callable
        The operator function to apply.
    a : Any
        First operand. If not a `DeepTrackNode`, it will be wrapped in one.
    b : Any
        Second operand. If not a `DeepTrackNode`, it will be wrapped in one.

    Returns
    -------
    DeepTrackNode
        A new `DeepTrackNode` that applies the operator `op` to the values of 
        nodes `a` and `b`.

    Notes
    -----
    - This function establishes bidirectional relationships between the new 
        node and its operands:
        - The new node is added as a child of both `a` and `b`.
        - The operands are added as dependencies of the new node.
    - The operator `op` is applied lazily, meaning it will be evaluated when 
        the new node is called.

    """
    
    # Ensure `a` is a `DeepTrackNode`. Wrap it if necessary.
    if not isinstance(a, DeepTrackNode) and callable(a):
        a = DeepTrackNode(a)
    else:
        raise TypeError("Operand 'a' must be callable or a DeepTrackNode, "
                        + f"got {type(a).__name__}.")

    # Ensure `b` is a `DeepTrackNode`. Wrap it if necessary.
    if not isinstance(b, DeepTrackNode) and callable(b):
        b = DeepTrackNode(b)
    else:
        raise TypeError("Operand 'b' must be callable or a DeepTrackNode, "
                        + f"got {type(b).__name__}.")

    # New node that applies the operator `op` to the outputs of `a` and `b`.
    new_node = DeepTrackNode(lambda _ID=(): op(a(_ID=_ID), b(_ID=_ID)))
    
    # Establish dependency relationships between the nodes.
    new_node.add_dependency(a)
    new_node.add_dependency(b)
    
    # Set the new node as a child of both `a` and `b`.
    a.add_child(new_node)
    b.add_child(new_node)

    return new_node