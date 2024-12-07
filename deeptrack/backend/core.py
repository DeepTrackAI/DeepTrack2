"""DeepTrack2 core package.

This package provides the core classes and functionality required for managing, 
processing, and evaluating data within the DeepTrack framework. It is designed 
to support building flexible pipelines for scientific data analysis and machine 
learning applications.

Main Features
-------------
1. **Data Management**:
   - `DeepTrackDataObject` and `DeepTrackDataDict` provide tools to store, validate, and manage data with dependency tracking.
   - Enables nested structures and flexible indexing for complex data hierarchies.

2. **Computational Graphs**:
   - `DeepTrackNode` forms the backbone of computation pipelines, representing nodes in a computation graph.
   - Nodes support lazy evaluation, dependency tracking, and caching to optimize performance.
   - Implements mathematical operators for easy composition of computational graphs.

3. **Citations**:
   - Supports citing the relevant publication (`Midtvedt et al., 2021`) to ensure proper attribution.

4. **Utilities**:
   - Includes helper functions like `_equivalent` and `_create_node_with_operator` to streamline graph operations.

Package Structure
-----------------
- **Data Containers**:
  - `DeepTrackDataObject`: A basic container for data with validation status.
  - `DeepTrackDataDict`: Stores multiple data objects indexed by unique access IDs, enabling nested data storage.

- **Computation Nodes**:
  - `DeepTrackNode`: Represents a node in a computation graph, capable of lazy evaluation, caching, and dependency management.

- **Citation Management**:
  - Provides support for including citations in pipelines for academic and scientific use.

- **Utilities**:
  - Functions for equivalence checking and operator-based node creation simplify working with computation graphs.

Dependencies
------------
- `numpy`: Provides efficient numerical operations.
- `operator`: Enables operator overloading for computation nodes.
- `weakref.WeakSet`: Manages relationships between nodes without creating circular dependencies.

Usage
-----
This package is the core component of the DeepTrack framework. It enables users to:
- Construct flexible and efficient computational pipelines.
- Manage data and dependencies in a hierarchical structure.
- Perform lazy evaluations for performance optimization.

Example
-------
```python
# Create a DeepTrackNode with an action
node = DeepTrackNode(lambda x: x**2)
node.store(5)

# Retrieve the stored value
print(node.current_value())  # Output: 25

"""

import operator
from weakref import WeakSet

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

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

    The default access ID is an empty tuple `()`. The length of the IDs stored 
    must be consistent once an entry is created. If an ID longer than the 
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
    valid_index(_ID : tuple) -> bool
        Checks if the given ID is valid for the current configuration.
    create_index(_ID : tuple = ())
        Creates an entry for the given ID if it does not exist.
    __getitem__(_ID : tuple) -> DeepTrackDataObject' 
                                | Dict[Tuple[int, ...], 'DeepTrackDataObject']
        Retrieves data associated with the ID. Can return a 
        `DeepTrackDataObject` or a dict of matching entries if `_ID` is shorter 
        than `keylength`.
    __contains__(_ID : tuple) -> bool
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
    ) -> Union['DeepTrackDataObject', 
               Dict[Tuple[int, ...], 'DeepTrackDataObject']]:
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

    This is a base class for all nodes in a computation graph. It is used to store and compute data.
    When evaluated, the node will call the `action` method. The action method defines a way to calculate the next data.
    If the data is already present, it will not be recalculated.
    """

    __nonelike_default = object()

    citations = [citation_Midtvet2021Quantitative]

    @property
    def action(self):
        return self._action
    
    @action.setter
    def action(self, value):
        self._action = value
        self._accepts_ID = utils.get_kwarg_names(value).__contains__("_ID")

    def __init__(self, action=__nonelike_default, **kwargs):
        self.data = DeepTrackDataDict()
        self.children = WeakSet()
        self.dependencies = WeakSet()
        self._action = lambda: None

        if action is not self.__nonelike_default:
            if callable(action):
                self.action = action
            else:
                self.action = lambda: action

        self._accepts_ID = utils.get_kwarg_names(self.action).__contains__("_ID")
        super().__init__(**kwargs)

        self._all_subchildren = set()
        self._all_subchildren.add(self)

    def add_child(self, other):
        self.children.add(other)
        if not self in other.dependencies:
            other.add_dependency(self)
        
        subchildren = other._all_subchildren.copy()
        subchildren.add(other)

        self._all_subchildren = self._all_subchildren.union(subchildren)
        for parent in self.recurse_dependencies():
            parent._all_subchildren = parent._all_subchildren.union(subchildren)

        return self

    def add_dependency(self, other):
        self.dependencies.add(other)
        other.add_child(self)

        return self

    def store(self, data, _ID=()):

        self.data.create_index(_ID)
        self.data[_ID].store(data)

        return self

    def is_valid(self, _ID=()):
        try:
            return self.data[_ID].is_valid()
        except (KeyError, AttributeError):
            return False

    def valid_index(self, _ID):

        return self.data.valid_index(_ID)

    def invalidate(self, _ID=()):
        for child in self.recurse_children():
            child.data.invalidate()

        return self

    def validate(self, _ID=()):
        for child in self.recurse_children():
            try:
                child.data[_ID].validate()
            except KeyError:
                pass

        return self

    def _update(self):
        # Pre-instantiate memory for optimization
        child_memory = []

        for dependency in self.recurse_dependencies():
            for dep_child in dependency.recurse_children(memory=child_memory):
                dep_child.data = DeepTrackDataDict()

        return self

    def set_value(self, value, _ID=()):

        # If set to same value, no need to invalidate

        if not (
           self.is_valid(_ID=_ID) and _equivalent(value, self.data[_ID].current_value())
        ):
            self.invalidate(_ID=_ID)
            self.store(value, _ID=_ID)

        return self

    def previous(self, _ID=()):
        if self.data.valid_index(_ID):
            return self.data[_ID].current_value()
        else:
            return []

    def recurse_children(self, memory=set()):
        return self._all_subchildren

    def old_recurse_children(self, memory=None):
        # On first call, instantiate memory
        if memory is None:
            memory = []

        # Make sure each DeepTrackNode is only yielded once
        if self in memory:
            return

        # Remember self
        memory.append(self)

        # Yield self and recurse children
        yield self

        for child in self.children:
            yield from child.recurse_children(memory=memory)

    def recurse_dependencies(self, memory=None):
        # On first call, instantiate memory
        if memory is None:
            memory = []

        # Make sure each DeepTrackNode is only yielded once
        if self in memory:
            return

        # Remember self
        memory.append(self)

        # Yield self and recurse dependencies
        yield self

        for dependency in self.dependencies:
            yield from dependency.recurse_dependencies(memory=memory)

    def get_citations(self):
        """Get citations for all objects in a pipeline."""

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

    def __call__(self, _ID=()):

        if self.is_valid(_ID):
            try:
                return self.current_value(_ID)
            except KeyError:
                pass
        
        if self._accepts_ID:
            new_value = self.action(_ID=_ID)
        else:
            new_value = self.action()

        self.store(new_value, _ID=_ID)
        return self.current_value(_ID)

    def current_value(self, _ID=()):
        return self.data[_ID].current_value()

    def __hash__(self):
        return id(self)

    def __getitem__(self, idx):
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

    def __add__(self, other):
        """Add node to other node or value.
        
        Operation performed:
            result = self + other
        """
        return _create_node_with_operator(operator.__add__, self, other)

    def __radd__(self, other):
        """Add other value to node (right-hand).
        
        Operation performed:
            result = other + self
        """
        return _create_node_with_operator(operator.__add__, other, self)

    def __sub__(self, other):
        """Subtract other node or value from node.
        
        Operation performed:
            result = self - other
        """
        return _create_node_with_operator(operator.__sub__, self, other)

    def __rsub__(self, other):
        """Subtract node from other value (right-hand).
        
        Operation performed:
            result = other - self
        """
        return _create_node_with_operator(operator.__sub__, other, self)

    def __mul__(self, other):
        """Multiply node by other node or value.
        
        Operation performed:
            result = self * other
        """
        return _create_node_with_operator(operator.__mul__, self, other)

    def __rmul__(self, other):
        """Multiply other value by node (right-hand).
        
        Operation performed:
            result = other * self
        """
        return _create_node_with_operator(operator.__mul__, other, self)

    def __truediv__(self, other):
        """Divide node by other node or value.
        
        Operation performed:
            result = self / other
        """
        return _create_node_with_operator(operator.__truediv__, self, other)

    def __rtruediv__(self, other):
        """Divide other value by node (right-hand).
        
        Operation performed:
            result = other / self
        """
        return _create_node_with_operator(operator.__truediv__, other, self)

    def __floordiv__(self, other):
        """Perform floor division of node by other node or value.
        
        Operation performed:
            result = self // other
        """
        return _create_node_with_operator(operator.__floordiv__, self, other)

    def __rfloordiv__(self, other):
        """Perform floor division of other value by node (right-hand).
        
        Operation performed:
            result = other // self
        """
        return _create_node_with_operator(operator.__floordiv__, other, self)

    def __lt__(self, other):
        """Check if node is less than other node or value.
        
        Operation performed:
            result = self < other
        """
        return _create_node_with_operator(operator.__lt__, self, other)

    def __rlt__(self, other):
        """Check if other value is less than node (right-hand).
        
        Operation performed:
            result = other < self
        """
        return _create_node_with_operator(operator.__lt__, other, self)

    def __gt__(self, other):
        """Check if node is greater than other node or value.
        
        Operation performed:
            result = self > other
        """
        return _create_node_with_operator(operator.__gt__, self, other)

    def __rgt__(self, other):
        """Check if other value is greater than node (right-hand).
        
        Operation performed:
            result = other > self
        """
        return _create_node_with_operator(operator.__gt__, other, self)

    def __le__(self, other):
        """Check if node is less than or equal to other node or value.
        
        Operation performed:
            result = self <= other
        """
        return _create_node_with_operator(operator.__le__, self, other)

    def __rle__(self, other):
        """Check if other value is less than or equal to node (right-hand).
        
        Operation performed:
            result = other <= self
        """
        return _create_node_with_operator(operator.__le__, other, self)

    def __ge__(self, other):
        """Check if node is greater than or equal to other node or value.
        
        Operation performed:
            result = self >= other
        """
        return _create_node_with_operator(operator.__ge__, self, other)

    def __rge__(self, other):
        """Check if other value is greater than or equal to node (right-hand).
        
        Operation performed:
            result = other >= self
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
    if not isinstance(a, DeepTrackNode):
        a = DeepTrackNode(a)

    # Ensure `b` is a `DeepTrackNode`. Wrap it if necessary.
    if not isinstance(b, DeepTrackNode):
        b = DeepTrackNode(b)

    # New node that applies the operator `op` to the outputs of `a` and `b`.
    new_node = DeepTrackNode(lambda _ID=(): op(a(_ID=_ID), b(_ID=_ID)))
    
    # Establish dependency relationships between the nodes.
    new_node.add_dependency(a)
    new_node.add_dependency(b)
    
    # Set the new node as a child of both `a` and `b`.
    a.add_child(new_node)
    b.add_child(new_node)

    return new_node