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
   - Includes helper functions like `equivalent` and `create_node_with_operator` to streamline graph operations.

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
from typing import Any

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

    """Stores multiple data objects indexed by an access id.

    The purpose of this class is to allow a single object to store multiple
    data objects at once. This is necessary for sequences and the feature `Repeat`.

    The access id is a tuple of integers. Consider the following example::

        F = Repeat(
            Repeat(DummyFeature(prop = np.random.rand), 2),
            2
        )

    `F` contains 2*2=4 instances of the feature prop. They would be accessed using the IDs
    (0, 0), (0, 1), (1, 0), and (1, 1). In this way nested structures are resolved.

    The default is an empty tuple.

    All IDs of a DataDict need to be of the same length. If a DataDict has value stored to the None ID, it can not store any other IDs.
    If a longer ID is requested than what is stored, the request is trimmed to the length of the stored IDs. This is important to
    correctly handle dependencies to lower depths of nested structures.

    If a shorter ID is requested than what is stored, the a slice of the DataDict is returned with all IDs matching the request.



    """

    def __init__(self):
        self.keylength = None
        self.dict = {}

    def invalidate(self):
        # self.dict = {}
        for d in self.dict.values():
            d.invalidate()

    def validate(self):
        for d in self.dict.values():
            d.validate()

    def valid_index(self, _ID):
        assert isinstance(_ID, tuple), f"Data index {_ID} is not a tuple"

        if self.keylength is None:
            # If keylength has not yet been set, all indexes are valid
            return True

        if _ID in self.dict:
            # If index is a key, always valid
            return True

        # Otherwise, check key is correct length
        return len(_ID) == self.keylength

    def create_index(self, _ID=()):

        assert isinstance(_ID, tuple), f"Data index {_ID} is not a tuple"

        if _ID in self.dict:
            return

        assert self.valid_index(_ID), f"{_ID} is not a valid index for dict {self.dict}"

        if self.keylength is None:
            self.keylength = len(_ID)

        self.dict[_ID] = DeepTrackDataObject()

    def __getitem__(self, _ID):
        assert isinstance(_ID, tuple), f"Data index {_ID} is not a tuple"

        if self.keylength is None:
            raise KeyError("Indexing an empty dict")
        
        if len(_ID) == self.keylength:
            return self.dict[_ID]

        elif len(_ID) > self.keylength:
            return self[_ID[: self.keylength]]

        else:
            return {k: v for k, v in self.dict.items() if k[: len(_ID)] == _ID}

    def __contains__(self, _ID):
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
           self.is_valid(_ID=_ID) and equivalent(value, self.data[_ID].current_value())
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

    # node-node operators
    def __add__(self, other):
        return create_node_with_operator(operator.__add__, self, other)

    def __radd__(self, other):
        return create_node_with_operator(operator.__add__, other, self)

    def __sub__(self, other):
        return create_node_with_operator(operator.__sub__, self, other)

    def __rsub__(self, other):
        return create_node_with_operator(operator.__sub__, other, self)

    def __mul__(self, other):
        return create_node_with_operator(operator.__mul__, self, other)

    def __rmul__(self, other):
        return create_node_with_operator(operator.__mul__, other, self)

    def __truediv__(self, other):
        return create_node_with_operator(operator.__truediv__, self, other)

    def __rtruediv__(self, other):
        return create_node_with_operator(operator.__truediv__, other, self)

    def __floordiv__(self, other):
        return create_node_with_operator(operator.__floordiv__, self, other)

    def __rfloordiv__(self, other):
        return create_node_with_operator(operator.__floordiv__, other, self)

    def __lt__(self, other):
        return create_node_with_operator(operator.__lt__, self, other)

    def __rlt__(self, other):
        return create_node_with_operator(operator.__lt__, other, self)

    def __gt__(self, other):
        return create_node_with_operator(operator.__gt__, self, other)

    def __rgt__(self, other):
        return create_node_with_operator(operator.__gt__, other, self)

    def __le__(self, other):
        return create_node_with_operator(operator.__le__, self, other)

    def __rle__(self, other):
        return create_node_with_operator(operator.__le__, other, self)

    def __ge__(self, other):
        return create_node_with_operator(operator.__ge__, self, other)

    def __rge__(self, other):
        return create_node_with_operator(operator.__ge__, other, self)


def equivalent(a, b):
    # This is a bare-bones implementation to check if two objects are equivalent.
    # We can implement more cases to reduce updates.

    if a is b:
        return True

    if isinstance(a, list) and isinstance(b, list):
        return len(a) == 0 and len(b) == 0

    return False


def create_node_with_operator(op, a, b):
    """Creates a new node with the given operator and operands."""

    if not isinstance(a, DeepTrackNode):
        a = DeepTrackNode(a)

    if not isinstance(b, DeepTrackNode):
        b = DeepTrackNode(b)

    new = DeepTrackNode(lambda _ID=(): op(a(_ID=_ID), b(_ID=_ID)))
    new.add_dependency(a)
    new.add_dependency(b)
    a.add_child(new)
    b.add_child(new)

    return new