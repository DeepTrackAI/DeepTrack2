from copy import copy, deepcopy
import re
from weakref import WeakSet
import numpy as np
import operator

from .. import utils, image
from .citations import deeptrack_bibtex


class DeepTrackDataObject:

    """Atomic data container for deeptrack.

    The purpose of this is to store some data, and if that data is valid.
    Data is not valid, if some dependency of the data has been changed or otherwise made invalid
    since the last time the data was validated.
    """

    def __init__(self):
        self.data = None
        self.valid = False

    def store(self, data):
        self.valid = True
        self.data = data

    def current_value(self):
        return self.data

    def is_valid(self):
        return self.valid

    def invalidate(self):
        self.valid = False

    def validate(self):
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

        elif len(_ID) > self.keylength:
            return self[_ID[: self.keylength]]

        elif len(_ID) < self.keylength:
            return {k: v for k, v in self.dict.items() if k[: len(_ID)] == _ID}

        else:
            return self.dict[_ID]

    def __contains__(self, _ID):
        return _ID in self.dict


class DeepTrackNode:
    """Object corresponding to a node in a computation graph.

    This is a base class for all nodes in a computation graph. It is used to store and compute data.
    When evaluated, the node will call the `action` method. The action method defines a way to calculate the next data.
    If the data is already present, it will not be recalculated.
    """

    __nonelike_default = object()

    citation = deeptrack_bibtex

    def __init__(self, action=__nonelike_default, **kwargs):
        self.data = DeepTrackDataDict()
        self.children = WeakSet()
        self.dependencies = WeakSet()

        if action is not self.__nonelike_default:
            if callable(action):
                self.action = action
            else:
                self.action = lambda: action

        super().__init__(**kwargs)

    def add_child(self, other):
        self.children.add(other)

        return self

    def add_dependency(self, other):
        self.dependencies.add(other)

        return self

    def store(self, data, _ID=()):

        self.data.create_index(_ID)
        self.data[_ID].store(data)

        return self

    def is_valid(self, _ID=()):
        try:
            return self.data[_ID].is_valid()
        except KeyError:
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
        return self.data[_ID].current_value()

    def recurse_children(self, memory=None):
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
        """Gets a set of citations for all objects in a pipeline."""
        cites = {self.citation}
        for dep in self.recurse_dependencies():
            for obj in type(dep).mro():
                if hasattr(obj, "citation"):
                    cites.add(obj.citation)

        return cites

    def __call__(self, _ID=()):

        if self.is_valid(_ID):
            try:
                return self.current_value(_ID)
            except KeyError:
                pass

        new_value = utils.safe_call(self.action, _ID=_ID)
        self.store(new_value, _ID=_ID)
        return self.current_value(_ID)

    def current_value(self, _ID=()):
        return self.data[_ID].current_value()

    def __hash__(self):
        return id(self)

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
