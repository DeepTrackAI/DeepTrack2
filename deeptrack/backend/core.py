from copy import copy, deepcopy
import re
import numpy as np
import operator

from .. import utils, image
from .citations import deeptrack_bibtex


class DeepTrackDataObject:

    """Atomic data container for deeptrack.

    The purpose of this is to store some data, and if that data is valid.
    Data is not valid, if some dependency of the data has been changed or otherwise made not valid
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

        # if self.default is not None:
        #     return self.default

        # if replicate_index is None:
        #     self.default = DeepTrackDataObject()
        #     return self.default

        # if isinstance(replicate_index, int):
        #     return self.list[replicate_index]

        # if isinstance(replicate_index, (tuple, list)):
        #     replicate_index, *rest = replicate_index

        #     if not rest:
        #         return self[replicate_index]

        #     while len(self.list) <= replicate_index:
        #         self.list.append(DeepTrackDataList())

        #     output = self.list[replicate_index]

        #     if isinstance(output, DeepTrackDataList):
        #         return output[rest]
        #     else:
        #         return output

        # raise NotImplementedError("Indexing with non-integer types not yet implemented")

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

    __nonelike_default = object()

    citation = deeptrack_bibtex

    def __init__(self, action=__nonelike_default, **kwargs):
        self.data = DeepTrackDataDict()
        self.children = []
        self.dependencies = []

        if action is not self.__nonelike_default:
            if callable(action):
                self.action = action
            else:
                self.action = lambda: action

        super().__init__(**kwargs)

    def add_child(self, other):
        self.children.append(other)

        return self

    def add_dependency(self, other):
        self.dependencies.append(other)

        return self

    def store(self, data, replicate_index=()):

        self.data.create_index(replicate_index)
        self.data[replicate_index].store(data)

        return self

    def is_valid(self, replicate_index=()):
        try:
            return self.data[replicate_index].is_valid()
        except KeyError:
            return False

    def valid_index(self, replicate_index):

        return self.data.valid_index(replicate_index)

    def invalidate(self, replicate_index=()):
        for child in self.recurse_children():
            child.data.invalidate()

        return self

    def validate(self, replicate_index=()):
        for child in self.recurse_children():
            try:
                child.data[replicate_index].validate()
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

    def set_value(self, value, replicate_index=()):

        # If set to same value, no need to invalidate

        if not (
            self.is_valid(replicate_index=replicate_index)
            and equivalent(value, self.data[replicate_index].current_value())
        ):
            self.invalidate(replicate_index=replicate_index)
            self.store(value, replicate_index=replicate_index)

        return self

    def previous(self, replicate_index=()):
        return self.data[replicate_index].current_value()

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
        cites = {self.citation}
        for dep in self.recurse_dependencies():
            for obj in type(dep).mro():
                if hasattr(obj, "citation"):
                    cites.add(obj.citation)

        return cites

    def __call__(self, replicate_index=()):

        if self.is_valid(replicate_index):
            try:
                return self.current_value(replicate_index)
            except KeyError:
                pass

        new_value = utils.safe_call(self.action, replicate_index=replicate_index)
        self.store(new_value, replicate_index=replicate_index)
        return self.current_value(replicate_index)

    def current_value(self, replicate_index=()):
        return self.data[replicate_index].current_value()

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
    if a is b:
        return True

    if isinstance(a, list) and isinstance(b, list):
        return len(a) == 0 and len(b) == 0

    return False
    try:

        if id(a) == id(b):
            return True

        # return False
        if type(a) != type(b):
            return False

        if isinstance(a, np.ndarray):
            # return False
            if a.shape != b.shape:
                return False
            return np.array_equal(a, b, equal_nan=True)

        eq = a == b
        try:
            # God this is stupid
            if eq:
                return eq
            else:
                return eq
        except ValueError as e:
            return np.array_equal(a, b, equal_nan=True)

    except:
        return False


def create_node_with_operator(op, a, b):

    if not isinstance(a, DeepTrackNode):
        a = DeepTrackNode(a)

    if not isinstance(b, DeepTrackNode):
        b = DeepTrackNode(b)

    new = DeepTrackNode(
        lambda replicate_index=(): op(
            a(replicate_index=replicate_index), b(replicate_index=replicate_index)
        )
    )
    new.add_dependency(a)
    new.add_dependency(b)
    a.add_child(new)
    b.add_child(new)

    return new
