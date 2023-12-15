import random
from typing import Any
import numpy as np
from deeptrack.backend.core import DeepTrackNode

class SourceItem(dict):

    def __init__(self, callback, **kwargs):
        self._callback = callback
        super().__init__(**kwargs)

    def __call__(self):
        return self._callback()
    
    def __repr__(self):
        return f"SourceItem({super().__repr__()})"

class Source:
    
    def __init__(self, **kwargs):
        self.validate_all_same_length(kwargs)
        self._length = len(kwargs[list(kwargs.keys())[0]])
        self._current_index = DeepTrackNode(0)
        self._dict = kwargs

        for k in kwargs.keys():
            setattr(self, k, self._wrap(k))
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._get_slice(index)
        else:
            return self._get_item(index)

    def validate_all_same_length(self, kwargs):
        lengths = [len(v) for v in kwargs.values()]
        if not all([l == lengths[0] for l in lengths]):
            raise ValueError("All sources must have the same length.")


    def _wrap(self, key):
        value = self._dict[key]
        if hasattr(value, "__getitem__"):
            return self._wrap_indexable(key)
        else:
            return self._wrap_iterable(key)

    def _wrap_indexable(self, key):
        value_getter = DeepTrackNode(lambda: self._dict[key][self._current_index()])
        value_getter.add_dependency(self._current_index)
        self._current_index.add_child(value_getter)
        return value_getter
    
    def _wrap_iterable(self, key):
        value_getter = DeepTrackNode(lambda: list(self._dict[key])[self._current_index()])
        value_getter.add_dependency(self._current_index)
        self._current_index.add_child(value_getter)
        return value_getter

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def set_index(self, index):
        self._current_index.set_value(index)
        return self

    def _get_item(self, index):
        values = {k: v[index] for k, v in self._dict.items()}
        return SourceItem(lambda: self.set_index(index), **values)

    def _get_slice(self, slice):

        # convert slice to list of indices
        indices = list(range(*slice.indices(len(self))))

        # get values for each index
        return [self[i] for i in indices]
        
         