import random
from typing import Any
import numpy as np
import itertools
from deeptrack.backend.core import DeepTrackNode
import weakref
import functools

class SourceDeepTrackNode(DeepTrackNode):
    """A node that creates child nodes when attributes are accessed.
    
    This class is used to create a node that creates child nodes when 
    attributes are accessed. Assumes the value of the node is dict-like
    (i.e. has a __getitem__ method that takes a string).

    Example:
    >>> node = SourceDeepTrackNode(lambda: {"a": 1, "b": 2})
    >>> child = node.a
    >>> child() # returns 1

    Parameters
    ----------
    action : callable
        The action that returns the value of the node.    
    """

    def __getattr__(self, name):
        node = SourceDeepTrackNode(lambda: self()[name])
        node.add_dependency(self)
        self.add_child(node)
        return node

class SourceItem(dict):
    """ A dict-like object that calls a list of callbacks when called.

    Used in conjunction with the Source class to call a list of callbacks
    when called. These callbacks are used to activate a certain item
    in the source, ensuring all DeepTrackNodes are updated.

    Example:
    >>> source = Source(a=[1, 2], b=[3, 4])
    >>> @source.on_activate
    >>> def callback(item):
    >>>     print(item)
    >>> source[0]() # prints SourceItem({'a': 1, 'b': 3})

    Parameters
    ----------
    callbacks : list
        A list of callables that are called when the SourceItem is called.
    """

    def __init__(self, callbacks, **kwargs):
        self._callbacks = callbacks
        super().__init__(**kwargs)

    def __call__(self):
        for callback in self._callbacks:
            callback(self)
        return self
    
    def __repr__(self):
        return f"SourceItem({super().__repr__()})"

class Source:
    """ A class that represents one or more sources of data.

    This class is used to represent one or more sources of data.
    When accessed, it returns a deeptrack object that can be passed
    as properties to features.

    The feature can then be called with an item from the source to get the 
    value of the feature for that item. 

    Example:
    >>> source = Source(a=[1, 2], b=[3, 4])
    >>> feature_a = dt.Value(source.a)
    >>> feature_b = dt.Value(source.b)
    >>> sum_feature = feature_a + feature_b
    >>> sum_feature(source[0]) # returns 4
    >>> sum_feature(source[1]) # returns 6

    Parameters
    ----------
    kwargs : dict
        A dictionary of lists or arrays. The keys of the dictionary are
        the names of the sources, and the values are the sources themselves.
    """
    
    def __init__(self, **kwargs):
        self.validate_all_same_length(kwargs)
        self._length = len(kwargs[list(kwargs.keys())[0]])
        self._current_index = DeepTrackNode(0)
        self._dict = kwargs
        self._callbacks = set()

        for k in kwargs:
            setattr(self, k, self._wrap(k))
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._get_slice(index)
        else:
            return self._get_item(index)

    def product(self, **kwargs):
        """Return the product of the source with the given sources.

        Returns a source that is the product of th
        source with the given sources.

        Example:
        >>> source = Source(a=[1, 2], b=[3, 4])
        >>> new_source = source.product(c=[5, 6])
        >>> new_source # returns Source(c=[5, 6, 5, 6],
                                        a=[1, 1, 2, 2],
                                        b=[3, 3, 4, 4]
                                    )

        Parameters
        ----------
        kwargs : dict
            A dictionary of lists or arrays.
            The keys of the dictionary are the names of the sources,
            and the values are the sources themselves.
        """
        return Product(self, **kwargs)
    
    def constants(self, **kwargs):
        """Return a new source where the given values are constant.

        Example:
        >>> source = Source(a=[1, 2], b=[3, 4])
        >>> new_source = source.constants(c=5)
        >>> new_source # returns Source(c=[5, 5], a=[1, 2], b=[3, 4])

        Parameters
        ----------
        kwargs : dict
            A dictionary of values. The keys of the dictionary are the
            names of the sources, and the values are the values themselves.
        """
        return Product(self, **{k: [v] for k, v in kwargs.items()}) 
    
    def filter(self, predicate):
        """Return a new source with only the items that satisfy the predicate.

        Example:
        >>> source = Source(a=[1, 2], b=[3, 4])
        >>> new_source = source.filter(lambda a, b: a > 1)
        >>> new_source # returns Source(a=[2], b=[4])
        """
        indices = [i for i, item in enumerate(self) if predicate(**item)]
        return Subset(self, indices)


    def validate_all_same_length(self, kwargs):
        lengths = [len(v) for v in kwargs.values()]
        if not all([l == lengths[0] for l in lengths]):
            raise ValueError("All sources must have the same length.")


    def _wrap(self, key):
        value = self._dict[key]
        if hasattr(value, "__getitem__"):
            return self._wrap_indexable(key)
        
        return self._wrap_iterable(key)

    def _wrap_indexable(self, key):
        value_getter = SourceDeepTrackNode(lambda: self._dict[key][self._current_index()])
        value_getter.add_dependency(self._current_index)
        self._current_index.add_child(value_getter)
        return value_getter
    
    def _wrap_iterable(self, key):
        value_getter = SourceDeepTrackNode(
            lambda: list(self._dict[key])[self._current_index()]
            )
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
        callbacks = list(self._callbacks)
        callbacks.append(lambda _: self.set_index(index))
        return SourceItem(callbacks, **values)

    def _get_slice(self, slice):

        # Convert slice to list of indices.
        indices = list(range(*slice.indices(len(self))))

        # Get values for each index.
        return [self[i] for i in indices]
    
    def on_activate(self, callback: callable):
        self._callbacks.add(callback)

class Product(Source):
    """Class that represents the product of a source with one or more sources.

    This class is used to represent the product of a source with
    one or more sources. When accessed, it returns a deeptrack object that
    can be passed as properties to features.

    The feature can then be called with an item from the source
    to get the value of the feature for that item.
    """

    def __init__(self, __source=[{}], **kwargs):

        product = itertools.product(__source, *kwargs.values())

        dict_of_lists = {k: [] for k in kwargs.keys()}
        source_dict = {k: [] for k in __source[0].keys()}
        
        # if overlapping keys, error
        if set(kwargs.keys()).intersection(set(source_dict.keys())):
            raise ValueError(f"Overlapping keys in product. Duplicate keys: {set(kwargs.keys()).intersection(set(source_dict.keys()))}")

        dict_of_lists.update(source_dict)

        for source, *items in product:
            for k, v in source.items():
                dict_of_lists[k].append(v)
            for k, v in zip(kwargs.keys(), items):
                dict_of_lists[k].append(v)

        super().__init__(**dict_of_lists)    


class Subset(Source):

    def __init__(self, source, indices):
        self.source = source
        self.indices = indices
        self._dict = {k: [v[i] for i in indices]
                      for k, v in source._dict.items()}

    def __iter__(self):
        for i in self.indices:
            yield self.source[i]
    
    def __getitem__(self, index):
        return self.source[self.indices[index]]

    def __len__(self):
        return len(self.indices)
    
    def __getattr__(self, name):
        return getattr(self.source, name)



class Sources:
    """Joins multiple sources into a single access point.

    Used when one of multiple sources can be passed to a feature.
    For example the sources are split into training and validation sets,
    and the user can choose which one to use.

    Example:
    >>> source1 = Source(a=[1, 2], b=[3, 4])
    >>> source2 = Source(a=[5, 6], b=[7, 8])
    >>> joined_source = Sources(source1, source2)
    >>> feature_a = dt.Value(joined_source.a)
    >>> feature_b = dt.Value(joined_source.b)
    >>> sum_feature = feature_a + feature_b
    >>> sum_feature(source1[0]) # returns (1 + 3) = 4
    >>> sum_feature(source2[0]) # returns (5 + 7) = 12

    Parameters
    ----------
    sources : Source
        The sources to join.
    """

    def __init__(self, *sources: Source):
        self.sources = sources

        keys = set()
        for source in sources:
            keys.update(source._dict.keys())
        
        self._dict = dict.fromkeys(keys)

        for key in keys:
            node = SourceDeepTrackNode(functools.partial(lambda key: self._dict[key], key))

            setattr(self, key, node)

        for source in sources:
            source.on_activate(self._callback)

    def _callback(self, item):
        for key in item:
            getattr(self, key).invalidate()
            getattr(self, key).set_value(item[key])

Join = Sources

def random_split(source, lengths, generator=np.random.default_rng()):
    """Randomly split source into non-overlapping new sources of given lengths.

    Parameters
    ----------
    source : Source
        The source to split.
    lengths : list of int or float
        The lengths of the new sources. If the lengths are floats,
        they are interpreted as fractions of the source.
    generator : numpy.random.Generator, optional
        The random number generator to use.
    """

    import math 
    import warnings
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(
                    f"Fraction at index {i} is not between 0 and 1"
                    )
            n_items_in_split = int(
                math.floor(len(source) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(source) - sum(subset_lengths)  # type: ignore[arg-type]
        
        # Add 1 to all the lengths in round-robin fashion
        #  until the remainder is 0.
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                                f"This might result in an empty source.")
                
        # Cannot verify that dataset is Sized
    if sum(lengths) != len(source):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not\
                          equal the length of the input dataset!")

    indices = generator.permutation(
        sum(lengths)).tolist()  # type: ignore[call-overload]
    return [Subset(source, indices[offset - length : offset])\
            for offset, length in zip(_accumulate(lengths), lengths)]

        

def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total
