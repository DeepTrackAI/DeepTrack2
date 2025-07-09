"""Utility classes for data sources.

This module provides a set of utility classes designed for managing and
manipulating data sources.

These tools are primarily used in scenarios where data needs to be dynamically
manipulated, filtered, or combined for feature generation in machine learning
pipelines.

Key Features
------------
- **Node Hierarchy**

    `SourceDeepTrackNode` extends `DeepTrackNode` with utilities to create
    nested nodes, and structured data access.
    
- **Dynamic Data Access**

    It retrieves data items as callable objects, supporting custom callbacks and dependency tracking.

- **Randomized Splitting**

    Enables splitting of data sources into non-overlapping
    subsets with user-specified length.
    

Module Structure
----------------
Classes:

- `SourceDeepTrackNode`: Creates child nodes when accessing attributes.

- `SourceItem`: Dict-like object that calls a list of callbacks when called.

- `Source`: Represents one or more sources of data.

- `Join`: Alias of `Source`.

- `Product`: Represents the product of the source with the given sources.

    This class is used to represent the product of a source with
    one or more sources. When accessed, it returns a deeptrack object that
    can be passed as properties to features.
        
- `Subset`: Represents the subset of a `Source`.

- `Sources`: Represents multiple sources as a single access point.

    Used when one of multiple sources can be passed to a feature.

Functions:

- `random_split(source, lengths, generator)`

    def random_split(
        source: Source,
        lengths: list[int or float],
        generator: np.random.Generator = np.random.default_rng()
    ) -> list[Subset]:
        Randomly split source into non-overlapping new sources of given lengths.

Examples
--------
Call a list of callbacks:

>>> from deeptrack.sources import Source

>>> source = Source(a=[1, 2], b=[3, 4])
>>> @source.on_activate
>>> def callback(item):
>>>     print(item)
>>> source[0]() 

Equivalent to:

>>> SourceItem({'a': 1, 'b': 3}).

Create a node that creates child nodes when attributes are accessed:

>>> from deeptrack.sources import SourceDeepTrackNode

>>> node = SourceDeepTrackNode(lambda: {"a": 1, "b": 2})
>>> child = node.a
>>> child()
1

Join multiple sources into a single access point:

>>> import deeptrack as dt
>>> from deeptrack.sources import Source

>>> source1 = Source(a=[1, 2], b=[3, 4])
>>> source2 = Source(a=[5, 6], b=[7, 8])
>>> joined_source = Sources(source1, source2)
>>> feature_a = dt.Value(joined_source.a)
>>> feature_b = dt.Value(joined_source.b)
>>> sum_feature = feature_a + feature_b

>>> sum_feature(source1[0])
4
>>> sum_feature(source2[0])
12

"""

from __future__ import annotations

import functools
import itertools
import math

from collections.abc import Sequence
from typing import Any, Callable, Generator

import numpy as np

from deeptrack.backend.core import DeepTrackNode


__all__ = [
    "Source",
    "SourceItem",
    "Product",
    "Subset",
    "Sources",
    "Join",
    "random_split",
]


class SourceDeepTrackNode(DeepTrackNode):
    """A node that creates child nodes when attributes are accessed.
    
    `SourceDeepTrackNode` is a subclass of `DeepTrackNode` designed to
    facilitate structured data access. When an attribute is accessed, it
    creates a new child node that retrieves the corresponding key from the
    underlying dictionary-like data.

    This is particularly useful when working with hierarchical or nested
    data sources, allowing intuitive access via attribute syntax (e.g.
    `source.position.x`) and automatic dependency tracking between nodes.
    
    It assumes the value of the node is dict-like (i.e., that it has a
    `__getitem__()` method that takes a string).

    Parameters
    ----------
    action: Callable[[...], Any]
        A callable that returns the value of the node. The return value
        must be a dictionary-like object supporting string-key indexing.

    Examples
    --------
    >>> from deeptrack.sources import SourceDeepTrackNode

    Basic usage with a dictionary-like source:
    >>> data = {"x": 42, "y": {"z": 3.14}}
    >>> source = SourceDeepTrackNode(lambda: data)
    >>> source.x()
    42

    >>> source.y()
    {'z': 3.14}

    >>> source.y.z()
    3.14

    """

    def __getattr__(
        self: SourceDeepTrackNode,
        name: str
    ) -> SourceDeepTrackNode:
        """Return a child node corresponding to a key in the underlying data.

        This method is triggered when an attribute is accessed and no
        explicitly defined attribute is found. It constructs a new
        `SourceDeepTrackNode` that retrieves the value associated with the
        given key from the parent node's dictionary-like output.

        The new node is registered as a dependent of the current node to ensure
        correct dependency tracking during evaluation.

        Parameters
        ----------
        name: str
            The key to retrieve from the dictionary-like data returned by
            `self()`.

        Returns
        -------
        SourceDeepTrackNode
            A new node that resolves to `self()[name]` when evaluated.

        Examples
        --------
        >>> from deeptrack.sources.base import SourceDeepTrackNode

        Basic usage with a dictionary-like source:
        >>> source = SourceDeepTrackNode(lambda: {"a": {"b": 1}})
        >>> source.a()
        {'b': 1}
    
        >>> source.a.b()
        1

        """

        node = SourceDeepTrackNode(lambda: self()[name])
        node.add_dependency(self)
        # self.add_child(node)
        return node


class SourceItem(dict):
    """A dict-like object that triggers a list of callbacks when called.

    `SourceItem` is used within the `Source` framework to wrap a dictionary
    entry that activates one or more callbacks when accessed via calling.
    This mechanism ensures that all dependent `DeepTrackNode`s are updated
    when a particular item in the source is selected.

    Parameters
    ----------
    callbacks: list[Callable[[Any], None]]
        A list of callback functions that are executed when the item is called.
        Each function receives the `SourceItem` itself as argument.

    Attributes
    ----------
    _callbacks : list[Callable[[SourceItem], None]]
        Internal list of callbacks that are triggered on call.

    Methods
    -------
    __call__() -> SourceItem
        Executes all callbacks and returns the item.

    __repr__() -> str
        Returns a string representation including the dictionary content
        and number of callbacks.

    Examples
    --------
    >>> from deeptrack.sources import SourceItem

    Implement a callback function:
    >>> def log_callback(item):
    ...     print(f"CALLBACK - Accessed item: {item}")

    Create a SourceItem with dictionary contents and callbacks:
    >>> item = dt.SourceItem(callbacks=[log_callback], a=1, b=2)

    Call the item to trigger the callbacks:
    >>> item();
    CALLBACK - Accessed item: SourceItem({'a': 1, 'b': 2}, 1 callback(s))

    """

    _callbacks: list[Callable[[Any], None]]

    def __init__(
        self: SourceItem,
        callbacks: list[Callable[[Any], None]],
        **kwargs: Any,
    ):
        """Initialize a SourceItem.

        Parameters
        ----------
        callbacks: list[Callable[[SourceItem], None]]
            The list of callbacks to trigger when the item is called.
        **kwargs: Any
            Additional key-value pairs stored in the dictionary.

        """

        self._callbacks = callbacks

        super().__init__(**kwargs)

    def __call__(
        self: SourceItem,
    ) -> SourceItem:
        """Call the item, triggering all associated callbacks.

        Returns
        -------
        SourceItem
            The item itself, after invoking all callbacks.

        """

        for callback in self._callbacks:
            callback(self)
        return self

    def __repr__(
        self: SourceItem,
    ) -> str:
        """Return a string representation of the item.

        Returns
        -------
        str
            The string representation of the dictionary contents.

        """

        return (f"SourceItem({super().__repr__()}, "
                f"{len(self._callbacks)} callback(s))")


class Source:
    """A class that represents one or more sources of data.

    `Source` holds one or more named sequences (e.g., lists, arrays) and
    makes them accessible by index. It returns `SourceItem` objects that
    activate registered callbacks (e.g., for dependency tracking) when called.

    Each named field is accessible as an attribute (e.g., `source.a`) and
    can be passed directly to DeepTrack features such as `Value`. Features
    can then be evaluated on specific items by indexing the source (e.g.,
    `feature(source[i])`).

    Parameters
    ----------
    *kwargs: Sequence[Any]
        Named data sources, where each key is the name of a source (e.g., "x",
        "label") and each value is an indexable sequence (e.g., list, NumPy
        array, PyTorch tensor). All sequences must have the same length and
        support integer indexing.

    Attributes
    ----------
    _dict: dict[str, Sequence[Any]]
        Internal mapping of source names to their corresponding data sequences.

    _length: int
        Number of items in the source. All fields must have the same length.

    _current_index: DeepTrackNode
        A node that holds the current active index. Used for dynamic access
        when a source attribute (e.g., `source.a`) is passed to a feature.

    _callbacks: set[Callable[[Any], None]]
        A set of callback functions triggered when a `SourceItem` is called.

    Methods #TODO ***GV*** check and complete list of methods
    -------
    __len__() -> int
        It returns the number of items in the source.

    __getitem__(index: int | slice) -> SourceItem or list[SourceItem]
        It retrieves one or more items by index or slice.

    _get_item(index: int) -> SourceItem:
        It retrieves a single SourceItem at a specified index.

    _get_slice(slice_obj: slice) -> list[SourceItem]:
        It retrieves a list of SourceItems corresponding to a slice.

    product(**kwargs: Sequence[Any]) -> Product
        It returns a new source representing the cartesian product of the
        current source with the given sequences.

    constants(**kwargs: Sequence[Any]) -> Product
        It returns a new source where the given values are treated as
        constants.

    filter(predicate: Callable[..., bool]) -> Subset
        It returns a new source containing only the items for which the
        predicate returns `True`.

    _validate_all_same_length(kwargs: dict[str, Sequence[Any]]) -> None:
        It validates that all input sequences have the same length.

    __iter__() -> Generator[SourceItem, None, None]
        It iterates over all items in the source.

    set_index(index) -> Source
        It sets the active index used when evaluating attributes, like in
        `source.a()`.

    on_activate(callback: Callable[[SourceItem], None]) -> None
        It registers a callback to be called when any item is activated.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.sources import Source

    Define a source with two fields:
    >>> source = Source(
    ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...     b=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    >>> )

    Create features from the source:
    >>> feature_a = dt.Value(source.a)
    >>> feature_b = dt.Value(source.b)
    >>> sum_feature = feature_a + feature_b

    Evaluate features on individual items:
    >>> sum_feature(source[0])
    11
    >>> sum_feature(source[8])
    99

    Filter items using a predicate:
    >>> filtered = source.filter(lambda a, b: a > 5 and b < 80)
    >>> list(filtered)
    [SourceItem({'a': 6, 'b': 60}, 1 callback(s)),
    SourceItem({'a': 7, 'b': 70}, 1 callback(s))]

    Slice the source:
    >>> subset = source[3:5]
    >>> subset
    [SourceItem({'a': 4, 'b': 40}, 1 callback(s)),
    SourceItem({'a': 5, 'b': 50}, 1 callback(s))]

    Add a constant field to the source:
    >>> augmented = source.constants(label="train")
    >>> augmented[0]["label"]
    'train'

    Take a Cartesian product with a new field:
    >>> extended = source.product(c=[100, 200])
    >>> len(extended)
    18  # 9 original items × 2 values in "c"
    >>> extended[0]["c"]
    100
    >>> extended[17]["c"]
    200

    Use set_index to manually select the active item:
    >>> source.set_index(1)
    >>> source.a()
    2
    >>> source.b()
    20

    Iterate over items in the source:
    >>> for item in source:
    ...     print(item["a"], item["b"])
    1 10
    2 20
    3 30
    4 40
    5 50
    6 60
    7 70
    8 80
    9 90

    """

    _dict: dict[str, Sequence[Any]]
    _length: int
    _current_index: DeepTrackNode
    _callbacks: set[Callable[[Any], None]]

    def __init__(
        self: Source,
        **kwargs: Sequence[Any],
    ):
        """Initialize a Source with one or more named data sequences.

        The input sequences must all have the same length and support integer
        indexing (i.e., implement both `__getitem__` and `__len__`). Each key
        becomes an attribute of the source and can be passed to DeepTrack
        features for dynamic evaluation.

        Parameters
        ----------
        **kwargs : Sequence[Any]
            Named data sources, where each key is the name of a field (e.g.,
            "x", "label") and each value is an indexable sequence (e.g., list,
            NumPy array, PyTorch tensor). All sequences must have the same
            length.

        Raises
        ------
        ValueError
            If the input sequences do not all have the same length.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source with two named sequences (note that they are of the
        same length):
        >>> source = Source(
        ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     b=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        >>> )

        Iterate over items in the source:
        >>> for item in source:
        ...     print(item["a"], item["b"])
        1 10
        2 20
        3 30
        4 40
        5 50
        6 60
        7 70
        8 80
        9 90

        """

        self._validate_all_same_length(kwargs)

        self._dict = kwargs
        self._length = len(kwargs[list(kwargs.keys())[0]])
        self._current_index = DeepTrackNode(0)
        self._callbacks = set()

        for k in kwargs:
            setattr(self, k, self._wrap(k))

    def __len__(
        self: Source,
    ) -> int:
        """Return the number of items in the source.

        This returns the number of indexed entries available in the source,
        which corresponds to the length of any of the underlying sequences.

        Returns
        -------
        int
            The number of items in the source.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source:
        >>> source = Source(a=[1, 2, 3], b=[10, 20, 30])

        Get its length:
        >>> len(source)
        3

        """

        return self._length

    def __getitem__(
        self: Source,
        index: int | slice,
    ) -> SourceItem | list[SourceItem]:
        """Retrieve one or more SourceItems by index or slice.

        If the input is an integer, this returns a single `SourceItem`
        at the specified index. If the input is a slice, it returns a list
        of `SourceItem`s corresponding to the slice range.

        Parameters
        ----------
        index: int or slice
            The index or slice specifying which item(s) to retrieve.

        Returns
        -------
        SourceItem or list[SourceItem]
            The item(s) corresponding to the given index or slice.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source:
        >>> source = Source(
        ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     b=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        ... )

        Retrieve a single item:
        >>> item = source[1]
        >>> item
        SourceItem({'a': 2, 'b': 20}, 1 callback(s))

        >>> item["a"]
        20

        >>> item["b"]
        2

        Retrieve a slice of items:
        >>> items = source[1:4]
        >>> items
        [SourceItem({'a': 2, 'b': 20}, 1 callback(s)),
        SourceItem({'a': 3, 'b': 30}, 1 callback(s)),
        SourceItem({'a': 4, 'b': 40}, 1 callback(s))]
 
        >>> [(item["a"], item["b"]) for item in items]
        [(2, 20), (3, 30), (4, 40)]

        """

        if isinstance(index, slice):
            return self._get_slice(index)
        else:
            return self._get_item(index)

    def _get_item(
        self: Source,
        index: int,
    ) -> SourceItem:
        """Retrieve a single SourceItem at a specified index.

        This method extracts the values at the given index from all fields
        in the source and wraps them in a `SourceItem`. It also attaches
        callbacks that are executed when the item is activated (i.e., called).

        The first callback sets the active index in the source to the given
        index. Additional callbacks come from those registered via the
        `on_activate()` method.

        Parameters
        ----------
        index: int
            The index of the item to retrieve.

        Returns
        -------
        SourceItem
            The item at the specified index, wrapped with activation callbacks.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source:
        >>> source = Source(a=[1, 2], b=[10, 20])

        Extract the source item corresponding to index 1:
        >>> item = source._get_item(1)
        >>> item
        SourceItem({'a': 2, 'b': 20}, 1 callback(s))

        Since the item has not been activated the current index of the source
        is still 0:
        >>> source._current_index()
        0

        Activate the item and sets the source's current index to 1:
        >>> item()
        >>> source._current_index()
        1

        """

        # Collect field values at the given index from all source sequences
        values = {k: v[index] for k, v in self._dict.items()}

        # Prepend the set_index callback so the active index is updated first
        callbacks = [lambda _: self.set_index(index)] + list(self._callbacks)

        return SourceItem(callbacks, **values)

    def _get_slice(
        self: Source,
        slice_obj: slice,
    ) -> list[SourceItem]:
        """Retrieve a list of SourceItems corresponding to a slice.

        This method returns a list of `SourceItem`s corresponding to the given
        slice object (e.g., `source[1:4]`). It converts the slice into a list
        of integer indices and uses `_get_item()` to retrieve each item.

        Parameters
        ----------
        slice_obj: slice
            A slice object representing the range of indices to retrieve.

        Returns
        -------
        list[SourceItem]
            A list of SourceItems corresponding to the selected range.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source:
        >>> source = Source(
        ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     b=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        ... )

        Get a slice of the source:
        >>> source[1:4]
        [SourceItem({'a': 2, 'b': 20}, 1 callback(s)),
        SourceItem({'a': 3, 'b': 30}, 1 callback(s)),
        SourceItem({'a': 4, 'b': 40}, 1 callback(s))]

        This is equivalent to:
        >>> source._get_slice(slice(1, 4))
        """

        # Convert the slice to a list of indices
        indices = list(range(*slice_obj.indices(len(self))))

        # Get values for each index using _get_item()
        return [self[i] for i in indices]

    def product(
        self: Source,
        **kwargs: Sequence[Any],
    ) -> Product:
        """Return the product of the source with the given sources.

        Returns a source that is the product of th
        source with the given sources.

        Example
        -------
        >>> from deeptrack.sources import Source
        
        >>> source = Source(a=[1, 2], b=[3, 4])
        >>> new_source = source.product(c=[5, 6])
        >>> new_source 
        Source(c=[5, 6, 5, 6],
               a=[1, 1, 2, 2],
               b=[3, 3, 4, 4]
        )

        Parameters
        ----------
        kwargs: dict
            A dictionary of lists or arrays.
            The keys of the dictionary are the names of the sources,
            and the values are the sources themselves.
            
        """
        return Product(self, **kwargs)

    def constants(
        self: Source,
        **kwargs: Sequence[Any],
    ) -> Product:
        """Return a new source where the given values are constant.

        Example
        -------
        from deeptrack.sources import Source
        
        >>> source = Source(a=[1, 2], b=[3, 4])
        >>> new_source = source.constants(c=5)
        >>> new_source
        Equivalent to:
        >>> Source(c=[5, 5], a=[1, 2], b=[3, 4]).

        Parameters
        ----------
        kwargs: dict
            A dictionary of values. The keys of the dictionary are the
            names of the sources, and the values are the values themselves.
            
        """
        return Product(self, **{k: [v] for k, v in kwargs.items()})

    def filter(
        self: Source,
        predicate: Callable[..., bool],
    ) -> Subset:
        """Return a new source with only the items that satisfy the predicate.

        Example
        -------
        >>> from deeptrack.sources import Source
        
        >>> source = Source(a=[1, 2], b=[3, 4])
        >>> new_source = source.filter(lambda a, b: a > 1)
        >>> new_source
        Equivalent to:
        >>> Source(a=[2], b=[4]).
        
        """
        indices = [i for i, item in enumerate(self) if predicate(**item)]
        return Subset(self, indices)

    def _validate_all_same_length(
        self: Source,
        kwargs: dict[str, Sequence[Any]],
    ) -> None:
        """Validate that all input sequences have the same length.

        This method checks that all sequences provided to the source have equal
        length. It is called during initialization to ensure consistent
        indexing behavior.

        Parameters
        ----------
        kwargs: dict[str, Sequence[Any]]
            Dictionary of named sequences to validate.

        Raises
        ------
        ValueError
            If the sequences do not all have the same length.

        Examples
        --------
        >>> from deeptrack.sources import Source

        This works:
        >>> source = Source(a=[1, 2, 3], b=[10, 20, 30])

        This raises a ValueError:
        >>> source = Source(a=[1, 2], b=[10, 20, 30])

        """

        lengths = [len(v) for v in kwargs.values()]
        unique_lengths = set(lengths)

        if len(unique_lengths) > 1:
            raise ValueError(
                "All sources must have the same length, but the following "
                f"lengths were found: {lengths}"
            )

    def _wrap_indexable(
        self: Source,
        key: str,
    ) -> SourceDeepTrackNode:
        value_getter = SourceDeepTrackNode(
            lambda: self._dict[key][self._current_index()]
        )
        value_getter.add_dependency(self._current_index)
        self._current_index.add_child(value_getter)
        return value_getter

    def _wrap(
        self: Source,
        key: str,
    ) -> SourceDeepTrackNode:
        value = self._dict[key]
        if hasattr(value, "__getitem__"):
            return self._wrap_indexable(key)

        return self._wrap_iterable(key)

    def _wrap_iterable(
        self: Source,
        key: str,
    ) -> SourceDeepTrackNode:
        value_getter = SourceDeepTrackNode(
            lambda: list(self._dict[key])[self._current_index()]
            )
        value_getter.add_dependency(self._current_index)
        self._current_index.add_child(value_getter)
        return value_getter

    def __iter__(
        self: Source,
    ) -> Generator[SourceItem, None, None]:
        """Iterate over all items in the source.

        This method allows the source to be used in for-loops and
        comprehensions by yielding each `SourceItem` in sequence. Each item is
        constructed using `__getitem__()`, which attaches the appropriate
        callbacks.

        Yields
        ------
        SourceItem
            Each item in the source, one at a time.

        Examples
        --------
        >>> from deeptrack.sources import Source

        >>> source = Source(a=[1, 2], b=[10, 20])

        >>> for item in source:
        ...     print(item["a"], item["b"])
        1 10
        2 20

        """

        for i in range(len(self)):
            yield self[i]

    def set_index(
        self: Source,
        index: int,
    ) -> Source:
        """Set the active index of the source for dynamic evaluation.

        This method updates the internal `_current_index` node, which is
        used when evaluating attribute-based access such as `source.a()`.
        It is typically called automatically when a `SourceItem` is
        activated, but can also be called manually to override the index.

        Parameters
        ----------
        index: int
            The index to set as the current active index.

        Returns
        -------
        Source
            The source itself, allowing method chaining.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source:
        >>> source = Source(
        ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     b=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        ... )

        >>> source.a(), source.b()
        (1, 10)

        >>> source.set_index(5)
        >>> source.a(), source.b()
        (6, 60)

        >>> source.set_index(-1)
        >>> source.a(), source.b()
        (9, 90)

        >>> source.set_index(1)
        >>> source.a(), source.b()
        (2, 20)

        """

        self._current_index.set_value(index)

        return self

    def on_activate(
        self: Source,
        callback: Callable[[SourceItem], None],
    ) -> None:
        """Register a callback to be triggered when a SourceItem is activated.

        The callback will be executed every time a `SourceItem` produced by
        this `Source` is called (i.e., when `item()` is invoked). The callback
        receives the `SourceItem` as its argument, allowing access or mutation
        of its contents.

        Parameters
        ----------
        callback : Callable[[SourceItem], None]
            A function that takes a `SourceItem` and performs a side-effect
            (e.g., logging, modifying metadata, triggering updates). The
            function must return None.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Define a callback function:
        >>> def log_access(item):
        ...     print(f"CALLBACK - Item accessed: {item}")

        Create a source and register the callback:
        >>> source = Source(a=[1, 2], b=[10, 20])
        >>> source.on_activate(log_access)

        >>> item = source[0]
        >>> item();
        CALLBACK - Item accessed: SourceItem({'a': 1, 'b': 10}, 2 callback(s))

        """

        self._callbacks.add(callback)


class Product(Source):
    """Class that represents the product of a source with one or more sources.

    This class is used to represent the product of a source with
    one or more sources. When accessed, it returns a deeptrack object that
    can be passed as properties to features.

    The feature can then be called with an item from the source
    to get the value of the feature for that item.
    """

    def __init__(
        self: Product,
        __source: Source = [{}],
        **kwargs: list[Any],
    ):

        product = itertools.product(__source, *kwargs.values())

        dict_of_lists = {k: [] for k in kwargs.keys()}
        source_dict = {k: [] for k in __source[0].keys()}

        # If overlapping keys, error.
        if set(kwargs.keys()).intersection(set(source_dict.keys())):
            raise ValueError(
                f"Overlapping keys in product. Duplicate keys: "
                f"{set(kwargs.keys()).intersection(set(source_dict.keys()))}"
            )

        dict_of_lists.update(source_dict)

        for source, *items in product:
            for k, v in source.items():
                dict_of_lists[k].append(v)
            for k, v in zip(kwargs.keys(), items):
                dict_of_lists[k].append(v)

        super().__init__(**dict_of_lists)    


class Subset(Source):

    def __init__(
        self: Subset,
        source: Source,
        indices: list[int],
    ):
        self.source = source
        self.indices = indices
        self._dict = {k: [v[i] for i in indices]
                      for k, v in source._dict.items()}

    def __iter__(
        self: Subset,
    ) -> Generator[SourceItem, None, None]:
        for i in self.indices:
            yield self.source[i]

    def __getitem__(
        self: Subset,
        index: int
    ) -> SourceItem:
        return self.source[self.indices[index]]

    def __len__(
        self: Subset,
    ) -> int:
        return len(self.indices)

    def __getattr__(
        self: Subset,
        name: str,
    ) -> Any:
        return getattr(self.source, name)


class Sources:
    """Joins multiple sources into a single access point.

    Used when one of multiple sources can be passed to a feature.
    For example the sources are split into training and validation sets,
    and the user can choose which one to use.

    Parameters
    ----------
    sources: Source
    
        The sources to join.
        
    """

    def __init__(
        self: Sources,
        *sources: Source,
    ):
        self.sources = sources

        keys = set()
        for source in sources:
            keys.update(source._dict.keys())

        self._dict = dict.fromkeys(keys)

        for key in keys:
            node = SourceDeepTrackNode(
                functools.partial(lambda key: self._dict[key], key)
            )

            setattr(self, key, node)

        for source in sources:
            source.on_activate(self._callback)

    def _callback(
        self: Sources,
        item: SourceItem,
    ) -> None:
        for key in item:
            getattr(self, key).invalidate()
            getattr(self, key).set_value(item[key])


Join = Sources


def random_split(
    source: Source,
    lengths: list[int | float],
    generator: np.random.Generator = np.random.default_rng()
) -> list[Subset]:
    """Randomly split source into non-overlapping new sources of given lengths.

    Parameters
    ----------
    source: Source
        The source to split.
        
    lengths: list of int or float
        The lengths of the new sources. If the lengths are floats,
        they are interpreted as fractions of the source.
        
    generator: numpy.random.Generator, optional
        The random number generator to use.
        
    """


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
                import warnings

                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    "This might result in an empty source."
                )

        # Cannot verify that dataset is Sized.
    if sum(lengths) != len(source):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not\
                          equal the length of the input dataset!")

    indices = generator.permutation(
        sum(lengths)).tolist()  # type: ignore[call-overload]
    return [Subset(source, indices[offset - length : offset])\
            for offset, length in zip(_accumulate(lengths), lengths)]


def _accumulate(
    iterable: list[int],
    fn: Callable [[int, int], int]=lambda x, y: x + y,
) -> Generator[int, None, None]:
    """Returns running totals with user specified operator.
    
    Default is summation.
    
    Examples
    --------
    >>> _accumulate([1,2,3,4,5])
    1 3 6 10 15
    
    >>> _accumulate([1,2,3,4,5], operator.mul)
    1 2 6 24 120   
    
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total
