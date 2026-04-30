"""Utility classes and functions for dynamic data sources in DeepTrack2.

This module provides core abstractions for representing and manipulating
collections of data in a modular and composable way. It defines the structure
and behavior of dynamic sources that can be indexed, filtered, combined,
and tracked through DeepTrack's computational graph.

These tools are primarily used in scenarios where data needs to be dynamically
manipulated, filtered, or combined for feature generation in machine learning
pipelines.

Key Features
------------
- **Dynamic Data Access**

    Sources return dictionary-like items (`SourceItem`) that activate
    custom callbacks when accessed, enabling dynamic behavior such as
    dependency tracking and delayed evaluation.

- **Composable Data Structures**

    Includes tools like `Product`, `Subset`, and `Sources` to manipulate
    and combine data sources for flexible pipeline construction.

- **Hierarchical Node System**

    The `SourceDeepTrackNode` class extends `DeepTrackNode` to support
    hierarchical field access and automatic dependency propagation.

- **Random Splitting Utilities**

    Provides utilities such as `random_split()` for reproducible partitioning
    of sources into disjoint subsets for training/validation/test workflows.

Module Structure
----------------
Classes:

- `Source`: Represents one or more named sequences of data.

    Provides access to items as `SourceItem` and integrates with features
    for graph-based computation.

- `SourceItem`: A dict-like object that triggers callbacks when called.

    Wraps data fields from a `Source` and activates dependency updates
    on use.

- `SourceDeepTrackNode`: A DeepTrack node that supports attribute access.

    Automatically creates child nodes when dictionary-like attributes
    are accessed (e.g., `source.a.b`).

- `Product`: Cartesian product of a `Source` with additional fields.

    Allows combining items with new fields, either with or without a base
    source.

- `Subset`: Represents a filtered view of a `Source` via explicit indices.

    Provides indexed access to a restricted set of items.

- `Sources`: Joins multiple `Source` objects into one dynamic access point.

    Enables field sharing and flexible evaluation across datasets.

- `Join`: Alias for `Sources`.

Functions:

- `random_split(source, lengths, generator) -> list[Subset]`

    Randomly splits a `Source` into multiple non-overlapping subsets.

Examples
--------
import deeptrack as dt

**Trigger callbacks when a source item is accessed**

>>> from deeptrack.sources import Source
>>>
>>> source = Source(a=[1, 2], b=[3, 4])
>>>
>>> @source.on_activate
... def callback(item):
...     print("Activated:", item)

>>> source[0]();
Activated: SourceItem({'a': 1, 'b': 3}, 2 callback(s))

**Access nested dictionary-like data with dynamic nodes**

>>> from deeptrack.sources.base import SourceDeepTrackNode
>>>
>>> node = SourceDeepTrackNode(lambda: {"a": 1, "b": {"x": 42}})
>>> node.a()
1

>>> node.b.x()
42

**Use shared features across multiple sources**

>>> from deeptrack.sources import Source, Sources
>>>
>>> train = Source(a=[1, 2], b=[3, 4])
>>> val = Source(a=[5, 6], b=[7, 8])
>>> joined = Sources(train, val)
>>> feature = dt.Value(joined.a) + dt.Value(joined.b)
>>> feature(train[0])
4

>>> feature(val[0])
12

**Create a Cartesian product of fields**

>>> from deeptrack.sources import Source
>>>
>>> source = Source(a=[1, 2])
>>> product = source.product(b=[10, 20])
>>> list(product)
[SourceItem({'b': 10, 'a': 1}, 1 callback(s)),
 SourceItem({'b': 20, 'a': 1}, 1 callback(s)),
 SourceItem({'b': 10, 'a': 2}, 1 callback(s)),
 SourceItem({'b': 20, 'a': 2}, 1 callback(s))]

**Extract a subset of selected indices**

>>> from deeptrack.sources import Source, Subset
>>>
>>> source = Source(a=[1, 2, 3], b=[10, 20, 30])
>>> subset = Subset(source, [0, 2])
>>> list(subset)
[SourceItem({'a': 1, 'b': 10}, 1 callback(s)),
 SourceItem({'a': 3, 'b': 30}, 1 callback(s))]

**Split a source randomly into multiple parts**

>>> from deeptrack.sources import random_split, Source
>>>
>>> source = Source(
...     a=list(range(10)),
...     b=list(range(10, 20)),
... )
>>> train, val, test = random_split(source, [0.5, 0.3, 0.2])
>>> len(train), len(val), len(test)
(5, 3, 2)

"""

from __future__ import annotations

import functools
import itertools
import math
import warnings

from collections.abc import Sequence
from typing import Any, Callable, Generator, overload, TYPE_CHECKING

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


if TYPE_CHECKING:
    import torch


class SourceDeepTrackNode(DeepTrackNode):
    """A node that creates and caches child nodes when attributes are accessed.

    `SourceDeepTrackNode` is a specialization of `DeepTrackNode` intended for
    structured access to dictionary-like data. When an attribute is accessed
    and no explicit attribute exists, the node returns a child node that
    resolves to the corresponding key in the parent node's value.

    In other words, accessing `source.a.b` constructs a small dependency chain
    of nodes that (when evaluated) retrieves `source()["a"]["b"]`.

    Child nodes are cached to provide stable identity (`source.a is source.a`)
    and to make the dependency/children trees inspectable even when the user
    does not hold external references.

    Notes
    -----
    - Attribute names starting with "_" are not treated as data keys. This
      prevents clashes with internal `DeepTrackNode` attributes and avoids
      accidental creation of nodes for private/dunder names.
    - The value returned by evaluating this node (`self()`) must support
      string-key indexing (i.e., implement the `.__getitem__(str)` method).

    Parameters
    ----------
    action: Any | Callable
        The node action. If callable, it is evaluated to produce the node's
        value. If non-callable, it is treated as a constant value.
        The produced value must be dictionary-like (support `value[key]` where
        `key` is a string).
    node_name: str | None, optional
        Optional name assigned to the node. Defaults to `None`.
    **kwargs: Any
        Additional arguments for subclasses or extended functionality.

    Examples
    --------
    >>> from deeptrack.sources.base import SourceDeepTrackNode

    Create a dictionary-like source:

    >>> data = {"x": 42, "y": {"z": 3.14}}
    >>> source = SourceDeepTrackNode(data, node_name="root")

    Access nested keys as nodes:

    >>> source.x()
    42

    >>> source.y()
    {'z': 3.14}

    >>> source.y.z()
    3.14

    Keys starting with "_" are not accessible via attribute syntax:

    >>> source = SourceDeepTrackNode({"_x": 1})
    >>> source._x
    AttributeError: 'SourceDeepTrackNode' object has no attribute '_x'

    """

    def __getattr__(
        self: SourceDeepTrackNode,
        name: str,
    ) -> SourceDeepTrackNode:
        """Create or return a cached child node for the given key.

        This method is invoked only if normal attribute lookup fails. It
        returns a child node that resolves to `self()[name]` when evaluated.

        Parameters
        ----------
        name: str
            The key to retrieve from the dictionary-like value returned by
            evaluating the parent node.

        Returns
        -------
        SourceDeepTrackNode
            A child node representing the requested key.

        Raises
        ------
        AttributeError
            If `name` starts with "_" (reserved for internal/private
            attributes).

        """

        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        cache = self._get_child_cache()
        cached = cache.get(name)
        if cached is not None:
            return cached

        parent_name = self.node_name
        child_name = f"{parent_name}.{name}" if parent_name else name

        node = SourceDeepTrackNode(
            lambda parent=self, key=name: parent()[key],
            node_name=child_name,
        )
        node.add_dependency(self)
        cache[name] = node
        return node

    def _get_child_cache(
        self: SourceDeepTrackNode,
    ) -> dict[str, SourceDeepTrackNode]:
        """Return the per-instance cache of attribute-created child nodes.

        The cache is stored in a private attribute to avoid polluting the
        instance namespace with arbitrary data keys, and is created lazily.

        Returns
        -------
        dict[str, SourceDeepTrackNode]
            Mapping from key name to cached child node.

        """
        try:
            return object.__getattribute__(self, "_child_cache")
        except AttributeError:
            cache: dict[str, SourceDeepTrackNode] = {}
            object.__setattr__(self, "_child_cache", cache)
            return cache


class SourceItem(dict):
    """A dictionary-like object that triggers a list of callbacks when called.

    `SourceItem` wraps a dictionary entry that activates one or more callbacks
    when accessed via calling. This mechanism ensures that all dependent
    `DeepTrackNode`s are updated when a particular item in the source is
    selected.

    Parameters
    ----------
    callbacks: Sequence[Callable[[SourceItem], None]]
        A sequence of callback functions that are executed when the item is
        called. Each function receives the `SourceItem` itself as argument.

    Attributes
    ----------
    _callbacks: list[Callable[[SourceItem], None]]
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

    >>> item = SourceItem(callbacks=[log_callback], a=1, b=2)

    Call the item to trigger the callbacks:

    >>> item();
    CALLBACK - Accessed item: SourceItem({'a': 1, 'b': 2}, 1 callback(s))

    """

    _callbacks: list[Callable[[SourceItem], None]]

    def __init__(
        self: SourceItem,
        callbacks: Sequence[Callable[[SourceItem], None]],
        **kwargs: Any,
    ) -> None:
        """Initialize a SourceItem.

        Parameters
        ----------
        callbacks: Sequence[Callable[[SourceItem], None]]
            The sequence of callbacks to trigger when the item is called.
        **kwargs: Any
            Additional key-value pairs stored in the dictionary.

        """

        self._callbacks = list(callbacks)

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

        return (
            f"{self.__class__.__name__}({dict.__repr__(self)}, "
            f"{len(self._callbacks)} callback(s))"
        )


class Source:
    """A class that represents one or more sources of data.

    `Source` holds one or more named sequences (e.g., lists, arrays) and
    makes them accessible by index. It returns `SourceItem` objects that
    activate registered callbacks (e.g., for dependency tracking) when called.

    Each named field is accessible as an attribute (e.g., `source.a`) and
    can be passed directly to DeepTrack2 features such as `Value`. Features
    can then be evaluated on specific items by indexing the source (e.g.,
    `feature(source[i])`).

    Parameters
    ----------
    **kwargs: Sequence[Any]
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
    _callbacks: set[Callable[[SourceItem], None]]
        A set of callback functions triggered when a `SourceItem` is called.

    Methods
    -------
    `product(**kwargs: Sequence[Any]) -> Product`
        Return a new source representing the cartesian product of the current
        source with the given sequences.
    `constants(**kwargs: Sequence[Any]) -> Product`
        Return a new source where the given values are treated as constants.
    `filter(predicate: Callable[..., bool]) -> Subset`
        Return a new source containing only the items for which the predicate
        returns `True`.
    `set_index(index) -> Source`
        Set the active index used when evaluating attributes, like in
        `source.a()`.

    **Callback registration.**
    `on_activate(callback: Callable[[SourceItem], None]) -> None`
        Register a callback to be called when any item is activated.

    **Private and internal methods.**
    `__len__() -> int`
        Return the number of items in the source.
    `__getitem__(index) -> SourceItem or list[SourceItem]`
        Retrieve one or more items by index or slice.
    `_get_item(index: int) -> SourceItem`
        Retrieve a single SourceItem at a specified index.
    `_get_slice(slice_obj) -> list[SourceItem]`
        Retrieve a list of SourceItems corresponding to a slice.
    `_validate_all_same_length(kwargs) -> None`
        Validate that all input sequences have the same length.
    `_wrap(key) -> SourceDeepTrackNode`
        Wrap a field from the source into a SourceDeepTrackNode.
    `_wrap_indexable(key) -> SourceDeepTrackNode`
        Wrap an indexable field as a SourceDeepTrackNode.
    `_wrap_iterable(key) -> SourceDeepTrackNode`
        Wrap a non-indexable iterable field as a SourceDeepTrackNode.
    `__iter__() -> Generator[SourceItem, None, None]`
        Iterate over all items in the source.
    `__repr__() -> str:`
        Return a string representation of the source object.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.sources import Source

    Define a source with two fields:

    >>> source = Source(
    ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...     b=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    ... )

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
    18  # 9 original items x 2 values in "c"

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
    _callbacks: set[Callable[[SourceItem], None]]

    def __init__(
        self: Source,
        **kwargs: Sequence[Any],
    ) -> None:
        """Initialize a Source with one or more named data sequences.

        The input sequences must all have the same length and support integer
        indexing (i.e., implement both `__getitem__` and `__len__`). Each key
        becomes an attribute of the source and can be passed to DeepTrack2
        features for dynamic evaluation.

        Parameters
        ----------
        **kwargs : Sequence[Any]
            Named data sources, where each key is the name of a field (e.g.,
            "x", "label") and each value is an indexable sequence (e.g., list,
            NumPy array, PyTorch tensor). All sequences must have the same
            length. At least one sequence is required.

        Raises
        ------
        ValueError
            If the input sequences do not all have the same length, or if there
            are no input sequences.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source with two named sequences (note that they are of the
        same length):

        >>> source = Source(
        ...     a=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     b=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        ... )

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

        if not kwargs:
            raise ValueError(
                "Source must be initialized with at least one field."
            )

        self._validate_all_same_length(kwargs)

        self._dict = kwargs
        self._length = len(kwargs[list(kwargs.keys())[0]])
        self._current_index = DeepTrackNode(0, node_name="index")
        self._callbacks = set()

        for key in kwargs:
            setattr(self, key, self._wrap(key))

    def __getattr__(self, name: str) -> SourceDeepTrackNode:
        """Fallback attribute access for dynamically created source fields.

        The `Source` class creates its public attributes dynamically in
        `.__init__()` using `setattr()` (e.g., `source.a`, `source.b`, ...).
        Because these attributes are injected at runtime, static type
        checkers cannot infer their existence.

        This method is defined primarily to support static typing tools.
        By declaring `.__getattr__()` with a return type of
        `SourceDeepTrackNode`, we explicitly signal that dynamically
        created attributes are expected and that they resolve to
        `SourceDeepTrackNode` instances.

        Importantly, this method is not expected to be reached at runtime for
        valid source keys, since they are assigned during initialization.
        If this method is invoked, it indicates that an invalid attribute
        was requested.

        Do not remove this method unless the dynamic attribute injection
        mechanism is changed accordingly.

        """

        raise AttributeError(name)

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

    # Overloads are required for static type checkers. They allow tools such
    # as PyLance to infer that `source[i]` returns a `SourceItem` while
    # `source[i:j]` returns a `list[SourceItem]`. Without these overloads,
    # the return type would be a union, and attribute access like
    # `source[i]["a"]` would raise typing errors.

    @overload
    def __getitem__(self, index: int) -> SourceItem: ...

    @overload
    def __getitem__(self, index: slice) -> list[SourceItem]: ...

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

        2

        >>> item["b"]
        20

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

        return SourceItem(callbacks=callbacks, **values)

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

        # Get values for each index using ._get_item()
        return [self[i] for i in indices]

    def product(
        self: Source,
        **kwargs: Sequence[Any],
    ) -> Product:
        """Cartesian product of the current source with additional fields.

        This method returns a new `Product` source formed by taking the
        Cartesian product of the current source with the provided sequences.
        The new source will contain one item for every combination of the
        original items and the new sequences.

        Parameters
        ----------
        **kwargs: Sequence[Any]
            One or more additional sequences to combine with the current
            source. The keys define the names of the new fields, and the
            values are indexable sequences (e.g., lists or arrays).

        Returns
        -------
        Product
            A new source representing the Cartesian product of the current
            source with the additional sequences.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create an initial source:

        >>> source = Source(a=[1, 2], b=[3, 4])

        Take the product with a new sequence:

        >>> new_source = source.product(c=[5, 6])
        >>> new_source
        Product(c=[5, 6, 5, 6], a=[1, 1, 2, 2], b=[3, 3, 4, 4])

        """

        return Product(self, **kwargs)

    def constants(
        self: Source,
        **kwargs: Sequence[Any],
    ) -> Product:
        """New source where the given values are treated as constants.

        This method extends the current source with one or more constant
        fields. Each value is repeated to match the length of the existing
        source.

        Parameters
        ----------
        **kwargs: Sequence[Any]
            Named constant values to add to the source. Each key defines
            the name of a new field, and each value will be broadcasted
            as a constant (e.g., scalar, string, etc.).

        Returns
        -------
        Product
            A new source that includes the constant fields in addition to
            the original fields.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source:

        >>> source = Source(a=[1, 2], b=[3, 4])

        Add a constant field:

        >>> new_source = source.constants(c=5)
        >>> new_source
        Product(c=[5, 5], a=[1, 2], b=[3, 4])

        """

        return Product(self, **{k: [v] for k, v in kwargs.items()})

    def filter(
        self: Source,
        predicate: Callable[..., bool],
    ) -> Subset:
        """New source containing only items that satisfy a predicate.

        This method filters the source based on a boolean-valued predicate
        applied to each `SourceItem`. The result is a `Subset` containing
        only the items for which the predicate returns `True`.

        Parameters
        ----------
        predicate: Callable[..., bool]
            A function that takes the fields of a `SourceItem` as keyword
            arguments and returns `True` if the item should be included.

        Returns
        -------
        Subset
            A new source containing only the filtered items.

        Examples
        --------
        >>> from deeptrack.sources import Source

        Create a source:

        >>> source = Source(a=[1, 2], b=[3, 4])

        Filter to keep only items where a > 1:

        >>> new_source = source.filter(lambda a, b: a > 1)
        >>> new_source
        Subset(a=[2], b=[4])

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

        lengths = [len(value) for value in kwargs.values()]
        unique_lengths = set(lengths)

        if len(unique_lengths) > 1:
            raise ValueError(
                "All sources must have the same length, but the following "
                f"lengths were found: {lengths}"
            )

    def _wrap(
        self: Source,
        key: str,
    ) -> SourceDeepTrackNode:
        """Wrap a field from the source into a SourceDeepTrackNode.

        This method is called during source initialization to convert
        input sequences into graph-compatible nodes.

        This method checks whether the field associated with the given key
        is indexable (i.e., supports `.__getitem__()` and `.__len__()`) and
        wraps it accordingly using either `._wrap_indexable()` or
        `._wrap_iterable()`.

        Parameters
        ----------
        key: str
            The name of the field in the source dictionary.

        Returns
        -------
        SourceDeepTrackNode
            A node representing access to the field at the current index.

        """

        value = self._dict[key]

        # If the value supports __getitem__ and __len__, treat it as indexable
        if hasattr(value, "__getitem__") and hasattr(value, "__len__"):
            return self._wrap_indexable(key)

        # Otherwise, attempt to convert it into a list and wrap it
        return self._wrap_iterable(key)

    def _wrap_indexable(
        self: Source,
        key: str,
    ) -> SourceDeepTrackNode:
        """Wrap an indexable field as a SourceDeepTrackNode.

        This method creates a node that returns the value at the current
        index for a field that supports direct indexing (i.e., implements
        `.__getitem__()`).

        The returned node depends on the `_current_index` node, allowing
        dynamic evaluation as the index changes.

        Parameters
        ----------
        key: str
            The name of the field in the source dictionary.

        Returns
        -------
        SourceDeepTrackNode
            A node that evaluates to `self._dict[key][self._current_index()]`.

        """

        value_getter = SourceDeepTrackNode(
            lambda: self._dict[key][self._current_index()]
        )
        value_getter.add_dependency(self._current_index)
        return value_getter

    def _wrap_iterable(
        self: Source,
        key: str,
    ) -> SourceDeepTrackNode:
        """Wrap a non-indexable iterable field as a SourceDeepTrackNode.

        This method converts the iterable to a list and creates a node that
        returns the value at the current index. It is used when the field does
        not support direct indexing.

        Like `_wrap_indexable`, the resulting node depends on the
        `_current_index` node for dynamic evaluation.

        Parameters
        ----------
        key: str
            The name of the field in the source dictionary.

        Returns
        -------
        SourceDeepTrackNode
            A node that evaluates to
            `list(self._dict[key])[self._current_index()]`.

        """

        value_getter = SourceDeepTrackNode(
            lambda: list(self._dict[key])[self._current_index()]
        )
        value_getter.add_dependency(self._current_index)
        return value_getter

    def __iter__(
        self: Source,
    ) -> Generator[SourceItem, None, None]:
        """Iterate over all items in the source.

        This method allows the source to be used in for-loops and
        comprehensions by yielding each `SourceItem` in sequence. Each item is
        constructed using `.__getitem__()`, which attaches the appropriate
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

        This method updates the internal `._current_index()` node, which is
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

    def __repr__(
        self: Source,
    ) -> str:
        """Return a string representation of the source object.

        Shows the class name and a truncated preview of each field, displaying
        the first four items followed by an ellipsis if longer.

        Returns
        -------
        str
            A readable summary of the source fields and their contents.

        """

        field_summaries = []

        for k, v in self._dict.items():
            try:
                preview = list(v[:4]) if len(v) > 4 else list(v)
            except Exception:
                preview = "<?>"

            if isinstance(preview, list):
                suffix = "..." if len(v) > 4 else ""
                summary = f"{k}={preview}{suffix}"
            else:
                summary = f"{k}={preview}"

            field_summaries.append(summary)

        fields_repr = ", ".join(field_summaries)
        return f"{self.__class__.__name__}({fields_repr})"


class Product(Source):
    """Cartesian product of a source with one or more additional fields.

    `Product` constructs a new source by taking the Cartesian product between
    an existing `Source` and one or more sequences passed as keyword arguments.
    Each item in the result is a unique combination of an item from the
    original source and a value from each added field.

    This is typically used via `Source.product(...)`, and the resulting
    `Product` can be passed to DeepTrack features for dynamic evaluation.

    If no base source is provided, a dummy source with a single empty item is
    used. This allows syntax such as:

    >>> Product(x=[1, 2], y=[3, 4])
    Product(x=[1, 1, 2, 2], y=[3, 4, 3, 4])

    to create a Cartesian product of just the keyword arguments.

    While a list of dictionaries like `[{}]` would also technically work, this
    approach is not type-safe. Internally, `Source(__dummy=[0])` is used and
    then cleaned up to preserve correctness and consistency.

    Notes
    -----
    If the base source is empty, the Cartesian product is also empty. In this
    case, the resulting `Product` contains the expected field names, but all
    fields have length 0.

    Parameters
    ----------
    __source: Source | None, optional
        The base source to be expanded. If None, a default single-item
        source is used, allowing `Product` to act on keyword arguments alone.
    **kwargs: Sequence[Any]
        Named sequences to take the product with. Each field will be
        broadcasted across all items in the base source.

    Examples
    --------
    >>> from deeptrack.sources import Source

    Using the recommended Source.product() method:

    >>> source = Source(a=[1, 2])
    >>> product = source.product(b=[10, 20])
    >>> product
    Product(b=[10, 20, 10, 20], a=[1, 1, 2, 2])

    >>> list(product)
    [SourceItem({'b': 10, 'a': 1}, 1 callback(s)),
     SourceItem({'b': 20, 'a': 1}, 1 callback(s)),
     SourceItem({'b': 10, 'a': 2}, 1 callback(s)),
     SourceItem({'b': 20, 'a': 2}, 1 callback(s))]

    Equivalent direct usage of Product (advanced):

    >>> from deeptrack.sources.base import Product
    >>>
    >>> product = Product(source, b=[10, 20])
    >>> product
    Product(b=[10, 20, 10, 20], a=[1, 1, 2, 2])

    Using Product without a base source:

    >>> product = Product(x=[1, 2], y=["a", "b"])
    >>> product
    Product(x=[1, 1, 2, 2], y=['a', 'b', 'a', 'b'])

    Empty base sources are supported:

    >>> empty = Source(a=[], b=[])
    >>> product = empty.product(c=[1, 2])
    >>> len(product)
    Product(b=[], a=[], c=[])

    """

    def __init__(
        self: Product,
        __source: Source | None = None,
        **kwargs: Sequence[Any],
    ) -> None:
        """Initialize the Cartesian product of a source with additional fields.

        Parameters
        ----------
        __source: Source or None
            The base source to be expanded via Cartesian product. Defaults to
            `None`.
        **kwargs: Sequence[Any]
            Named sequences to take the product with.

        Raises
        ------
        ValueError
            If any key in `kwargs` overlaps with a key in the original source.

        """

        if __source is None:
            __source = Source(__dummy=[0])
            remove_dummy = True
        else:
            remove_dummy = False

        base_keys = set(__source._dict.keys())
        new_keys = set(kwargs.keys())

        # Check for overlapping keys. If overlapping keys, error.
        overlap = base_keys & new_keys
        if overlap:
            raise ValueError(
                f"Overlapping keys in product. Duplicate keys: {overlap}"
            )

        dict_of_lists: dict[str, list[Any]] = {
            k: [] for k in base_keys | new_keys
        }
        for base_item, *items in itertools.product(__source, *kwargs.values()):
            for k, v in base_item.items():
                dict_of_lists[k].append(v)
            for k, v in zip(kwargs.keys(), items):
                dict_of_lists[k].append(v)

        if remove_dummy:
            dict_of_lists.pop("__dummy", None)

        super().__init__(**dict_of_lists)


class Subset(Source):
    """A subset of a source defined by a list of indices.

    `Subset` represents a restricted version of a parent `Source`, containing
    only the items at the specified indices. The subset is materialized: all
    fields are sliced at construction time and stored as new sequences.

    The subset behaves like a normal `Source` while preserving activation
    compatibility with the parent source:

    - `len(subset)` equals the number of selected indices.
    - `subset[i]` returns the i-th element of the subset.
    - Dynamic field access (e.g., `subset.a()`) uses the subset's own active
      item.
    - Activating an item from the subset also activates the corresponding item
      in the parent source.

    This parent activation propagation preserves compatibility with pipelines
    built from the original source. For example, if a pipeline depends on
    `source.a`, evaluating it on `subset[i]` updates both `subset.a` and
    `source.a`.

    Parameters
    ----------
    source: Source
        The original source to take a subset from.
    indices: Sequence[int]
        Indices of the items to include in the subset. Indices follow normal
        Python indexing rules for the original source (including negatives).

    Attributes
    ----------
    source: Source
        The original source this subset was created from.
    indices: list[int]
        The indices used to construct the subset.

    Examples
    --------
    >>> from deeptrack.sources import Source, Subset

    Create a source:
    >>> source = Source(a=[1, 2, 3], b=[10, 20, 30])

    Extract a subset:
    >>> subset = Subset(source, [0, 2])
    >>> subset
    Subset(a=[1, 3], b=[10, 30])

    Activate the first subset item. This updates both the subset and the
    parent source.

    >>> subset[0]()
    SourceItem({'a': 1, 'b': 10}, 1 callback(s))

    >>> subset.a(), subset.b()
    (1, 10)

    >>> source.a(), source.b()
    (1, 10)

    Activate the second subset item. The corresponding parent item is also
    activated.
    >>> subset[1]()
    SourceItem({'a': 3, 'b': 30}, 1 callback(s))

    >>> subset.a(), subset.b()
    (3, 30)

    >>> source.a(), source.b()
    (3, 30)

    """

    source: Source
    indices: list[int]

    def __init__(
        self: Subset,
        source: Source,
        indices: Sequence[int],
    ) -> None:
        """Initialize a materialized subset from a source and indices.

        Parameters
        ----------
        source: Source
            The source to slice.
        indices: Sequence[int]
            Indices to include in the subset.

        Raises
        ------
        IndexError
            If any index is out of range for the source.

        """
        
        self.source = source
        self.indices = list(indices)

        sliced: dict[str, list[Any]] = {
            k: [v[i] for i in self.indices] for k, v in source._dict.items()
        }

        super().__init__(**sliced)

        # Backward-compatible behavior:
        # activating a subset item also activates the corresponding parent item.
        def activate_parent(item: SourceItem) -> None:
            for local_index in range(len(self)):
                is_match = all(
                    np.array_equal(item[key], self._dict[key][local_index])
                    for key in self._dict
                )

                if is_match:
                    parent_index = self.indices[local_index]
                    self.source[parent_index]()
                    return

        self.on_activate(activate_parent)


class Sources:
    """Join multiple sources into a single dynamic access point.

    `Sources` is used to combine multiple `Source` objects into one logical
    interface. It enables multiple independent sources to share the same
    features dynamically. This is particularly useful in cases like training/
    validation/test splits, where features are defined once and evaluated
    on different datasets.

    When any item from one of the joined sources is activated (i.e., called),
    the corresponding fields in the `Sources` object are updated and
    propagated through the computational graph via `SourceDeepTrackNode`.

    Fields that are not present in the activated item remain unchanged (or
    `None` if never set).

    Aliased as `Join` for semantic clarity in different contexts.

    Parameters
    ----------
    *sources: Source
        One or more `Source` instances to join. Each source must have
        compatible field names (e.g., all sources used with a common feature
        must define that feature’s required fields).

    Attributes
    ----------
    sources: tuple[Source, ...]
        The tuple of joined source instances.
    _dict: dict[str, Any]
        Dictionary used internally to store the currently active values
        for each field.

    Methods
    -------
    `_callback(item) -> None`
        Internal method triggered on activation. Updates dynamic fields
        with the activated item values.

    Examples
    --------
    >>> import deeptrack as dt
    >>> from deeptrack.sources import Source, Sources

    Create two disjoint sources:

    >>> train = Source(a=[1, 2], b=[10, 20])
    >>> val = Source(a=[3, 4], b=[30, 40])

    Join them together:

    >>> joined = Sources(train, val)

    Create a shared feature:

    >>> feature = dt.Value(joined.a) + dt.Value(joined.b)

    Evaluate on items from different sources:

    >>> feature(train[0])
    11

    >>> feature(train[1])
    22

    >>> feature(val[0])
    33

    >>> feature(val[1])
    44

    """

    sources: tuple[Source, ...]
    _dict: dict[str, Any]

    def __init__(
        self: Sources,
        *sources: Source,
    ) -> None:
        """Initialize a joined multi-source access point.

        Parameters
        ----------
        *sources: Source
            One or more `Source` instances to join.

        """

        self.sources = sources

        # Determine all unique keys across all sources
        keys = set()
        for source in sources:
            keys.update(source._dict.keys())

        # Initialize internal storage
        self._dict = dict.fromkeys(keys)

        # Create dynamic nodes for each key
        for key in keys:
            node = SourceDeepTrackNode(
                lambda k=key: self._dict[k],
                node_name=key,
            )
            setattr(self, key, node)

        # Register callback for each source
        for source in sources:
            source.on_activate(self._callback)

    def _callback(
        self: Sources,
        item: SourceItem,
    ) -> None:
        """Update the active field values from an activated item.

        This method is called when a `SourceItem` from any joined source is
        activated (i.e., when `item()` is invoked). It updates the internal
        dictionary of active values and invalidates the corresponding field
        nodes so that downstream computations see the new active values.

        Notes
        -----
        The field nodes created in `__init__` read their values from `self._dict`.
        Therefore, this callback updates `self._dict` and invalidates the nodes,
        rather than setting node values directly.

        Parameters
        ----------
        item: SourceItem
            The activated item whose values should become the current active
            values for this `Sources` instance.

        """

        for key, value in item.items():
            getattr(self, key).invalidate()
            self._dict[key] = value

    def __getattr__(self, name: str) -> SourceDeepTrackNode:
        """Fallback for dynamically injected field accessors.

        Field nodes are created dynamically in `__init__` via `setattr`.
        This method exists primarily to inform static type checkers that
        such attributes resolve to `SourceDeepTrackNode` instances.

        It is not expected to be reached at runtime for valid field names.

        """
        
        raise AttributeError(name)


Join = Sources


def random_split(
    source: Source,
    lengths: list[int] | list[float],
    generator: np.random.Generator | torch.Generator | None = None,
) -> list[Subset]:
    """Randomly split a source into non-overlapping subsets of specified sizes.

    This function splits a `Source` into multiple disjoint `Subset`s either
    by specifying absolute lengths (integers) or relative proportions (floats).

    If all entries in `lengths` are floats that sum to 1 or less, they are
    interpreted as fractions and scaled to match the total size of the source.
    Remaining items (due to rounding) are distributed round-robin to ensure
    full coverage.

    Parameters
    ----------
    source: Source
        The input `Source` to split.
    lengths: list[int] | list[float]
        A list of lengths for the resulting splits. If all values are floats
        summing to 1 (or slightly less), they are treated as proportions.
    generator: np.random.Generator | torch.Generator | None, optional
        A NumPy random generator used for shuffling. Defaults to `None`, in
        which case it is initialized to `np.random.default_rng()`.

    Returns
    -------
    list[Subset]
        A list of `Subset` instances corresponding to the split parts.

    Raises
    ------
    ValueError
        If the sum of provided lengths does not match the length of the source.

    Examples
    --------
    >>> from deeptrack.sources import Source, random_split

    Create a source:

    >>> source = Source(
    ...     a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...     b=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ... )

    Split into train (70%) and validation (30%):

    >>> train, val, test = random_split(source, [0.4, 0.3, 0.3])
    >>> train
    Subset(a=[3, 2, 7, 9], b=[13, 12, 17, 19])

    >>> val
    Subset(a=[5, 6, 1], b=[15, 16, 11])

    >>> test
    Subset(a=[0, 8, 4], b=[10, 18, 14])

    Split into fixed sizes:

    >>> train, val, test = random_split(source, [4, 3, 3])
    >>> train
    Subset(a=[3, 2, 7, 9], b=[13, 12, 17, 19])

    >>> val
    Subset(a=[5, 6, 1], b=[15, 16, 11])

    >>> test
    Subset(a=[0, 8, 4], b=[10, 18, 14])

    """

    n_total = len(source)

    # Determine subset lengths
    if (
        all(isinstance(x, float) for x in lengths)
        and float(sum(lengths)) <= 1.0 + 1e-12
    ):
        subset_lengths: list[int] = []

        for i, fraction in enumerate(lengths):
            if not (0.0 <= fraction <= 1.0):
                raise ValueError(
                    f"Fraction at index {i} is not between 0 and 1. "
                    f"Instead, it is {fraction}."
                )

            subset_lengths.append(int(math.floor(n_total * fraction)))

        remainder = n_total - sum(subset_lengths)

        # Add 1 to lengths in round-robin fashion until the remainder is 0.
        for i in range(remainder):
            subset_lengths[i % len(subset_lengths)] += 1

        for i, subset_length in enumerate(subset_lengths):
            if subset_length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    "This might result in an empty source.",
                    stacklevel=2,
                )
    else:
        if any(isinstance(x, float) for x in lengths):
            raise ValueError(
                "If `lengths` contains floats, all entries must be floats "
                "and their sum must be <= 1."
            )

        subset_lengths = list(lengths)

        for i, subset_length in enumerate(subset_lengths):
            if subset_length < 0:
                raise ValueError(
                    f"Length at index {i} is negative: {subset_length}."
                )

    if sum(subset_lengths) != n_total:
        raise ValueError(
            f"The sum of input lengths ({sum(subset_lengths)}) does not "
            f"equal the length of the input dataset ({n_total})."
        )

    # Generate permutation
    if generator is None:
        indices = np.random.default_rng().permutation(n_total).tolist()

    elif isinstance(generator, np.random.Generator):
        indices = generator.permutation(n_total).tolist()

    else:
        try:
            import torch  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise TypeError(
                "A torch.Generator was provided, but torch is not "
                "installed."
            ) from exc

        if isinstance(generator, torch.Generator):
            indices = torch.randperm(
                n_total,
                generator=generator,
            ).tolist()
        else:
            raise TypeError(
                "Unsupported generator type. Expected "
                "np.random.Generator, torch.Generator, or None."
            )

    # Build subsets
    return [
        Subset(source, indices[offset - subset_length : offset])
        for offset, subset_length in zip(
            _accumulate(subset_lengths),
            subset_lengths,
        )
    ]


def _accumulate(
    iterable: list[int],
    fn: Callable[[int, int], int] = lambda x, y: x + y,
) -> Generator[int, None, None]:
    """Return running totals using a binary accumulation function.

    This utility function computes cumulative values from a list using a
    user-defined binary operator. By default, it performs cumulative summation
    (i.e., partial sums), similar to `itertools.accumulate()`.

    Parameters
    ----------
    iterable: list[int]
        A list of integers to be accumulated.
    fn: Callable[[int, int], int], optional
        A binary function that takes two integers and returns a new integer.
        Defaults to addition.

    Yields
    ------
    int
        The cumulative value at each step of the accumulation.

    Examples
    --------
    >>> from deeptrack.sources.base import _accumulate

    Default behavior (cumulative sum):

    >>> for value in _accumulate([1, 2, 3, 4, 5]):
    ...     print(value)
    1
    3
    6
    10
    15

    Using a custom operator (e.g., multiplication):

    >>> import operator
    >>>
    >>> for value in _accumulate([1, 2, 3, 4, 5], fn=operator.mul):
    ...     print(value)
    1
    2
    6
    24
    120

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
