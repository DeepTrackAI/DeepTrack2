""" Image class and relative functions

Defines the Image class and functions that operate on it.
"""

import numpy as np
import operator as ops
import numpy as np

from .backend._config import cupy


def _binary_method(op):
    """Implement a forward binary method with a noperator, e.g., __add__."""

    def func(self, other):
        self, other = coerce([self, other])
        if isinstance(other, Image):
            return Image(
                op(self._value, other._value), copy=False
            ).merge_properties_from([self, other])
        else:
            return Image(op(self._value, other), copy=False).merge_properties_from(self)

    func.__name__ = "__{}__".format(op.__name__)
    return func


def _reflected_binary_method(op):
    """Implement a reflected binary method with a noperator, e.g., __radd__."""

    def func(self, other):
        self, other = coerce([self, other])
        if isinstance(other, Image):
            return Image(
                op(other._value, self._value), copy=False
            ).merge_properties_from([other, self])
        else:
            return Image(op(other, self._value), copy=False).merge_properties_from(self)

    func.__name__ = "__r{}__".format(op.__name__)
    return func


def _inplace_binary_method(op):
    """Implement a reflected binary method with a noperator, e.g., __radd__."""

    def func(self, other):
        self, other = coerce([self, other])
        if isinstance(other, Image):
            self._value = op(self._value, other._value)
            self.merge_properties_from(other)
        else:
            self._value = op(self._value, other)

        return self

    func.__name__ = "__i{}__".format(op.__name__)
    return func


def _numeric_methods(op):
    """Implement forward, reflected and inplace binary methods with an ufunc."""
    return (
        _binary_method(op),
        _reflected_binary_method(op),
        _inplace_binary_method(op),
    )


def _unary_method(
    op,
):
    """Implement a unary special method with an ufunc."""

    def func(self):
        return Image(op(self._value)).merge_properties_from(self)

    func.__name__ = "__{}__".format(op)
    return func

def is_tensorflow_object(value):
    """Checks if value is a tensorflow tensor without importing tensorflow."""
    
    # module starts with tensorflow
    if hasattr(value, "__module__") and value.__module__.startswith("tensorflow"):
        return True
    



class Image:
    """Wrapper of array-like values.

    This class is used to wrap array-like values such as numpy arrays, lists, and tensors. It can also wrap
    numbers and booleans. It has the same interface as numpy arrays, but it is not a numpy array. It has a two-fold
    purpose:

    1. To provide a unified interface for all array-like values, including tensors such as TensorFlow tensors or cupy arrays.
       This means that numpy functions can be used on Image objects regardless of whether they are numpy arrays or tensors.
    2. To store properties of `Feature`s used to create the image.



    Parameters
    ----------
    value : array-like
        The value to wrap. This can be a numpy array, list, or tensor.
    copy : bool
        Whether to copy the value or not. If False, the value is not copied.
    """

    def __init__(self, value, copy=True):
        super().__init__()

        if copy:
            self._value = self._view(value)
        else:
            if isinstance(value, Image):
                self._value = value._value
            else:
                self._value = value

        if isinstance(value, Image):
            self.properties = list(value.properties)
        else:
            self.properties = []

    def append(self, property_dict: dict):
        """Appends a dictionary to the properties list.

        Parameters
        ----------
        property_dict : dict
            A dictionary to append to the property list. Most commonly
            the current values of the properties of a feature.

        Returns
        -------
        Image
            Returns itself.
        """
        # TODO: Check if we still need to make a copy of the list
        self.properties = [*self.properties, property_dict]
        return self

    def get_property(
        self, key: str, get_one: bool = True, default: any = None
    ) -> list or any:
        """Retrieve a property.

        If the feature has the property defined by `key`, return
        its current_value. Otherwise, return `default`.
        If `get_one` is True, the first instance is returned;
        otherwise, all instances are returned as a list.

        Parameters
        ----------
        key
            The name of the property.
        get_one: optional
            Whether to return all instances of that property or just the first.
        default : optional
            What is returned if the property is not found.

        Returns
        -------
        any
            The value of the property if found, else `default`.

        """

        if get_one:
            for prop in self.properties:
                if key in prop:
                    return prop[key]
            return default
        else:
            return [prop[key] for prop in self.properties if key in prop] or default

    def merge_properties_from(self, other) -> "Image":
        """Merge properties with those from another Image.

        Appends properties from another images such that no property is duplicated.
        Uniqueness of a dictionary of properties is determined from the
        property `hash_key`.

        Most functions involving two images should automatically output an image with
        merged properties. However, since each property is guaranteed to be unique,
        it is safe to manually call this function if there is any uncertainty.

        Parameters
        ----------
        other : Image or list of Image or numpy array
            The data to retrieve properties from.

        """
        if isinstance(other, Image):
            for new_prop in other.properties:

                # Check if the property is already in the list
                should_append = True
                for my_prop in self.properties:
                    if my_prop is new_prop:
                        # Prop already added
                        should_append = False
                        break

                if should_append:
                    self.append(new_prop)

        elif isinstance(other, np.ndarray):
            # arrays can not contain properties
            return self
        else:

            # Ensure that the recursion is not infinite
            if not isinstance(other, str):
                # Check if other is iterable. If it is, recurse.
                if hasattr(other, "__iter__"):
                    for item in other:
                        self.merge_properties_from(item)

        return self

    def _view(self, value):

        # To ensure that we don't create nested Image objects.
        if isinstance(value, Image):
            return self._view(value._value)
        if isinstance(value, (np.ndarray, list, int, float, bool)):
            return np.array(value)

        return value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        # Allows to use numpy ufuncs on Image objects.

        # Ensures all inputs are the same type
        args = coerce(inputs)

        # Strips the Image object from the inputs
        args = tuple(strip(arg) for arg in args)

        # Check if the ufunc is supported
        if is_tensorflow_object(self._value):
            from deeptrack.backend.tensorflow_bindings import TENSORFLOW_BINDINGS
            if ufunc in TENSORFLOW_BINDINGS:
                ufunc = TENSORFLOW_BINDINGS[ufunc]
            else:
                return NotImplemented

        # If an output array is defined and is an Image, we redirect the output to the value of that image.
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x._value if isinstance(x, Image) else x for x in out)

        # Call the ufunc
        results = getattr(ufunc, method)(*args, **kwargs)

        if type(results) is tuple:
            # If the ufunc returns a tuple, we return a tuple of Image objects.
            outputs = []
            for result in results:
                out = Image(result, copy=False)
                out.merge_properties_from(inputs)
                outputs.append(out)

            return tuple(outputs)
        elif method == "at":
            # does not have an output
            return None
        else:
            # If the ufunc returns a single value, we return an Image object.
            result = Image(results, copy=False)
            result.merge_properties_from(inputs)
            return result

    def __array_function__(self, func, types, args, kwargs):

        # Allows to use numpy functions on Image objects.

        # Ensures all inputs are the same type
        values = coerce(args)
        # Strips the Image object from the inputs
        values = [strip(arg) for arg in values]

        # Check if the function is supported
        if is_tensorflow_object(self._value):
            from deeptrack.backend.tensorflow_bindings import TENSORFLOW_BINDINGS
            if func in TENSORFLOW_BINDINGS:
                func = TENSORFLOW_BINDINGS[func]
            else:
                return NotImplemented
        elif not (
            isinstance(self._value, (np.ndarray, tuple, list))
            or np.isscalar(self._value)
        ) and not hasattr(self._value, "__array_function__"):
            return NotImplemented

        out = func(*values, **kwargs)

        # Constants are not wrapped as Image objects
        if isinstance(out, (bool, int, float)):
            return out

        # If the function returns a tuple, we return a tuple of Image objects.
        if isinstance(out, tuple):
            outputs = []
            for result in out:
                out = Image(result, copy=False)
                out.merge_properties_from(args)
                outputs.append(out)

            return tuple(outputs)
        else:
            # If the function returns a single value, we return an Image object.
            result = Image(out, copy=False)
            result.merge_properties_from(args)
            return result

    def __array__(self, *args, **kwargs):
        """Convert to numpy array."""
        return np.array(self.to_numpy()._value, *args)

    def to_tf(self):
        """Convert to tensorflow tensor."""
        import tensorflow

        if isinstance(self._value, np.ndarray):
            return Image(
                tensorflow.constant(self._value), copy=False
            ).merge_properties_from(self)

        if isinstance(self._value, cupy.ndarray):
            return Image(
                tensorflow.constant(self._value.get()), copy=False
            ).merge_properties_from(self)

        return self

    def to_cupy(self):
        """Convert to cupy array."""

        if isinstance(self._value, np.ndarray):
            return Image(cupy.array(self._value), copy=False).merge_properties_from(
                self
            )

        return self

    def to_numpy(self):
        """Convert to numpy array."""
        if isinstance(self._value, np.ndarray):
            return self
        if isinstance(self._value, cupy.ndarray):
            return Image(self._value.get(), copy=False).merge_properties_from(self)
        if is_tensorflow_object(self._value):
            return Image(self._value.numpy(), copy=False).merge_properties_from(self)

        return self

    def __getattr__(self, key):
        # Allows to access properties of the value.
        return getattr(self._value, key)

    def __getitem__(self, idx):
        # Allows indexing of the value.

        idx = strip(idx)
        out = self._value.__getitem__(idx)
        if isinstance(out, (bool, int, float, complex)):
            return out

        out = Image(out, copy=False)
        out.merge_properties_from([self, idx])
        return out

    def __setitem__(self, key, value):
        # Allows indexing and assignment of the value.
        key = strip(key)
        value = strip(value)
        o = self._value.__setitem__(key, value)
        self.merge_properties_from([key, value])
        return o

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __nonzero__(self):
        return bool(self._value)

    def __bool__(self):
        return bool(self._value)

    def __round__(self, *args, **kwargs):
        return round(self._value, *args, **kwargs)

    def __len__(self):
        return len(self._value)

    def __repr__(self):
        return f"Image({repr(self._value)})"

    __lt__ = _binary_method(ops.lt)
    __le__ = _binary_method(ops.le)
    __eq__ = _binary_method(ops.eq)
    __ne__ = _binary_method(ops.ne)
    __gt__ = _binary_method(ops.gt)
    __ge__ = _binary_method(ops.ge)

    # numeric methods
    __add__, __radd__, __iadd__ = _numeric_methods(ops.add)
    __sub__, __rsub__, __isub__ = _numeric_methods(ops.sub)
    __mul__, __rmul__, __imul__ = _numeric_methods(ops.mul)
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(ops.matmul)
    # Python 3 does not use __div__, __rdiv__, or __idiv__
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(ops.truediv)
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(ops.floordiv)
    __mod__, __rmod__, __imod__ = _numeric_methods(ops.mod)
    __divmod__ = _binary_method(divmod)
    __rdivmod__ = _reflected_binary_method(divmod)
    # __idivmod__ does not exist
    # TODO: handle the optional third argument for __pow__?
    __pow__, __rpow__, __ipow__ = _numeric_methods(ops.pow)
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods(ops.lshift)
    __rshift__, __rrshift__, __irshift__ = _numeric_methods(ops.rshift)
    __and__, __rand__, __iand__ = _numeric_methods(ops.and_)
    __xor__, __rxor__, __ixor__ = _numeric_methods(ops.xor)
    __or__, __ror__, __ior__ = _numeric_methods(ops.or_)

    # unary methods
    __neg__ = _unary_method(ops.neg)
    __pos__ = _unary_method(ops.pos)
    __abs__ = _unary_method(ops.abs)
    __invert__ = _unary_method(ops.invert)


def strip(v):
    # Strip the Image object from the value.
    if isinstance(v, Image):
        return v._value

    if isinstance(v, (list, tuple)):
        return type(v)([strip(i) for i in v])

    return v


def coerce(images):
    # Coerce the images to the same type.
    images = [Image(image, copy=False) for image in images]
    if any(isinstance(i._value, cupy.ndarray) for i in images):
        return [i.to_cupy() for i in images]
    else:
        return images


FASTEST_SIZES = [0]
for n in range(1, 10):
    FASTEST_SIZES += [2 ** a * 3 ** (n - a - 1) for a in range(n)]
FASTEST_SIZES = np.sort(FASTEST_SIZES)


def pad_image_to_fft(image: Image, axes=(0, 1)) -> Image:
    """Pads image to speed up fast fourier transforms.
    Pads image to speed up fast fourier transforms by adding 0s to the
    end of the image.

    Parameters
    ----------
    image
        The image to pad
    axes : iterable of int, optional
        The axes along which to pad.
    """

    def _closest(dim):
        # Returns the smallest value frin FASTEST_SIZES
        # larger than dim
        for size in FASTEST_SIZES:
            if size >= dim:
                return size

    new_shape = np.array(image.shape)
    for axis in axes:
        new_shape[axis] = _closest(new_shape[axis])

    increase = np.array(new_shape) - image.shape
    pad_width = [(0, inc) for inc in increase]

    return np.pad(image, pad_width, mode="constant")


def maybe_cupy(array):
    """Converts array to cupy if gpu is available and enabled."""
    from . import config

    if config.gpu_enabled:
        return cupy.array(array)

    return array
