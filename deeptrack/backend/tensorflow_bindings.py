""" Contains bindings for tensorflow.

Registers __array_function__ implementations for dealing with tensorflow Tensor objects.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

TENSORFLOW_BINDINGS = {}

__all__ = ["TENSORFLOW_BINDINGS", "implements_tf"]


def implements_tf(np_function):
    "Register an __array_function__ implementation for Image objects wrapping Tensors."

    def decorator(func):
        TENSORFLOW_BINDINGS[np_function] = func
        return func

    return decorator


# ===================================================================
# MATH UFUNCS
# ===================================================================


@implements_tf(np.square)
def _tf_square(x):
    return tf.square(x)


@implements_tf(np.logaddexp)
def _tf_logaddexp(x1, x2):
    return tf.math.log(tf.exp(x1) + tf.exp(x2))


# Uncomment when log2 not experimental
# @implements_tf(np.logaddexp2)
# def _tf_logaddexp2(x1, x2):
#     return tf.math.log2(tf.square(x1) + tf.square(x2))


@implements_tf(np.abs)
def _tf_abs(x):
    return tf.abs(x)


@implements_tf(np.fabs)
def _tf_fabs(x):
    return tf.abs(x)


@implements_tf(np.sign)
def _tf_sign(x):
    return tf.sign(x)


# Uncomment when heaviside not experimental
# @implements_tf(np.heaviside)
# def _tf_heaviside(x):
#     return tf.heaviside(x)


@implements_tf(np.sqrt)
def _tf_sqrt(x):
    return tf.sqrt(x)


@implements_tf(np.cbrt)
def _tf_cbrt(x):
    return x ** (1 / 3)


@implements_tf(np.reciprocal)
def _tf_reciprocal(x):
    return 1 / x


# ===================================================================
# TRIGONOMETRIC UFUNCS
# ===================================================================


@implements_tf(np.sin)
def _tf_sin(x):
    return tf.sin(x)


@implements_tf(np.cos)
def _tf_cos(x):
    return tf.cos(x)


@implements_tf(np.tan)
def _tf_tan(x):
    return tf.tan(x)


@implements_tf(np.arcsin)
def _tf_arcsin(x):
    return tf.asin(x)


@implements_tf(np.arccos)
def _tf_arccos(x):
    return tf.acos(x)


@implements_tf(np.arctan)
def _tf_arctan(x):
    return tf.atan(x)


@implements_tf(np.arctan2)
def _tf_arctan2(x1, x2):
    return tf.atan2(x1, x2)


@implements_tf(np.hypot)
def _tf_hypot(x1, x2):
    return tf.sqrt(tf.square(x1) + tf.square(x2))


@implements_tf(np.sinh)
def _tf_sinh(x):
    return tf.sinh(x)


@implements_tf(np.cosh)
def _tf_cosh(x):
    return tf.cosh(x)


@implements_tf(np.tanh)
def _tf_tanh(x):
    return tf.tanh(x)


@implements_tf(np.arccosh)
def _tf_arccosh(x):
    return tf.acosh(x)


@implements_tf(np.arcsinh)
def _tf_arcsinh(x):
    return tf.asinh(x)


@implements_tf(np.arctanh)
def _tf_arctanh(x):
    return tf.atanh(x)


@implements_tf(np.maximum)
def _tf_maximum(x1, x2):
    return tf.maximum(x1, x2)


@implements_tf(np.fmax)
def _tf_fmax(x1, x2):
    return tf.maximum(x1, x2)


@implements_tf(np.minimum)
def _tf_minimum(x1, x2):
    return tf.minimum(x1, x2)


@implements_tf(np.minimum)
def _tf_fmin(x1, x2):
    return tf.minimum(x1, x2)


# ===================================================================
# REDUCING FUNCTIONS
# ===================================================================


@implements_tf(np.sum)
def _tf_sum(x, axis=None, keepdims=False):
    return K.sum(x, axis=axis, keepdims=keepdims)


@implements_tf(np.prod)
def _tf_prod(x, axis=None, keepdims=False):
    return K.prod(x, axis=axis, keepdims=keepdims)


@implements_tf(np.mean)
def _tf_mean(x, axis=None, keepdims=False):
    return K.mean(x, axis=axis, keepdims=keepdims)


@implements_tf(np.median)
def _tf_median(x, axis=None, keepdims=False, interpolation=None):
    return tfp.stats.quantiles(
        x, 0.5, axis=axis, keepdims=keepdims, interpolation=interpolation
    )


@implements_tf(np.std)
def _tf_std(x, axis=None, keepdims=False):
    return K.std(x, axis=axis, keepdims=keepdims)


@implements_tf(np.var)
def _tf_var(x, axis=None, keepdims=False):
    return K.var(x, axis=axis, keepdims=keepdims)


@implements_tf(np.cumsum)
def _tf_cumsum(x, axis=None):
    return K.cumsum(x, axis=axis)


@implements_tf(np.min)
def _tf_min(x, axis=None, keepdims=False):
    return K.min(x, axis=axis, keepdims=keepdims)


@implements_tf(np.max)
def _tf_max(x, axis=None, keepdims=False):
    return K.max(x, axis=axis, keepdims=keepdims)


@implements_tf(np.ptp)
def _tf_ptp(x, axis=None, keepdims=False):
    return K.max(x, axis=axis, keepdims=keepdims) - K.min(
        x, axis=axis, keepdims=keepdims
    )


@implements_tf(np.quantile)
def _tf_quantile(x, q, axis=None, keepdims=False, interpolation=None):
    return tfp.stats.quantiles(
        x, q, axis=axis, keepdims=keepdims, interpolation=interpolation
    )


@implements_tf(np.percentile)
def _tf_percentile(x, q, axis=None, keepdims=False, interpolation=None):
    return tfp.stats.percentile(
        x, q, axis=axis, keepdims=keepdims, interpolation=interpolation
    )
