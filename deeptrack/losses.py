""" Loss functions specialized for images and tracking tasks.

Functions
---------
flatten
    Flattends the inputs before calling the loss function.
sigmoid
    Adds a signmoid transformation to the prediction before calling the loss function.
weighted_crossentropy
    Binary crossentropy with weighted classes.
nd_mean_squared_error
    Mean square error with flattened inputs.
nd_mean_squared_logarithmic_error
    Mean square log error with flattened inputs.
nd_poisson
    Poisson error loss flattened inputs.
nd_squared_hinge
    Squared hinge error with flattened inputs.
nd_binary_crossentropy
    Binary crossentropy error with flattened inputs.
nd_kullback_leibler_divergence
    Kullback-Leibler divergence error with flattened inputs.
nd_mean_absolute_error
    Mean absolute error with flattened inputs.
nd_mean_absolute_percentage_error
    Mean absolute percentage error with flattened inputs.
"""

from functools import wraps

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

losses = keras.losses
K = keras.backend


# Extends the following loss functions
_COMPATIBLE_LOSS_FUNCTIONS = [
    losses.mse,
    losses.msle,
    losses.poisson,
    losses.squared_hinge,
    losses.binary_crossentropy,
    losses.kld,
    losses.mae,
    losses.mape,
]


def squared(func):
    @wraps(func)
    def inner(T, P):
        error = func(T, P)
        return K.square(error)

    return inner


def abs(func):
    @wraps(func)
    def inner(T, P):
        error = func(T, P)
        return K.abs(error)

    return inner


# LOSS WRAPPERS
def flatten(func):
    """Flattens the inputs before calling the loss function.

    Parameters
    ----------
    func : loss function
        The loss function to wrap.

    Returns
    -------
    function
        The new loss function.
    """

    def wrapper(T, P):
        T = K.flatten(T)
        P = K.flatten(P)
        return func(T, P)

    wrapper.__name__ = "nd_" + func.__name__
    return wrapper


def sigmoid(func):
    """Adds a signmoid transformation to the prediction before calling the loss function.

    Parameters
    ----------
    func : loss function
        The loss function to wrap.

    Returns
    -------
    function
        The new loss function.
    """
    # Takes the Sigmoid function of the prediction
    def wrapper(T, P):
        P = K.clip(P, -50, 50)
        P = 1 / (1 + K.exp(-1 * P))
        return func(T, P)

    wrapper.__name__ = "sigmoid_" + func.__name__
    return wrapper


def softmax(X, axis=(1, 2, 3)):
    X = K.exp(X - K.max(X, axis=axis, keepdims=True))
    return X / K.sum(X, axis, keepdims=True)


def weighted_crossentropy(weight=(1, 1), eps=1e-4):
    """Binary crossentropy with weighted classes.

    Parameters
    ----------
    weight : Tuple[float, float]
        Tuple of two numbers, indicating the weighting of the two classes -- 1 and 0.

    Returns
    -------
    function
        Weighted binary crossentropy loss function
    """

    def unet_crossentropy(T, P):
        return -K.mean(
            weight[0] * T * K.log(P + eps) + weight[1] * (1 - T) * K.log(1 - P + eps)
        ) / (weight[0] + weight[1])

    return unet_crossentropy


def affine_consistency(T, P):
    """Guides the network to be consistent under augmentations"""

    # Reconstruct transformation matrix
    T = K.reshape(T[:, :6], (-1, 2, 3))

    offset_vector = T[:, :, 2]
    transformation_matrix = T[:, :2, :2]

    # Prediction on first image is transformed
    transformed_origin = tf.linalg.matvec(
        transformation_matrix, P[0, :2], transpose_a=True
    )

    error = offset_vector - (P[:, :2] - transformed_origin)

    return error


def rotational_consistency(T, P):
    T = K.reshape(T[:, :6], (-1, 2, 3))
    transformation_matrix = T[:, :2, :2]
    normed_transf_matrix, _ = tf.linalg.normalize(transformation_matrix, axis=(1, 2))
    relative_normed_transf_matrix = tf.matmul(
        tf.linalg.inv(normed_transf_matrix[:1]), normed_transf_matrix
    )

    true_relative_cos = relative_normed_transf_matrix[:, 0, 0]
    true_relative_sin = relative_normed_transf_matrix[:, 0, 1]

    # Processing the prediction
    rotation_prediction = P[:, :2]  # cos(th), sin(th)
    norm_factor = K.sqrt(K.sum(K.square(rotation_prediction), axis=-1, keepdims=True))
    normed_predictions = rotation_prediction / norm_factor

    relative_cos = (
        normed_predictions[:1, 0] * normed_predictions[:, 0]
        + normed_predictions[:1, 1] * normed_predictions[:, 1]
    )

    relative_sin = (
        normed_predictions[:1, 0] * normed_predictions[:, 1]
        - normed_predictions[:1, 1] * normed_predictions[:, 0]
    )

    cos_err = K.abs(true_relative_cos - relative_cos)
    sin_err = K.abs(true_relative_sin - relative_sin)
    norm_err = K.square(norm_factor - 1)

    return K.mean(cos_err + sin_err + norm_err) / 3


def size_consistency(T, P):
    T = K.reshape(T[:, :6], (-1, 2, 3))
    transformation_matrix = T[:, :2, :2]
    determinant = tf.linalg.det(transformation_matrix)

    relative_predicted_size = P / P[:1]

    err = K.square(determinant - relative_predicted_size)
    return K.mean(err)


def adjacency_consistency(_, P):

    dFdx = P[:, 1:, :, :2] - P[:, :-1, :, :2]
    dFdy = P[:, :, 1:, :2] - P[:, :, :-1, :2]

    Wx = softmax(P[:, 1:, :, 2], axis=(1, 2))
    Wy = softmax(P[:, :, 1:, 2], axis=(1, 2))

    dFdx_err = (K.abs(dFdx[..., 0] + 1) + K.abs(dFdx[..., 1])) * Wx
    dFdy_err = (K.abs(dFdy[..., 0]) + K.abs(dFdy[..., 1] + 1)) * Wy

    return (K.sum(dFdx_err) + K.sum(dFdy_err)) / 2


def field_affine_consistency(T, P):

    transform = K.reshape(T[:, :6], (-1, 2, 3))

    offset_vector = transform[:, :, 2]

    non_mirrored = P[2::2]
    mirrored = P[1::2, ::-1, ::-1] * np.array([-1, -1, 1])

    offset_vector = K.reshape(offset_vector, (-1, 1, 1, 2))

    non_mirrored_error = (
        (P[:1, ..., :2] - non_mirrored[..., :2]) + offset_vector[2::2]
    ) * softmax(non_mirrored[..., 2:3], axis=(1, 2))

    mirrored_error = (
        (P[:1, ..., :2] - mirrored[..., :2]) + offset_vector[1::2]
    ) * softmax(mirrored[..., 2:3], axis=(1, 2))

    error = K.concatenate((non_mirrored_error, mirrored_error), axis=0)
    # error = error * K.cast_to_floatx(K.shape(error)[1] * K.shape(error)[2])
    return error * 8


squared_field_affine_consistency = squared(field_affine_consistency)
abs_field_affine_consistency = abs(field_affine_consistency)
squared_affine_consistency = squared(affine_consistency)
abs_affine_consistency = abs(affine_consistency)

# Wrap standard keras loss function with flatten.
for keras_loss_function in _COMPATIBLE_LOSS_FUNCTIONS:
    deeptrack_loss_function = flatten(keras_loss_function)
    globals()[deeptrack_loss_function.__name__] = deeptrack_loss_function
