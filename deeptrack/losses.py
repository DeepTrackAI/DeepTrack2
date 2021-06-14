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
        P = 1 / (1 + K.exp(-P))
        return func(T, P)

    wrapper.__name__ = "sigmoid_" + func.__name__
    return wrapper


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
    T = K.reshape(T, (-1, 3, 2))

    offset_vector = T[:, -1, :]
    transformation_matrix = T[:, :2, :2]

    # Prediction on first image is transformed
    transformed_origin = tf.linalg.matvec(
        transformation_matrix, P[0, :2], transpose_a=True
    )

    error = offset_vector - (P - transformed_origin)

    return error


squared_affine_consistency = squared(affine_consistency)
abs_affine_consistency = abs(affine_consistency)

# Wrap standard keras loss function with flatten.
for keras_loss_function in _COMPATIBLE_LOSS_FUNCTIONS:
    deeptrack_loss_function = flatten(keras_loss_function)
    globals()[deeptrack_loss_function.__name__] = deeptrack_loss_function
