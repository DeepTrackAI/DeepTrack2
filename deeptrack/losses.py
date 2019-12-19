from tensorflow import keras

losses = keras.losses
K = keras.backend

_COMPATIBLE_LOSS_FUNCTIONS = [
    losses.mse,
    losses.msle,
    losses.poisson,
    losses.squared_hinge,
    losses.binary_crossentropy,
    losses.kld,
    losses.mae,
    losses.mape
]

# LOSS WRAPPERS
def flatten(func):
    # Flattens T and P before calling
    def wrapper(T, P):
        T = K.flatten(T)
        P = K.flatten(P)
        return func(T, P)
    wrapper.__name__ = "nd_" + func.__name__
    return wrapper


def sigmoid(func):
    # Takes the Sigmoid function of the prediction
    def wrapper(T, P):
        P = 1 / (1 + K.exp(-P))
        return func(T, P)
    wrapper.__name__ = "sigmoid_" + func.__name__
    return wrapper


def weighted_crossentropy(weight=(1, 1), eps=1e-4):
    def unet_crossentropy(T, P):

        return -K.mean(weight[0] * T * K.log(P + eps) + weight[1] * (1 - T) * K.log(1 - P + eps)) / (weight[0] + weight[1])
    return unet_crossentropy


# Wrap standard keras loss function with flatten.
for keras_loss_function in _COMPATIBLE_LOSS_FUNCTIONS:
    deeptrack_loss_function = flatten(keras_loss_function)
    globals()[deeptrack_loss_function.__name__] = deeptrack_loss_function

