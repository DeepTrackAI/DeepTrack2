from .generators import AutoTrackGenerator
from ..utils import KerasModel
from ...augmentations import Affine
import tensorflow as tf
import numpy as np


class AutoTrackerBaseModel(tf.keras.Model):
    """Base wrapper for self-reinforced tracking models

    Learns to solve problems of the form::

       y_i = A_i + f(x_0) @ B_i

    where @ denotes matrix multiplication. Expects training data formated as::

       (X, (A, B))

    The implementation also supports defining additional constraints on the solution
    by overriding the `call` method, by having it return additional values. The model
    is trained to keep the constraints at zero. For example::

       def call(self, x, training=False):
          y = super().call(x)
          if training:
             constraint_0 = K.sum(K.square(y), axis=(1, 2)) - 1
             constraint_1 = K.sum(y, axis=(1, 2))
             return y, constraint_0, constraint_1
          else:
             return y

    where the model is trained such that sum(y^2) == 1 and sum(y) == 0.

    If used to predict a vector, use the call function to add a dimension such that the shape
    matches (batch_dim, vector_dim, 1).

    Parameters
    ----------
    model : Tensorflow model
        Model to wrap


    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        strides = np.ones((2,))
        for layer in self.model.layers:
            if hasattr(layer, "strides"):
                strides *= layer.strides
        self.strides = strides

    def get_config(self):
        return {"model": self.model}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.model.compile(*args, **kwargs)

    def train_step(self, data):

        x, (offset_vector, transformation_matrix) = data

        with tf.GradientTape() as tape:
            y_pred, weights, *constraints = self(x, training=True)  # Forward pass

            weights = self.softmax(weights, 0.2)
            y_mean_pred = self.global_pool(y_pred, weights)

            loss_const = (
                tf.keras.backend.sum(
                    tf.keras.backend.abs(
                        y_pred - y_mean_pred[:, tf.newaxis, tf.newaxis]
                    )
                    * weights
                )
                / 100
            )

            y_mean_pred = y_mean_pred[..., tf.newaxis]

            # Prediction on first image is transformed
            transformed_origin = tf.linalg.matmul(
                transformation_matrix, y_mean_pred[:1]
            )
            y_diff = y_mean_pred - transformed_origin
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                offset_vector,
                y_diff,
                regularization_losses=self.losses,
            )

            loss = loss + loss_const

            constraint_losses = [tf.square(constraint) for constraint in constraints]
            for constraint_loss in constraint_losses:
                loss += constraint_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(
            offset_vector,
            y_pred,
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            **{m.name: m.result() for m in self.metrics},
            **{"consistency_loss": loss_const},
            **{
                f"constraint_{i}": constraint_loss[i]
                for i in range(len(constraint_losses))
            },
        }

    def global_pool(self, pred, weights):
        y_mean_pred = pred * weights
        y_mean_pred = tf.reduce_sum(y_mean_pred, axis=(1, 2)) / tf.reduce_sum(
            weights, axis=(1, 2)
        )
        return y_mean_pred

    def softmax(self, weights, dropout=0.0):
        weights = tf.keras.backend.dropout(weights, dropout)
        weights = weights + 1e-6
        weights = weights / tf.keras.backend.sum(weights, axis=(1, 2), keepdims=True)
        return weights

    def call(self, x, training=False):
        y = self.model(x)

        x_shape = tf.cast(tf.shape(x), "float")
        y_shape = tf.cast(tf.shape(y), "float")

        x_end, y_end = y_shape[1], y_shape[2]
        x_range = tf.range(x_end, dtype="float") * self.strides[0]
        y_range = tf.range(y_end, dtype="float") * self.strides[1]
        Y_mat, X_mat = tf.meshgrid(y_range, x_range)

        pred_x = y[..., 0]
        pred_y = y[..., 1]
        weights = tf.keras.activations.sigmoid(y[..., -1:])

        pred_x = pred_x + X_mat
        pred_y = pred_y + Y_mat

        pred = tf.stack((pred_x, pred_y), axis=-1)

        if training:
            pred = pred - x_shape[1:3] / 2

        return pred, weights


class AutoTracker(KerasModel):
    """Model that automatically learns to track a single object.

    For best results, keep the size of the images small (40-70 px).

    Parameters
    ----------
    model : Tensorflow model, optional
       A model that returns a vector of two numbers. If not defined,
       a default model is used instead.
    input_shape : tuple of ints
       Shape of the input images. Should match the expected shape of the model.
    loss, optimizer : compilation arguments
       Keras arguments used to compile the model
    symmetries : Int, optional
       The number of symmetries in the system. Only required for orientation tracking.
    """

    data_generator = AutoTrackGenerator

    class AutoTrackerModel(AutoTrackerBaseModel):
        def call(self, x, training=False):
            pred, weights = super().call(x, training=training)
            if not training:
                weights = self.softmax(weights)
                pred = self.global_pool(pred, weights)
                return pred
            return pred, weights

    def __init__(
        self,
        model=None,
        input_shape=(64, 64, 1),
        loss="mae",
        symmetries=1,
        **kwargs,
    ):
        self.symmetries = symmetries

        if model is None:
            model = self.default_model(input_shape=input_shape)

        if isinstance(model, KerasModel):
            model = model.model

        model = self.AutoTrackerModel(model)

        super().__init__(model, loss=loss, **kwargs)

    def default_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                32, 3, input_shape=input_shape, padding="same", activation="relu"
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                32, 3, padding="same", activation="relu"
            )
        )
        model.add(tf.keras.layers.MaxPool2D(2, padding="same"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.Conv2D(3, 1, padding="same"))
        return model


class AutoMultiTracker(AutoTracker):
    """Model that automatically learns to track a multiple objects.

    During training, expects ROIs of single objects. For best results, keep the size of
    the ROIs small (40-70 px).

    Parameters
    ----------
    model : Tensorflow model, optional
       A model that returns a (N, N, 2) array. If not defined,
       a default model is used instead.
    loss, optimizer : compilation arguments
       Keras arguments used to compile the model
    symmetries : Int, optional
       The number of symmetries in the system. Only required for orientation tracking.
    """

    AutoTrackerModel = AutoTrackerBaseModel

import scipy
from skimage import morphology

def local_consistency(pred):
    kernel = np.ones((3, 3, 1)) / 3**2

    pred_local_squared = scipy.signal.convolve(pred, kernel, "same") ** 2
    squared_pred_local = scipy.signal.convolve(pred ** 2, kernel, "same")

    squared_diff = (squared_pred_local - pred_local_squared).sum(-1)
    return 1 / (1e-6 + squared_diff)


def get_detection_score(pred, weights, alpha=0.5, beta=0.5):
    return weights[..., 0] ** alpha * local_consistency(pred) ** beta


def find_local_maxima(pred, score, cutoff=0.9, mode="quantile"):

    score = score[3:-3, 3:-3]

    th = cutoff

    if mode == "quantile":
        th = np.quantile(score, cutoff)
    elif mode == "ratio":
        th = np.max(score.flatten()) * cutoff

    hmax = morphology.h_maxima(np.squeeze(score), th) == 1

    hmax = np.pad(hmax, ((3,3), (3,3)))
    detections = pred[hmax, :]
    return np.array(detections)


def detect(pred, weights, alpha=0.5, beta=0.5, cutoff=0.95, mode="quantile"):

    score = get_detection_score(pred, weights, alpha=alpha, beta=beta)
    return find_local_maxima(pred, score, cutoff=cutoff, mode=mode)