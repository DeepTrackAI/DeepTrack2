from .utils import KerasModel, as_KerasModel
from .convolutional import Convolutional, UNet
from ..generators import AutoTrackGenerator
from ..losses import (
    rotational_consistency,
    squared_affine_consistency,
    squared_field_affine_consistency,
    size_consistency,
)
from ..layers import ConvolutionalBlock, PoolingBlock, DeconvolutionalBlock
from ..augmentations import Affine
import tensorflow as tf

import numpy as np

try:
    import tensorflow_addons as tfa

    TFA_INSTALLED = True
except:
    TFA_INSTALLED = False


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

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.model.compile(*args, **kwargs)

    def train_step(self, data):

        x, (offset_vector, transformation_matrix) = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Extract the constraints
            if not isinstance(y_pred, (tuple, list)):
                y_pred = (y_pred,)
            y_pred, *constraints = y_pred

            # Prediction on first image is transformed to get a label for
            # subsequent images in the batch.
            transformed_origin = tf.linalg.matmul(
                y_pred[:1],
                transformation_matrix,
                transpose_a=True,
                transpose_b=True,
            )

            y_pred = y_pred - transformed_origin
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                offset_vector,
                y_pred,
                regularization_losses=self.losses,
            )

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
            **{
                f"constraint_{i}": constraint_loss[i]
                for i in range(len(constraint_losses))
            },
        }

    def call(self, x, **kwargs):
        return self.model(x)


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

    class AutoTrackerModel(AutoTrackerBaseModel):
        def call(self, x, training=False):
            y = super().call(x)
            if training:
                y = tf.expand_dims(y, axis=-1)
            return y

    def __init__(
        self,
        model=None,
        input_shape=(64, 64, 1),
        loss="mse",
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

    def data_generator(self, *args, **kwargs):

        # Default affine transformation. Shown to be sufficient in most cases.
        transformation_function = Affine(
            translate=lambda: np.random.randn(2) * 2,
            rotate=lambda: np.random.rand() * np.pi * 2,
        )

        return AutoTrackGenerator(
            transformation_function, *args, symmetries=1, **kwargs
        )

    def default_model(self, input_shape):
        return Convolutional(
            input_shape=input_shape,
            conv_layers_dimensions=[32, 64, 128],
            dense_layers_dimensions=(32, 32),
            steps_per_pooling=1,
            number_of_outputs=2,
        )


class AutoMultiTracker(KerasModel):
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

    class AutoTrackerModel(AutoTrackerBaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            strides = np.ones((2,))
            for layer in self.model.layers:
                if hasattr(layer, "strides"):
                    strides *= layer.strides
            self.strides = strides

        def train_step(self, data):
            x, (offset_vector, transformation_matrix) = data

            offset_vector = offset_vector[:, tf.newaxis, tf.newaxis]
            transformation_matrix = transformation_matrix[:, tf.newaxis, tf.newaxis]
            # Unpack the data. Its structure depends on your model and
            # on what you pass to `fit()`.

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass

                # y_pred = y_pred[:, .., :]
                loss_const = tf.keras.backend.mean(
                    tf.keras.backend.abs(
                        y_pred
                        - tf.keras.backend.mean(y_pred, axis=(1, 2), keepdims=True)
                    )
                )

                # Prediction on first image is transformed
                transformed_origin = tf.linalg.matmul(
                    transformation_matrix, y_pred[:1], transpose_a=True
                )
                y_pred = y_pred - transformed_origin
                # Compute the loss value.
                # The loss function is configured in `compile()`.
                loss = self.compiled_loss(
                    offset_vector,
                    y_pred,
                    regularization_losses=self.losses,
                )
                total_loss = loss + loss_const

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)

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
                "consistency": loss_const,
            }

        def call(self, x, training=False):
            y = super().call(x)

            y_shape = tf.cast(tf.shape(y), "float")

            x_end, y_end = y_shape[1], y_shape[2]
            x_range = tf.range(x_end, dtype="float") * self.strides[0]
            y_range = tf.range(y_end, dtype="float") * self.strides[1]
            Y_mat, X_mat = tf.meshgrid(x_range, y_range)

            pred_x = y[..., 0] + X_mat
            pred_y = y[..., 1] + Y_mat
            pred = tf.stack((pred_x, pred_y), axis=-1)

            return pred

    def __init__(
        self,
        model=None,
        input_shape=(None, None, 1),
        loss="mse",
        symmetries=1,
        mode="tracking",
        **kwargs,
    ):
        self.symmetries = symmetries
        self.mode = mode
        if model is None:
            model = self.default_model(input_shape=input_shape)
        model = self.AutoTrackerModel(model)
        super().__init__(model, loss=loss, **kwargs)

    def default_model(self, input_shape=(None, None, 1)):
        # Update layer functions
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, 3, padding="valid", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(2, padding="valid"))
        model.add(tf.keras.layers.Conv2D(32, 3, padding="valid", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D(2, padding="valid"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="valid", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="valid", activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, padding="valid", activation="relu"))
        model.add(tf.keras.layers.Conv2D(2, 3, padding="valid"))
        return model

    def data_generator(self, *args, **kwargs):

        transformation_function = Affine(
            translate=lambda: np.random.randn(2) * 2,
            rotate=lambda: np.random.rand() * 2 * np.pi,
        )

        return AutoTrackGenerator(
            transformation_function, *args, symmetries=1, **kwargs
        )
