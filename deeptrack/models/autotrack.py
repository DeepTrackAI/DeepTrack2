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


class AutoTrackerModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.model.compile(*args, **kwargs)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x, (offset_vector, transformation_matrix) = data

        with tf.GradientTape() as tape:
            y_pred = self(x)  # Forward pass

            # Prediction on first image is transformed
            transformed_origin = tf.linalg.matvec(
                transformation_matrix, y_pred[0, :2], transpose_a=True
            )

            y_pred = y_pred[:, :2] - transformed_origin
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                offset_vector,
                y_pred,
                regularization_losses=self.losses,
            )

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
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        return self.model(x)


class AutoTracker(KerasModel):
    def __init__(
        self,
        model=None,
        input_shape=(64, 64, 1),
        loss="mse",
        symmetries=1,
        mode="tracking",
        **kwargs
    ):
        self.symmetries = symmetries
        self.mode = mode

        # if loss == "auto":
        #     if mode == "tracking":
        #         loss = squared_affine_consistency
        #     elif mode == "orientation":
        #         loss = rotational_consistency
        #     elif mode == "sizing":
        #         loss = size_consistency
        #     else:
        #         raise ValueError(
        #             "Unknown mode provided to the auto tracker. Valid modes are 'tracking' and 'orientation'"
        #         )

        if model is None:
            model = self.default_model(input_shape=input_shape)

        model = AutoTrackerModel(model.model)

        super().__init__(model, loss=loss, **kwargs)

    def default_model(self, input_shape):

        return Convolutional(
            input_shape=input_shape,
            conv_layers_dimensions=[32, 64, 128],
            dense_layers_dimensions=(32, 32),
            steps_per_pooling=1,
            number_of_outputs=1 if self.mode == "sizing" else 2,
        )

    def data_generator(self, *args, **kwargs):

        transformation_function = None
        if self.mode == "tracking":
            transformation_function = Affine(
                translate=lambda: np.random.randn(2) * 2,
                rotate=lambda: np.random.rand() * np.pi * 2,
            )
        elif self.mode == "orientation":
            transformation_function = Affine(
                translate=lambda: np.random.randn(2) * 2,
                rotate=lambda: np.random.rand() * np.pi * 2,
            )
        elif self.mode == "sizing":
            transformation_function = Affine(
                translate=lambda: np.random.randn(2) * 2,
                scale=lambda: np.random.rand() * 0.6 + 0.7,
                rotate=lambda: np.random.rand() * np.pi * 2,
            )

        return AutoTrackGenerator(
            transformation_function, *args, symmetries=self.symmetries, **kwargs
        )


class AutoMultiTracker(KerasModel):

    data_generator = AutoTrackGenerator

    def __init__(
        self,
        model=None,
        input_shape=(None, None, 1),
        loss=squared_field_affine_consistency,
        **kwargs
    ):
        if not TFA_INSTALLED:
            raise RuntimeError(
                "The multipartcle version of the autotracker requires Tensorflow addons. You may need to update Tensorflow to install it."
            )

        if model is None:
            model = self.default_model(input_shape)

        super().__init__(model=model, loss=loss, **kwargs)

    def default_model(self, input_shape):

        valid_conv_block = ConvolutionalBlock(padding="valid")
        valid_pooling_block = PoolingBlock(padding="valid")
        valid_deconv_block = DeconvolutionalBlock(padding="valid")

        return UNet(
            input_shape=input_shape,
            conv_layers_dimensions=[16, 32, 64],
            base_conv_layers_dimensions=(128,),
            steps_per_pooling=1,
            number_of_outputs=3,
            encoder_convolution_block=valid_conv_block,
            decoder_convolution_block=valid_conv_block,
            base_convolution_block=valid_conv_block,
            output_convolution_block=valid_conv_block,
            pooling_block=valid_pooling_block,
            upsampling_block=valid_deconv_block,
        )
