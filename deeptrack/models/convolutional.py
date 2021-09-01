from tensorflow.keras import layers, models

from ..backend.citations import unet_bibtex
from .layers import as_block
from .utils import KerasModel, as_KerasModel, with_citation


def center_crop(layer, target_layer):
    def inner(x):
        x, y = x
        shape = x.shape
        target_shape = y.shape
        diff_y = (shape[1] - target_shape[1]) // 2
        diff_x = (shape[2] - target_shape[2]) // 2
        return x[
            :,
            diff_y : (diff_y + target_shape[1]),
            diff_x : (diff_x + target_shape[2]),
        ]

    return layers.Lambda(inner, output_shape=lambda x: x[1].shape)(
        [layer, target_layer]
    )


class Convolutional(KerasModel):
    def __init__(
        self,
        input_shape=(51, 51, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        dense_layers_dimensions=(32, 32),
        steps_per_pooling=1,
        dropout=(),
        dense_top=True,
        number_of_outputs=3,
        output_activation=None,
        output_kernel_size=3,
        loss="mae",
        input_layer=None,
        convolution_block="convolutional",
        pooling_block="pooling",
        dense_block="dense",
        **kwargs
    ):
        """Creates and compiles a convolutional neural network.
        A convolutional network with a dense top.
        Parameters
        ----------
        input_shape : tuple of ints
            Size of the images to be analyzed.
        conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer.
        dense_layers_dimensions : tuple of ints
            Number of units in each dense layer.
        dropout : tuple of float
            Adds a dropout between the convolutional layers
        number_of_outputs : int
            Number of units in the output layer.
        output_activation : str or keras activation
            The activation function of the output.
        loss : str or keras loss function
            The loss function of the network.
        layer_function : Callable[int] -> keras layer
            Function that returns a convolutional layer with convolutions
            determined by the input argument. Can be use to futher customize the network.
        Returns
        -------
        keras.models.Model
            Deep learning network
        """

        # Update layer functions
        dense_block = as_block(dense_block)
        convolution_block = as_block(convolution_block)
        pooling_block = as_block(pooling_block)

        # INITIALIZE DEEP LEARNING NETWORK

        if isinstance(input_shape, list):
            network_input = [layers.Input(shape) for shape in input_shape]
            inputs = layers.Concatenate(axis=-1)(network_input)
        else:
            network_input = layers.Input(input_shape)
            inputs = network_input

        layer = inputs

        if input_layer:
            layer = input_layer(layer)

        # CONVOLUTIONAL BASIS
        for conv_layer_dimension in conv_layers_dimensions:

            for _ in range(steps_per_pooling):
                layer = convolution_block(conv_layer_dimension)(layer)

            if dropout:
                layer = layers.SpatialDropout2D(dropout[0])(layer)
                dropout = dropout[1:]

            # add pooling layer
            layer = pooling_block(conv_layer_dimension)(layer)

        # DENSE TOP

        if dense_top:
            layer = layers.Flatten()(layer)
            for dense_layer_dimension in dense_layers_dimensions:
                layer = dense_block(dense_layer_dimension)(layer)
            output_layer = layers.Dense(
                number_of_outputs, activation=output_activation
            )(layer)
        else:

            output_layer = layers.Conv2D(
                number_of_outputs,
                kernel_size=output_kernel_size,
                activation=output_activation,
                padding="same",
                name="output",
            )(layer)

        model = models.Model(network_input, output_layer)
        super().__init__(model, loss=loss, **kwargs)


convolutional = Convolutional


class UNet(KerasModel):

    citation = unet_bibtex

    def __init__(
        self,
        input_shape=(None, None, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        base_conv_layers_dimensions=(128, 128),
        output_conv_layers_dimensions=(16, 16),
        dropout=(),
        steps_per_pooling=1,
        number_of_outputs=1,
        output_kernel_size=3,
        output_activation=None,
        input_layer=None,
        encoder_convolution_block="convolutional",
        base_convolution_block="convolutional",
        decoder_convolution_block="convolutional",
        output_convolution_block="convolutional",
        pooling_block="pooling",
        upsampling_block="deconvolutional",
        **kwargs
    ):

        """Creates and compiles a U-Net.
        Parameters
        ----------
        input_shape : tuple of ints
            Size of the images to be analyzed.
        conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer during down-
            and upsampling.
        base_conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer at the base
            of the unet, where the image is the most downsampled.
        output_conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer after the
            upsampling.
        steps_per_pooling : int
            Number of convolutional layers between each pooling and upsampling
            step.
        number_of_outputs : int
            Number of convolutions in output layer.
        output_activation : str or keras activation
            The activation function of the output.
        loss : str or keras loss function
            The loss function of the network.
        layer_function : Callable[int] -> keras layer
            Function that returns a convolutional layer with convolutions
            determined by the input argument. Can be use to futher customize the network.
        Returns
        -------
        keras.models.Model
            Deep learning network.
        """

        # Update layer functions

        encoder_convolution_block = as_block(encoder_convolution_block)
        base_convolution_block = as_block(base_convolution_block)
        output_convolution_block = as_block(output_convolution_block)
        decoder_convolution_block = as_block(decoder_convolution_block)
        pooling_block = as_block(pooling_block)
        upsampling_block = as_block(upsampling_block)

        unet_input = layers.Input(input_shape)

        concat_layers = []

        layer = unet_input

        if input_layer:
            layer = input_layer(layer)

        # Downsampling path
        for conv_layer_dimension in conv_layers_dimensions:
            for _ in range(steps_per_pooling):
                layer = encoder_convolution_block(conv_layer_dimension)(layer)
            concat_layers.append(layer)

            if dropout:
                layer = layers.SpatialDropout2D(dropout[0])(layer)
                dropout = dropout[1:]

            layer = pooling_block(conv_layer_dimension)(layer)

        # Bottleneck path
        for conv_layer_dimension in base_conv_layers_dimensions:
            layer = base_convolution_block(conv_layer_dimension)(layer)

        # Upsampling path
        for conv_layer_dimension, concat_layer in zip(
            reversed(conv_layers_dimensions), reversed(concat_layers)
        ):

            layer = upsampling_block(conv_layer_dimension)(layer)

            # concat_layer = center_crop(concat_layer, layer) Not currently working

            layer = layers.Concatenate(axis=-1)([layer, concat_layer])

            for _ in range(steps_per_pooling):
                layer = decoder_convolution_block(conv_layer_dimension)(layer)

        # Output step
        for conv_layer_dimension in output_conv_layers_dimensions:
            layer = output_convolution_block(conv_layer_dimension)(layer)

        output_layer = layers.Conv2D(
            number_of_outputs,
            kernel_size=output_kernel_size,
            activation=output_activation,
            padding="same",
        )(layer)

        model = models.Model(unet_input, output_layer)
        super().__init__(model, **kwargs)


unet = UNet


class EncoderDecoder(KerasModel):
    def __init__(
        self,
        input_shape=(None, None, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        base_conv_layers_dimensions=(128, 128),
        output_conv_layers_dimensions=(16, 16),
        dropout=(),
        steps_per_pooling=1,
        number_of_outputs=1,
        output_kernel_size=3,
        output_activation=None,
        input_layer=None,
        encoder_convolution_block="convolutional",
        base_convolution_block="convolutional",
        decoder_convolution_block="convolutional",
        output_convolution_block="convolutional",
        pooling_block="pooling",
        upsampling_block="deconvolutional",
        **kwargs
    ):

        """Creates and compiles an EncoderDecoder.
        Parameters
        ----------
        input_shape : tuple of ints
            Size of the images to be analyzed.
        conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer during down-
            and upsampling.
        base_conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer at the base
            of the unet, where the image is the most downsampled.
        output_conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer after the
            upsampling.
        steps_per_pooling : int
            Number of convolutional layers between each pooling and upsampling
            step.
        number_of_outputs : int
            Number of convolutions in output layer.
        output_activation : str or keras activation
            The activation function of the output.
        loss : str or keras loss function
            The loss function of the network.

        Returns
        -------
        keras.models.Model
            Deep learning network.
        """

        # Update layer functions

        encoder_convolution_block = as_block(encoder_convolution_block)
        base_convolution_block = as_block(base_convolution_block)
        output_convolution_block = as_block(output_convolution_block)
        decoder_convolution_block = as_block(decoder_convolution_block)
        pooling_block = as_block(pooling_block)
        upsampling_block = as_block(upsampling_block)

        unet_input = layers.Input(input_shape)

        layer = unet_input

        if input_layer:
            layer = input_layer(layer)

        # Downsampling path
        for conv_layer_dimension in conv_layers_dimensions:
            for _ in range(steps_per_pooling):
                layer = encoder_convolution_block(conv_layer_dimension)(layer)

            if dropout:
                layer = layers.SpatialDropout2D(dropout[0])(layer)
                dropout = dropout[1:]

            layer = pooling_block(conv_layer_dimension)(layer)

        # Bottleneck path
        for conv_layer_dimension in base_conv_layers_dimensions:
            layer = base_convolution_block(conv_layer_dimension)(layer)

        # Upsampling path
        for conv_layer_dimension in reversed(conv_layers_dimensions):

            layer = upsampling_block(conv_layer_dimension)(layer)

            for _ in range(steps_per_pooling):
                layer = decoder_convolution_block(conv_layer_dimension)(layer)

        # Output step
        for conv_layer_dimension in output_conv_layers_dimensions:
            layer = output_convolution_block(conv_layer_dimension)(layer)

        output_layer = layers.Conv2D(
            number_of_outputs,
            kernel_size=output_kernel_size,
            activation=output_activation,
            padding="same",
        )(layer)

        model = models.Model(unet_input, output_layer)
        super().__init__(model, **kwargs)
