from .utils import KerasModel, as_KerasModel
from .layers import as_block
from tensorflow.keras import layers, models


class FullyConnected(KerasModel):
    def __init__(
        self,
        input_shape,
        dense_layers_dimensions=(32, 32),
        dropout=(),
        flatten_input=True,
        number_of_outputs=3,
        output_activation=None,
        dense_block="dense",
        **kwargs
    ):
        """Creates and compiles a fully connected neural network.

        A convolutional network with a dense top.

        Parameters
        ----------
        input_shape : tuple of ints
            Size of the images to be analyzed.
        dense_layers_dimensions : tuple of ints
            Number of units in each dense layer.
        flatten_input : bool
            Whether to add a flattening layer to the input
        number_of_outputs : int
            Number of units in the output layer.
        output_activation : str or keras activation
            The activation function of the output.
        dense_block
        loss : str or keras loss function
            The loss function of the network.

        Returns
        -------
        keras.models.Model
            Deep learning network
        """

        dense_block = as_block(dense_block)

        # INITIALIZE DEEP LEARNING NETWORK
        input_layer = layers.Input(shape=input_shape)

        layer = input_layer
        if flatten_input:
            layer = layers.Flatten()(layer)

        # DENSE TOP
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layers_dimensions)), dense_layers_dimensions
        ):

            if dense_layer_number == 0 and not flatten_input:
                layer = dense_block(dense_layer_dimension, input_shape=input_shape)(
                    layer
                )
            else:
                layer = dense_block(dense_layer_dimension)(layer)

            if dropout:
                layer = layers.Dropout(dropout[0])(layer)
                dropout = dropout[1:]

        # OUTPUT LAYER

        output_layer = layers.Dense(number_of_outputs, activation=output_activation)(
            layer
        )

        model = models.Model(input_layer, output_layer)

        super().__init__(model, **kwargs)
