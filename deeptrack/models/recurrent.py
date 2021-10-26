from .utils import KerasModel, as_KerasModel

from tensorflow.keras import layers, models


class RNN(KerasModel):
    def __init__(
        self,
        input_shape=(51, 51, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        dense_layers_dimensions=(32,),
        rnn_layers_dimensions=(32,),
        return_sequences=False,
        output_activation=None,
        number_of_outputs=3,
        **kwargs
    ):
        """Creates and compiles a recurrent neural network.
        Parameters
        ----------
        input_shape : tuple of ints
            Size of the images to be analyzed.
        conv_layers_dimensions : tuple of ints
            Number of convolutions in each convolutional layer during down-
            and upsampling.
        dense_layers_dimensions : tuple of ints
            Number of units in each dense layer.
        rnn_layers_dimensions : tuple of ints
            Number of units in each recurrent layer.
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

        # INITIALIZE DEEP LEARNING NETWORK
        network = models.Sequential()

        # CONVOLUTIONAL BASIS
        for conv_layer_number, conv_layer_dimension in zip(
            range(len(conv_layers_dimensions)), conv_layers_dimensions
        ):

            # add convolutional layer
            conv_layer_name = "conv_" + str(conv_layer_number + 1)
            if conv_layer_number == 0:
                conv_layer = layers.Conv2D(
                    conv_layer_dimension,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    name=conv_layer_name,
                )
            else:
                conv_layer = layers.Conv2D(
                    conv_layer_dimension,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    name=conv_layer_name,
                )
            if conv_layer_number == 0:
                network.add(layers.TimeDistributed(conv_layer, input_shape=input_shape))
            else:
                network.add(layers.TimeDistributed(conv_layer))

            # add pooling layer
            pooling_layer_name = "pooling_" + str(conv_layer_number + 1)
            pooling_layer = layers.MaxPooling2D(2, 2, name=pooling_layer_name)
            network.add(layers.TimeDistributed(pooling_layer))
        # FLATTENING
        flatten_layer_name = "flatten"
        flatten_layer = layers.Flatten(name=flatten_layer_name)
        network.add(layers.TimeDistributed(flatten_layer))

        # DENSE TOP
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layers_dimensions)), dense_layers_dimensions
        ):

            # add dense layer
            dense_layer_name = "dense_" + str(dense_layer_number + 1)
            dense_layer = layers.Dense(
                dense_layer_dimension, activation="relu", name=dense_layer_name
            )
            network.add(layers.TimeDistributed(dense_layer))

        for rnn_layer_number, rnn_layer_dimension in zip(
            range(len(rnn_layers_dimensions)), rnn_layers_dimensions
        ):

            # add dense layer
            rnn_layer_name = "rnn_" + str(rnn_layer_number + 1)
            rnn_layer = layers.LSTM(
                rnn_layer_dimension,
                name=rnn_layer_name,
                return_sequences=rnn_layer_number < len(rnn_layers_dimensions) - 1
                or return_sequences,
            )
            network.add(rnn_layer)

        # OUTPUT LAYER

        output_layer = layers.Dense(
            number_of_outputs, activation=output_activation, name="output"
        )
        if return_sequences:
            network.add(layers.TimeDistributed(output_layer))
        else:
            network.add(output_layer)

        super().__init__(network, **kwargs)

    def data_generator(self, *args, **kwargs):
        return super().data_generator(*args, **{"ndim": 5, **kwargs})


# Alias for backwards compatability
rnn = RNN