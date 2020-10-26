""" Standard models for neural networks.

Classes
-------
ModelFeature 
    Base model feature class.
Convolutional, convolutional
    Creates and compiles a convolutional neural network.
UNet, unet
    Creates and compiles a U-Net neural network.
RNN, rnn
    Creates and compiles a recurrent neural network.
"""

from deeptrack.losses import nd_mean_absolute_error
from deeptrack.features import Feature
from deeptrack.layers import as_block
from tensorflow.keras import models, layers, optimizers
import numpy as np
import warnings


def _compile(
    model: models.Model, *, loss="mae", optimizer="adam", metrics=[], **kwargs
):
    """Compiles a model.

    Parameters
    ----------
    model : keras.models.Model
        The keras model to interface.
    loss : str or keras loss
        The loss function of the model.
    optimizer : str or keras optimizer
        The optimizer of the model.
    metrics : list, optional
    """

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


class Model(Feature):
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def __getattr__(self, key):
        # Allows access to the model methods and properties
        try:
            return getattr(super(), key)
        except AttributeError:
            return getattr(self.model, key)


class KerasModel(Model):
    def __init__(
        self,
        model,
        loss="mae",
        optimizer="adam",
        metrics=[],
        compile=True,
        add_batch_dimension_on_resolve=True,
        **kwargs
    ):

        if compile:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        super().__init__(
            model,
            add_batch_dimension_on_resolve=add_batch_dimension_on_resolve,
            metrics=metrics,
            **kwargs
        )

    def get(self, image, add_batch_dimension_on_resolve, **kwargs):
        if add_batch_dimension_on_resolve:
            image = np.expand_dims(image, axis=0)

        return self.model.predict(image)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def LoadModel(path, compile_from_file=False, custom_objects={}, **kwargs):
    """Loads a keras model from disk.

    Parameters
    ----------
    path : str
        Path to the keras model to load.
    compile_from_file : bool
        Whether to compile the model using the loss and optimizer in the saved model. If false,
        it will be compiled from the arguments in kwargs (loss, optimizer and metrics).
    custom_objects : dict
        Dict of objects to use when loading the model. Needed to load a model with a custom loss,
        optimizer or metric.
    """
    model = models.load_model(
        path, compile=compile_from_file, custom_objects=custom_objects
    )
    return KerasModel(model, compile=not compile_from_file, **kwargs)


def FullyConnected(
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

    ### INITIALIZE DEEP LEARNING NETWORK
    input_layer = layers.Input(shape=input_shape)

    layer = input_layer
    if flatten_input:
        layer = layers.Flatten()(layer)

    # DENSE TOP
    for dense_layer_number, dense_layer_dimension in zip(
        range(len(dense_layers_dimensions)), dense_layers_dimensions
    ):

        if dense_layer_number is 0 and not flatten_input:
            layer = dense_block(dense_layer_dimension, input_shape=input_shape)(layer)
        else:
            layer = dense_block(dense_layer_dimension)(layer)

        if dropout:
            layer = layers.Dropout(dropout[0])(layer)
            dropout = dropout[1:]

    # OUTPUT LAYER

    output_layer = layers.Dense(number_of_outputs, activation=output_activation)(layer)

    model = models.Model(input_layer, output_layer)

    return KerasModel(model, **kwargs)


def Convolutional(
    input_shape=(51, 51, 1),
    conv_layers_dimensions=(16, 32, 64, 128),
    dense_layers_dimensions=(32, 32),
    steps_per_pooling=1,
    dropout=(),
    dense_top=True,
    number_of_outputs=3,
    output_activation=None,
    output_kernel_size=3,
    loss=nd_mean_absolute_error,
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

    ### INITIALIZE DEEP LEARNING NETWORK

    if isinstance(input_shape, list):
        network_input = [layers.Input(shape) for shape in input_shape]
        inputs = layers.Concatenate(axis=-1)(network_input)
    else:
        network_input = layers.Input(input_shape)
        inputs = network_input

    layer = inputs

    ### CONVOLUTIONAL BASIS
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
        output_layer = layers.Dense(number_of_outputs, activation=output_activation)(
            layer
        )
    else:

        output_layer = layers.Conv2D(
            number_of_outputs,
            kernel_size=output_kernel_size,
            activation=output_activation,
            padding="same",
            name="output",
        )(layer)

    model = models.Model(network_input, output_layer)

    return KerasModel(model, loss=loss, **kwargs)


# Alias for backwards compatability
convolutional = Convolutional


def UNet(
    input_shape=(None, None, 1),
    conv_layers_dimensions=(16, 32, 64, 128),
    base_conv_layers_dimensions=(128, 128),
    output_conv_layers_dimensions=(16, 16),
    dropout=(),
    steps_per_pooling=1,
    number_of_outputs=1,
    output_kernel_size=3,
    output_activation=None,
    loss=nd_mean_absolute_error,
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

    return KerasModel(model, loss=loss, **kwargs)


# Alias for backwards compatability
unet = UNet


def RNN(
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

    ### INITIALIZE DEEP LEARNING NETWORK
    network = models.Sequential()

    ### CONVOLUTIONAL BASIS
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

    return KerasModel(network, **kwargs)


# Alias for backwards compatability
rnn = RNN


class cgan(Model):
    def __init__(
        self,
        generator=None,
        discriminator=None,
        discriminator_loss=None,
        discriminator_optimizer=None,
        discriminator_metrics=None,
        assemble_loss=None,
        assemble_optimizer=None,
        assemble_loss_weights=None,
        **kwargs
    ):

        # Build and compile the discriminator
        self.discriminator = discriminator
        self.discriminator.compile(
            loss=discriminator_loss,
            optimizer=discriminator_optimizer,
            metrics=discriminator_metrics,
        )

        # Build the generator
        self.generator = generator

        # Input shape
        self.model_input = self.generator.input

        # The generator model_input and generates img
        img = self.generator(self.model_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes the generated images as input and determines validity
        validity = self.discriminator([img, self.model_input])

        # The assembled model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.assemble = models.Model(self.model_input, [validity, img])
        self.assemble.compile(
            loss=assemble_loss,
            optimizer=assemble_optimizer,
            loss_weights=assemble_loss_weights,
        )

        super().__init__(self.generator, **kwargs)

    def fit(self, data_generator, epochs, steps_per_epoch=None, **kwargs):
        for key in kwargs.keys():
            warnings.warn(
                "{0} not implemented for cgan. Does not affect the execution.".format(
                    key
                )
            )
        for epoch in range(epochs):
            if not steps_per_epoch:
                try:
                    steps = len(data_generator)
                except:
                    steps = 1
            else:
                steps = steps_per_epoch

            d_loss = 0
            g_loss = 0

            for step in range(steps):
                # update data
                try:
                    data, labels = next(data_generator)
                except:
                    data, labels = data_generator[step]

                # Grab disriminator labels
                shape = (data.shape[0], *self.discriminator.output.shape[1:])
                valid, fake = np.ones(shape), np.zeros(shape)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Generate a batch of new images
                gen_imgs = self.generator(data)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([labels, data], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, data], fake)
                d_loss += 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (to have the discriminator label samples as valid)
                g_loss += np.array(self.assemble.train_on_batch(data, [valid, labels]))

                # Plot the progress

            try:
                data_generator.on_epoch_end()
            except:
                pass

            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f, %f, %f]"
                % (
                    epoch,
                    d_loss[0] / steps,
                    100 * d_loss[1] / steps,
                    g_loss[0] / steps,
                    g_loss[1] / steps,
                    g_loss[2] / steps,
                )
            )
