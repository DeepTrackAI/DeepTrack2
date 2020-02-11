from tensorflow.keras import models, layers, optimizers
from deeptrack.losses import nd_mean_absolute_error


def convolutional(
        input_shape=(51, 51, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        dense_layers_dimensions=(32, 32),
        number_of_outputs=3,
        output_activation=None,
        loss="mse"):
    """Creates and compiles a deep learning network.

    Inputs:
    input_shape: size of the images to be analyzed [3-ple of positive integers, x-pixels by y-pixels by color channels]
    conv_layers_dimensions: number of convolutions in each convolutional layer [tuple of positive integers]
    dense_layers_dimensions: number of units in each dense layer [tuple of positive integers]

    Output:
    network: deep learning network
    """

    ### INITIALIZE DEEP LEARNING NETWORK
    network = models.Sequential()

    ### CONVOLUTIONAL BASIS
    for conv_layer_number, conv_layer_dimension in zip(range(len(conv_layers_dimensions)), conv_layers_dimensions):

        # add convolutional layer
        conv_layer_name = 'conv_' + str(conv_layer_number + 1)
        if conv_layer_number == 0:
            conv_layer = layers.Conv2D(conv_layer_dimension,
                                       (3, 3),
                                       activation='relu',
                                       input_shape=input_shape,
                                       name=conv_layer_name)
        else:
            conv_layer = layers.Conv2D(conv_layer_dimension,
                                       (3, 3), 
                                       activation='relu',
                                       name=conv_layer_name)
        network.add(conv_layer)

        # add pooling layer
        pooling_layer_name = 'pooling_' + str(conv_layer_number+1)
        pooling_layer = layers.MaxPooling2D(2, 2, name=pooling_layer_name)
        network.add(pooling_layer)

    # FLATTENING
    flatten_layer_name = 'flatten'
    flatten_layer = layers.Flatten(name=flatten_layer_name)
    network.add(flatten_layer)

    # DENSE TOP
    for dense_layer_number, dense_layer_dimension in zip(range(len(dense_layers_dimensions)), dense_layers_dimensions):

        # add dense layer
        dense_layer_name = 'dense_' + str(dense_layer_number + 1)
        dense_layer = layers.Dense(dense_layer_dimension, 
                                   activation='relu', 
                                   name=dense_layer_name)
        network.add(dense_layer)

    # OUTPUT LAYER

    output_layer = layers.Dense(number_of_outputs, activation=output_activation, name='output')
    network.add(output_layer)

    network.compile(optimizer='rmsprop', loss=loss, metrics=['mse', 'mae'])

    return network



def unet(
        input_shape=(None, None, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        base_conv_layers_dimensions=(128, 128),
        output_conv_layers_dimensions=(16, 16),
        steps_per_pooling=1,
        number_of_outputs=1,
        layer_function=None,
        output_activation="sigmoid",    
        loss=nd_mean_absolute_error):

    if layer_function is None:
        layer_function = lambda dimensions: layers.Conv2D(
            conv_layer_dimension,
            kernel_size=3,
            activation="relu",
            padding="same"
        )

    unet_input = layers.Input(input_shape)

    concat_layers = []

    layer = unet_input

    # Downsampling step
    for conv_layer_dimension in conv_layers_dimensions:
        for _ in range(steps_per_pooling):
            layer = layer_function(conv_layer_dimension)(layer)
        concat_layers.append(layer)
        layer = layers.MaxPooling2D(2)(layer)

    # Base steps
    for conv_layer_dimension in base_conv_layers_dimensions:
        layer = layer_function(conv_layer_dimension)(layer)

    # Upsampling step
    for conv_layer_dimension, concat_layer in zip(reversed(conv_layers_dimensions), reversed(concat_layers)):

        layer = layers.Conv2DTranspose(conv_layer_dimension,
                                       kernel_size=2,
                                       strides=2)(layer)

        layer = layers.Concatenate(axis=-1)([layer, concat_layer])
        for _ in range(steps_per_pooling):
            layer = layer_function(conv_layer_dimension)(layer)

    # Output step
    for conv_layer_dimension in output_conv_layers_dimensions:
        layer = layer_function(conv_layer_dimension)(layer)

    layer = layers.Conv2D(
        number_of_outputs,
        kernel_size=3,
        activation=output_activation,
        padding="same")(layer)

    model = models.Model(unet_input, layer)

    model.compile(optimizers.Adam(lr=0.001, amsgrad=True), loss=loss)
    return model


def rnn(
    input_shape=(51, 51, 1),
    conv_layers_dimensions=(16, 32, 64, 128),
    dense_layers_dimensions=(32, ),
    rnn_layers_dimensions=(32, ),
    return_sequences=False,
    output_activation="sigmoid",
    number_of_outputs=3,
    loss="mse",
    data_format="channels_last"):

    ### INITIALIZE DEEP LEARNING NETWORK
    network = models.Sequential()

    ### CONVOLUTIONAL BASIS
    for conv_layer_number, conv_layer_dimension in zip(range(len(conv_layers_dimensions)), conv_layers_dimensions):

        
        # add convolutional layer
        conv_layer_name = 'conv_' + str(conv_layer_number + 1)
        if conv_layer_number == 0:
            conv_layer = layers.Conv2D(conv_layer_dimension,
                                       (3, 3),
                                       activation='relu',
                                       padding="same",
                                       name=conv_layer_name)
        else:
            conv_layer = layers.Conv2D(conv_layer_dimension,
                                       (3, 3), 
                                       activation='relu',
                                       padding="same",
                                       name=conv_layer_name)
        if conv_layer_number == 0:
            network.add(layers.TimeDistributed(conv_layer, input_shape=input_shape))
        else:
            network.add(layers.TimeDistributed(conv_layer))

        # add pooling layer
        pooling_layer_name = 'pooling_' + str(conv_layer_number+1)
        pooling_layer = layers.MaxPooling2D(2, 2, name=pooling_layer_name)
        network.add(layers.TimeDistributed(pooling_layer))
    # FLATTENING
    flatten_layer_name = 'flatten'
    flatten_layer = layers.Flatten(name=flatten_layer_name)
    network.add(layers.TimeDistributed(flatten_layer))

    # DENSE TOP
    for dense_layer_number, dense_layer_dimension in zip(range(len(dense_layers_dimensions)), dense_layers_dimensions):

        # add dense layer
        dense_layer_name = 'dense_' + str(dense_layer_number + 1)
        dense_layer = layers.Dense(dense_layer_dimension, 
                                   activation='relu', 
                                   name=dense_layer_name)
        network.add(layers.TimeDistributed(dense_layer))

    for rnn_layer_number, rnn_layer_dimension in zip(range(len(rnn_layers_dimensions)), rnn_layers_dimensions):

        # add dense layer
        rnn_layer_name = 'rnn_' + str(rnn_layer_number + 1)
        rnn_layer = layers.LSTM(rnn_layer_dimension,
                                   name=rnn_layer_name,
                                   return_sequences = rnn_layer_number < len(rnn_layers_dimensions) - 1 or return_sequences
                                   )
        network.add(rnn_layer)

    # OUTPUT LAYER

    output_layer = layers.Dense(number_of_outputs, activation=output_activation, name='output')
    if return_sequences:
        network.add(layers.TimeDistributed(output_layer))
    else:
        network.add(output_layer)

    network.compile(optimizer=optimizers.Adam(lr=0.0005, amsgrad=True), loss=loss, metrics=['mse', 'mae'])

    return network