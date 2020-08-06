''' Standard models for neural networks.

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
'''

from deeptrack.losses import nd_mean_absolute_error
from deeptrack.features import Feature

from keras import models, layers, optimizers
import numpy as np

def _compile(model: models.Model, 
            *,
            loss="mae", 
            optimizer="adam", 
            metrics=[],
            **kwargs):
    ''' Compiles a model.

    Parameters
    ----------
    model : keras.models.Model
        The keras model to interface.
    loss : str or keras loss
        The loss function of the model.
    optimizer : str or keras optimizer
        The optimizer of the model.
    metrics : list, optional
    '''

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
    def __init__(self, model,
                    loss="mae", 
                    optimizer="adam", 
                    metrics=[],
                    compile=True,
                    add_batch_dimension_on_resolve=True,
                    **kwargs):

        if compile:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        super().__init__(model, add_batch_dimension_on_resolve=add_batch_dimension_on_resolve, metrics=metrics, **kwargs)

    def get(self, image, add_batch_dimension_on_resolve, **kwargs):
        if add_batch_dimension_on_resolve:
            image = np.expand_dims(image, axis=0)
        
        return self.model.predict(image)



def FullyConnected(input_shape,
                 dense_layers_dimensions=(32, 32),
                 dropout=(),
                 flatten_input=True,
                 number_of_outputs=3,
                 output_activation=None,
                 **kwargs):
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
    loss : str or keras loss function
        The loss function of the network.

    Returns
    -------
    keras.models.Model 
        Deep learning network
    """

    ### INITIALIZE DEEP LEARNING NETWORK
    network = models.Sequential()
    if flatten_input:
        network.add(layers.Flatten(input_shape=input_shape))

    # DENSE TOP
    for dense_layer_number, dense_layer_dimension in zip(range(len(dense_layers_dimensions)), dense_layers_dimensions):

        # add dense layer
        dense_layer_name = 'dense_' + str(dense_layer_number + 1)
        if dense_layer_number is 0 and not flatten_input:
            dense_layer = layers.Dense(dense_layer_dimension, 
                                    activation='tanh', 
                                    name=dense_layer_name,
                                    input_shape=input_shape)
        else:
            dense_layer = layers.Dense(dense_layer_dimension, 
                                    activation='tanh', 
                                    name=dense_layer_name)
        network.add(dense_layer)

        if dropout:
            network.add(layers.Dropout(dropout[0]))
            dropout = dropout[1:]

    # OUTPUT LAYER

    output_layer = layers.Dense(number_of_outputs, activation=output_activation, name='output')
    network.add(output_layer)

    
    return KerasModel(network, **kwargs)


def Convolutional(input_shape=(51, 51, 1),
                 conv_layers_dimensions=(16, 32, 64, 128),
                 dense_layers_dimensions=(32, 32),
                 steps_per_pooling=1,
                 dropout=(),
                 number_of_outputs=3,
                 output_activation=None,
                 **kwargs):
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

    Returns
    -------
    keras.models.Model 
        Deep learning network
    """

    ### INITIALIZE DEEP LEARNING NETWORK
    network = models.Sequential()
    
    ### CONVOLUTIONAL BASIS
    for conv_layer_number, conv_layer_dimension in zip(range(len(conv_layers_dimensions)), conv_layers_dimensions):
        
        for step in range(steps_per_pooling):
        # add convolutional layer
            try:
                if conv_layer_number == 0 and step == 0:
                    conv_layer = layers.Conv2D(conv_layer_dimension,
                                            (3, 3),
                                            activation='relu',
                                            input_shape=input_shape,
                                            padding="same")
                else:
                    conv_layer = layers.Conv2D(conv_layer_dimension,
                                            (3, 3), 
                                            activation='relu',
                                            padding="same")
                
                network.add(conv_layer)
            except Exception as e:
                print(e)
            
        
        if dropout:
            network.add(layers.SpatialDropout2D(dropout[0]))
            dropout = dropout[1:]

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

    
    return KerasModel(network, **kwargs)


# Alias for backwards compatability
convolutional = Convolutional




def UNet(input_shape=(None, None, 1),
            conv_layers_dimensions=(16, 32, 64, 128),
            base_conv_layers_dimensions=(128, 128),
            output_conv_layers_dimensions=(16, 16),
            dropout=(),
            steps_per_pooling=1,
            number_of_outputs=1,
            output_activation=None,
            loss=nd_mean_absolute_error,
            layer_function=None,
            **kwargs):
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

    if layer_function is None:
        layer_function = lambda dimensions: layers.Conv2D(
            dimensions,
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

        if dropout:
            layer = layers.SpatialDropout2D(dropout[0])(layer)
            dropout = dropout[1:]

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
    
    
    return KerasModel(model, loss=loss, **kwargs)



# Alias for backwards compatability
unet = UNet



def RNN(input_shape=(51, 51, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        dense_layers_dimensions=(32,),
        rnn_layers_dimensions=(32,),
        return_sequences=False,
        output_activation=None,
        number_of_outputs=3,
        **kwargs):
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

    return KerasModel(network, **kwargs)



# Alias for backwards compatability
rnn = RNN

class GAN(Model):
    def __init__(self, generator: Feature, discriminator: Feature, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        super().__init__(generator, **kwargs)

    def fit(self, generatorSeed, Y=None, batch_size=32, validation_data=None):
        if Y and len(X) == len(Y):
            X = []