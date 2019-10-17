from tensorflow.keras import models, layers

def DeepTrackNetwork(
    input_shape = (51, 51, 1),
    conv_layers_dimensions = (16, 32, 64, 128), 
    dense_layers_dimensions = (32, 32),
    number_of_outputs = 3):
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
        if conv_layer_number==0:
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
    
    output_layer = layers.Dense(number_of_outputs, name='output')
    network.add(output_layer)
    
    network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
    
    return network