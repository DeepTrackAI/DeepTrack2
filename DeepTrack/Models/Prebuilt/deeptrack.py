''' DeepTrack 1.0
Digital Video Microscopy enhanced with Deep Learning
version 1.0 - 15 November 2018
© Saga Helgadottir, Aykut Argun & Giovanni Volpe
http://www.softmatterlab.org
'''

def get_image_parameters(
    particle_center_x_list=lambda : [0, ], 
    particle_center_y_list=lambda : [0, ], 
    particle_radius_list=lambda : [3, ], 
    particle_bessel_orders_list=lambda : [[1, ], ], 
    particle_intensities_list=lambda : [[.5, ], ],
    image_half_size=lambda : 25, 
    image_background_level=lambda : .5,
    signal_to_noise_ratio=lambda : 30,
    gradient_intensity=lambda : .2, 
    gradient_direction=lambda : 0,
    ellipsoidal_orientation=lambda : [0, ], 
    ellipticity=lambda : 1):
    """Get image parameters.
    
    Inputs:
    particle_center_x_list: x-centers of the particles [px, list of real numbers]
    particle_center_y_list: y-centers of the particles [px, list of real numbers]
    particle_radius_list: radii of the particles [px, list of real numbers]
    particle_bessel_orders_list: Bessel orders of the particles [list (of lists) of positive integers]
    particle_intensities_list: intensities of the particles [list (of lists) of real numbers, normalized to 1]
    image_half_size: half size of the image in pixels [px, positive integer]
    image_background_level: background level [real number normalized to 1]
    signal_to_noise_ratio: signal to noise ratio [positive real number]
    gradient_intensity: gradient intensity [real number normalized to 1]
    gradient_direction: gradient angle [rad, real number]
    ellipsoidal_orientation: Orientation of elliptical particles [rad, real number] 
    ellipticity: shape of the particles, from spherical to elliptical [real number]
    
    Note: particle_center_x, particle_center_x, particle_radius, 
    particle_bessel_order, particle_intensity, ellipsoidal_orientation must have the same length.
    
    Output:
    image_parameters: list with the values of the image parameters in a dictionary:
        image_parameters['Particle Center X List']
        image_parameters['Particle Center Y List']
        image_parameters['Particle Radius List']
        image_parameters['Particle Bessel Orders List']
        image_parameters['Particle Intensities List']
        image_parameters['Image Half-Size']
        image_parameters['Image Background Level']
        image_parameters['Signal to Noise Ratio']
        image_parameters['Gradient Intensity']
        image_parameters['Gradient Direction']
        image_parameters['Ellipsoid Orientation']
        image_parameters['Ellipticity']
    """
    
    image_parameters = {}
    image_parameters['Particle Center X List'] = particle_center_x_list()
    image_parameters['Particle Center Y List'] = particle_center_y_list()
    image_parameters['Particle Radius List'] = particle_radius_list()
    image_parameters['Particle Bessel Orders List'] = particle_bessel_orders_list()
    image_parameters['Particle Intensities List'] = particle_intensities_list()
    image_parameters['Image Half-Size'] = image_half_size()
    image_parameters['Image Background Level'] = image_background_level()
    image_parameters['Signal to Noise Ratio'] = signal_to_noise_ratio()
    image_parameters['Gradient Intensity'] = gradient_intensity()
    image_parameters['Gradient Direction'] = gradient_direction()
    image_parameters['Ellipsoid Orientation'] = ellipsoidal_orientation()
    image_parameters['Ellipticity'] = ellipticity()

    return image_parameters

def generate_image(image_parameters):
    """Generate image with particles.
    
    Input:
    image_parameters: list with the values of the image parameters in a dictionary:
        image_parameters['Particle Center X List']
        image_parameters['Particle Center Y List']
        image_parameters['Particle Radius List']
        image_parameters['Particle Bessel Orders List']
        image_parameters['Particle Intensities List']
        image_parameters['Image Half-Size']
        image_parameters['Image Background Level']
        image_parameters['Signal to Noise Ratio']
        image_parameters['Gradient Intensity']
        image_parameters['Gradient Direction']
        image_parameters['Ellipsoid Orientation']
        image_parameters['Ellipticity']
        
    Note: image_parameters is typically obained from the function get_image_parameters()
        
    Output:
    image: image of the particle [2D numpy array of real numbers betwen 0 and 1]
    """
    
    from numpy import meshgrid, arange, ones, zeros, sin, cos, sqrt, clip, array
    from scipy.special import jv as bessel
    from numpy.random import poisson as poisson
    
    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    particle_bessel_orders_list = image_parameters['Particle Bessel Orders List']
    particle_intensities_list = image_parameters['Particle Intensities List']
    image_half_size = image_parameters['Image Half-Size'] 
    image_background_level = image_parameters['Image Background Level']
    signal_to_noise_ratio = image_parameters['Signal to Noise Ratio']
    gradient_intensity = image_parameters['Gradient Intensity']
    gradient_direction = image_parameters['Gradient Direction']
    ellipsoidal_orientation_list = image_parameters['Ellipsoid Orientation']
    ellipticity = image_parameters['Ellipticity']
    
    ### CALCULATE IMAGE PARAMETERS
    # calculate image full size
    image_size = image_half_size * 2 + 1

    # calculate matrix coordinates from the center of the image
    image_coordinate_x, image_coordinate_y = meshgrid(arange(-image_half_size, image_half_size + 1), 
                                                      arange(-image_half_size, image_half_size + 1), 
                                                      sparse=False, 
                                                      indexing='ij')

    ### CALCULATE BACKGROUND
    # initialize the image at the background level
    image_background = ones((image_size, image_size)) * image_background_level
    
    # add gradient to image background
    if gradient_intensity!=0:
        image_background = image_background + gradient_intensity * (image_coordinate_x * sin(gradient_direction) + 
                                                                    image_coordinate_y * cos(gradient_direction) ) / (sqrt(2) * image_size)

    ### CALCULATE IMAGE PARTICLES
    image_particles = zeros((image_size, image_size))
    for particle_center_x, particle_center_y, particle_radius, particle_bessel_orders, particle_intensities, ellipsoidal_orientation in zip(particle_center_x_list, particle_center_y_list, particle_radius_list, particle_bessel_orders_list, particle_intensities_list, ellipsoidal_orientation_list):
        # calculate the radial distance from the center of the particle 
        # normalized by the particle radius
        radial_distance_from_particle = sqrt((image_coordinate_x - particle_center_x)**2 
                                         + (image_coordinate_y - particle_center_y)**2 
                                         + .001**2) / particle_radius
        
        # for elliptical particles
        rotated_distance_x = (image_coordinate_x - particle_center_x)*cos(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*sin(ellipsoidal_orientation)
        rotated_distance_y = -(image_coordinate_x - particle_center_x)*sin(ellipsoidal_orientation) + (image_coordinate_y - particle_center_y)*cos(ellipsoidal_orientation)
        
        
        elliptical_distance_from_particle = sqrt((rotated_distance_x)**2 
                                         + (rotated_distance_y / ellipticity)**2 
                                         + .001**2) / particle_radius

        # calculate particle profile
        for particle_bessel_order, particle_intensity in zip(particle_bessel_orders, particle_intensities):
            image_particle = 4 * particle_bessel_order**2.5 * (bessel(particle_bessel_order, elliptical_distance_from_particle) / elliptical_distance_from_particle)**2
            image_particles = image_particles + particle_intensity * image_particle

    # calculate image without noise as background image plus particle image
    image_particles_without_noise = clip(image_background + image_particles, 0, 1)

    ### ADD NOISE
    image_particles_with_noise = poisson(image_particles_without_noise * signal_to_noise_ratio**2) / signal_to_noise_ratio**2
    
    return image_particles_with_noise

def get_image_generator(image_parameters_function=lambda : get_image_parameters(), max_number_of_images=1e+9):
    """Generator of particle images.
    
    Inputs:
    image_parameters_function: lambda function to generate the image parameters (this is typically get_image_parameters())
    max_number_of_images: maximum number of images to be generated (positive integer)
        
    Outputs:
    image_number: image number in the current generation cycle
    image: image of the particles [2D numpy array of real numebrs betwen 0 and 1]
    image_parameters: list with the values of the image parameters in a dictionary:
        image_parameters['Particle Center X List']
        image_parameters['Particle Center Y List']
        image_parameters['Particle Radius List']
        image_parameters['Particle Bessel Orders List']
        image_parameters['Particle Intensities List']
        image_parameters['Image Half-Size']
        image_parameters['Image Background Level']
        image_parameters['Signal to Noise Ratio']
        image_parameters['Gradient Intensity']
        image_parameters['Gradient Direction']
        image_parameters['Ellipsoid Orientation']
        image_parameters['Ellipticity']
    """    
    
    image_number = 0
    while image_number<max_number_of_images:
        
        image_parameters = image_parameters_function()
        image = generate_image(image_parameters)

        yield image_number, image, image_parameters
        image_number += 1
        
def plot_sample_image(image, image_parameters, figsize=(15,5)):
    """Plot a sample image.
    
    Inputs:
    image: image of the particles
    image_parameters: list with the values of the image parameters
    figsize: figure size [list of two positive numbers]
    
    
    Output: none
    """
    
    import matplotlib.pyplot as plt 

    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    particle_bessel_orders_list = image_parameters['Particle Bessel Orders List']
    particle_intensities_list = image_parameters['Particle Intensities List']
    image_half_size = image_parameters['Image Half-Size'] 
    image_background_level = image_parameters['Image Background Level']
    signal_to_noise_ratio = image_parameters['Signal to Noise Ratio']
    gradient_intensity = image_parameters['Gradient Intensity']
    gradient_direction = image_parameters['Gradient Direction']
    ellipsoidal_orientation_list = image_parameters['Ellipsoid Orientation']
    ellipticity = image_parameters['Ellipticity']

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='equal',
               extent=(-image_half_size, image_half_size, -image_half_size, image_half_size))
    plt.plot(particle_center_y_list[0], particle_center_x_list[0], 'o', color='r')
    plt.xlabel('y (px)', fontsize=16)
    plt.ylabel('x (px)', fontsize=16)

    subplot131_handle = plt.subplot(1, 3, 2)
    plt.text(0, .9, 'particle center x = %5.2f px' % particle_center_x_list[0], fontsize=16)
    plt.text(0, .8, 'particle center y = %5.2f px' % particle_center_y_list[0], fontsize=16)
    plt.text(0, .7, 'particle radius = %5.2f px' % particle_radius_list[0], fontsize=16)
    plt.text(0, .6, 'Bessel order = %5.2f' % particle_bessel_orders_list[0][0], fontsize=16)
    plt.text(0, .5, 'particle intensity = %5.2f' % particle_intensities_list[0][0], fontsize=16)
    plt.text(0, .4, 'ellipsoidal_orientation = %5.2f' % ellipsoidal_orientation_list[0], fontsize=16)
    plt.text(0, .3, 'ellipticity = %5.2f' % ellipticity, fontsize=16)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.text(0, .9, 'image half size = %5.2f px' % image_half_size, fontsize=16)
    plt.text(0, .8, 'image background level = %5.2f' % image_background_level, fontsize=16)
    plt.text(0, .7, 'signal to noise ratio = %5.2f' % signal_to_noise_ratio, fontsize=16)
    plt.text(0, .6, 'gradient intensity = %5.2f' % gradient_intensity, fontsize=16)
    plt.text(0, .5, 'gradient direction = %5.2f' % gradient_direction, fontsize=16)
    plt.axis('off')

    plt.show()
    
def create_deep_learning_network(
    input_shape = (51, 51, 1),
    conv_layers_dimensions = (16, 32, 64, 128), 
    dense_layers_dimensions = (32, 32)):
    """Creates and compiles a deep learning network.
    
    Inputs:    
    input_shape: size of the images to be analyzed [3-ple of positive integers, x-pixels by y-pixels by color channels]
    conv_layers_dimensions: number of convolutions in each convolutional layer [tuple of positive integers]
    dense_layers_dimensions: number of units in each dense layer [tuple of positive integers]
        
    Output:
    network: deep learning network
    """    

    from keras import models, layers
    
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
    number_of_outputs = 3
    
    output_layer = layers.Dense(number_of_outputs, name='output')
    network.add(output_layer)
    
    network.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
    
    return network

def resize_image(image, final_image_shape):
    """Utility to resize an image.
    
    Input:
    image: original image (2D numpy array of real numbers)
    final_image_shape: final size of the resized image (2-ple of positive integers)
        
    Output:
    resized_image: resized image (2D numpy array of real numbers)
    """    

    from numpy import asarray
    from PIL import Image
    
    img = Image.fromarray(image).resize(final_image_shape)
    resized_image = asarray(img)

    return resized_image

def train_deep_learning_network(
    network,
    image_generator,
    sample_sizes = (32, 128, 512, 2048),
    iteration_numbers = (3001, 2001, 1001, 101),
    verbose=True):
    """Train a deep learning network.
    
    Input:
    network: deep learning network
    image_generator: image generator
    sample_sizes: sizes of the batches of images used in the training [tuple of positive integers]
    iteration_numbers: numbers of batches used in the training [tuple of positive integers]
    verbose: frequency of the update messages [number between 0 and 1]
        
    Output:
    training_history: dictionary with training history
    
    Note: The MSE is in px^2 and the MAE in px
    """  
    
    import numpy as np
    from time import time
    
    number_of_outputs = 3
    
    training_history = {}
    training_history['Sample Size'] = []
    training_history['Iteration Number'] = []
    training_history['Iteration Time'] = []
    training_history['MSE'] = []
    training_history['MAE'] = []
    
    for sample_size, iteration_number in zip(sample_sizes, iteration_numbers):
        for iteration in range(iteration_number):
            
            # meaure initial time for iteration
            initial_time = time()

            # generate images and targets
            image_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]
            
            input_shape = (sample_size, image_shape[0], image_shape[1], image_shape[2])
            images = np.zeros(input_shape)
            
            output_shape = (sample_size, number_of_outputs)
            targets = np.zeros(output_shape)
            
            for image_number, image, image_parameters in image_generator():
                if image_number>=sample_size:
                    break
                    
                resized_image = resize_image(image, (image_shape[0], image_shape[1]))
                images[image_number] = resized_image.reshape(image_shape)
                
                half_image_size = (image_shape[0] - 1) / 2
                particle_center_x = image_parameters['Particle Center X List'][0]
                particle_center_y = image_parameters['Particle Center Y List'][0]
                
                targets[image_number] = [particle_center_x / half_image_size,
                                         particle_center_y / half_image_size,
                                         (particle_center_x**2 + particle_center_y**2)**.5 / half_image_size]

            # training
            history = network.fit(images,
                                targets,
                                epochs=1, 
                                batch_size=sample_size,
                                verbose=False)
                        
            # measure elapsed time during iteration
            iteration_time = time() - initial_time

            # record training history
            mse = history.history['mean_squared_error'][0] * half_image_size**2
            mae = history.history['mean_absolute_error'][0] * half_image_size
                        
            training_history['Sample Size'].append(sample_size)
            training_history['Iteration Number'].append(iteration)
            training_history['Iteration Time'].append(iteration_time)
            training_history['MSE'].append(mse)
            training_history['MAE'].append(mae)

            if not(iteration%int(verbose**-1)):
                print('Sample size %6d   iteration number %6d   MSE %10.2f px^2   MAE %10.2f px   Time %10.2f ms' % (sample_size, iteration + 1, mse, mae, iteration_time * 1000))
                
    return training_history

def plot_learning_performance(training_history, number_of_timesteps_for_average = 100, figsize=(20,20)):
    """Plot the learning performance of the deep learning network.
    
    Input:
    training_history: dictionary with training history, typically obtained from train_deep_learning_network()
    number_of_timesteps_for_average: length of the average [positive integer number]
    figsize: figure size [list of two positive numbers]
        
    Output: none
    """    

    import matplotlib.pyplot as plt
    from numpy import convolve, ones
    
    plt.figure(figsize=figsize)

    plt.subplot(5, 1, 1)
    plt.semilogy(training_history['MSE'], 'k')
    plt.semilogy(convolve(training_history['MSE'], ones(number_of_timesteps_for_average) / number_of_timesteps_for_average, mode='valid'), 'r')
    plt.ylabel('MSE (px^2)', fontsize=24)

    plt.subplot(5, 1, 2)
    plt.semilogy(training_history['MAE'], 'k')
    plt.semilogy(convolve(training_history['MAE'], ones(number_of_timesteps_for_average) / number_of_timesteps_for_average, mode='valid'), 'r')
    plt.ylabel('MAE (px)', fontsize=24)

    plt.subplot(5, 1, 3)
    plt.plot(training_history['Sample Size'], 'k')
    plt.ylabel('Sample size', fontsize=24)

    plt.subplot(5, 1, 4)
    plt.plot(training_history['Iteration Number'], 'k')
    plt.ylabel('Iteration number', fontsize=24)

    plt.subplot(5, 1, 5)
    plt.plot(training_history['Iteration Time'], 'k')
    plt.ylabel('Iteration time', fontsize=24)

    plt.show()
    
def predict(network, image):
    """ Predict position of particle in a image using the deep learnign network.
    
    Inputs:
    network: deep learning network
    image: image [2D numpy array of real numbers between 0 and 1]
    
    Output:
    predicted_position: predicted position of the particle [1D numpy array containing x, y and r position]
    """
    
    from numpy import reshape
    
    image_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]

    resized_image = resize_image(image, (image_shape[0], image_shape[1]))
    predicted_position = network.predict(reshape(resized_image, (1, image_shape[0], image_shape[1], image_shape[2])))

    half_image_size = (image_shape[0] - 1) / 2
   
    predicted_position = half_image_size * predicted_position[0]
   
        
    return predicted_position
    

def plot_prediction(image, image_parameters, predicted_position, figsize=(15, 5)):
    """Plot a sample image.
    
    Inputs:
    image: image of the particles
    image_parameters: list with the values of the image parameters
    predicted_position: predicted position of the particle [1D numpy array containing x, y and r position]
    figsize: figure size [list of two positive numbers]
        
    Output: none
    """
    
    import matplotlib.pyplot as plt 
    from numpy import sin, cos, arange
    from math import pi

    particle_center_x_list = image_parameters['Particle Center X List']
    particle_center_y_list = image_parameters['Particle Center Y List']
    particle_radius_list = image_parameters['Particle Radius List']
    particle_bessel_orders_list = image_parameters['Particle Bessel Orders List']
    particle_intensities_list = image_parameters['Particle Intensities List']
    image_half_size = image_parameters['Image Half-Size'] 
    image_background_level = image_parameters['Image Background Level']
    signal_to_noise_ratio = image_parameters['Signal to Noise Ratio']
    gradient_intensity = image_parameters['Gradient Intensity']
    gradient_direction = image_parameters['Gradient Direction']
    ellipsoidal_orientation_list = image_parameters['Ellipsoid Orientation']
    ellipticity = image_parameters['Ellipticity']
    
    predicted_position_x = predicted_position[0]
    predicted_position_y = predicted_position[1]
    predicted_position_r = predicted_position[2]
   

    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1, origin='lower', aspect='equal',
               extent=(-image_half_size, image_half_size, -image_half_size, image_half_size))
    plt.plot(particle_center_y_list[0], particle_center_x_list[0], 'o', color='r')
    plt.plot(predicted_position_y, predicted_position_x, 'o', color='#e6661a')
    plt.plot(predicted_position_r * sin(arange(-pi / 30, 2 * pi, pi / 30)), predicted_position_r * cos(arange(-pi / 30, 2 * pi, pi / 30)), ':', color='#e6661a')
    plt.xlabel('y (px)', fontsize=16)
    plt.ylabel('x (px)', fontsize=16)

    subplot131_handle = plt.subplot(1, 3, 2)
    plt.text(0, 1, 'particle center x = %5.2f px' % particle_center_x_list[0], fontsize=16)
    plt.text(0, .9, 'particle center y = %5.2f px' % particle_center_y_list[0], fontsize=16)
    plt.text(0, .8, 'particle radius = %5.2f px' % particle_radius_list[0], fontsize=16)
    plt.text(0, .7, 'Bessel order = %5.2f' % particle_bessel_orders_list[0][0], fontsize=16)
    plt.text(0, .6, 'particle intensity = %5.2f' % particle_intensities_list[0][0], fontsize=16)
    plt.text(0, .5, 'ellipsoidal_orientation = %5.2f' % ellipsoidal_orientation_list[0], fontsize=16)
    plt.text(0, .4, 'ellipticity = %5.2f' % ellipticity, fontsize=16)
    
    plt.text(0, .3, 'predicted x = %5.2f px' % predicted_position_x, fontsize=16, color='#e6661a')
    plt.text(0, .2, 'predicted y = %5.2f px' % predicted_position_y, fontsize=16, color='#e6661a')
    plt.text(0, .1, 'predicted r = %5.2f px' % predicted_position_r, fontsize=16, color='#e6661a')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.text(0, .9, 'image half size = %5.2f px' % image_half_size, fontsize=16)
    plt.text(0, .8, 'image background level = %5.2f' % image_background_level, fontsize=16)
    plt.text(0, .7, 'signal to noise ratio = %5.2f' % signal_to_noise_ratio, fontsize=16)
    plt.text(0, .6, 'gradient intensity = %5.2f' % gradient_intensity, fontsize=16)
    plt.text(0, .5, 'gradient direction = %5.2f rad' % gradient_direction, fontsize=16)
    plt.axis('off')

    plt.show()

def track_frame(
    network,
    frame,
    box_half_size=25,
    box_scanning_step=5,
    ):
    """Tracks a frame box by box.
    
    Inputs:    
    network: the pretrained network
    frame: the frame to by analyzed
    box_half_size: half the size of the scanning box
    box_scanning_step: the size of the scanning step 
    
    Output:
    prediction_wrt_box: x, y and r coordiantes with respect to each box (pixels)
    prediction_wrt_frame: x, y and r coordinates with respect to each frame (pixels)
    boxes: the part of the frame corresponding to each box
    """  

    import numpy as np
    import cv2
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
   
    
    box_center_x = np.arange(box_half_size, 
                             frame_height - box_half_size, 
                             box_scanning_step)
    box_center_y = np.arange(box_half_size, 
                             frame_width - box_half_size, 
                             box_scanning_step)


    boxes = np.zeros((len(box_center_x), 
                      len(box_center_y),
                      box_half_size * 2 + 1, 
                      box_half_size * 2 + 1))  
    
    prediction_wrt_box = np.zeros((len(box_center_x), 
                                   len(box_center_y), 
                                   3)) 
    prediction_wrt_frame = np.zeros((len(box_center_x), 
                                     len(box_center_y), 
                                     3)) 
    
    
    # Scanning over the frame row- and column-wise
    for j in range(len(box_center_x)):
        for k in range(len(box_center_y)):
            
            
            
            # Define the scanning box
            boxes[j, k] = frame[int(box_center_x[j] - box_half_size):int(box_center_x[j] + box_half_size + 1), 
                                int(box_center_y[k] - box_half_size):int(box_center_y[k] + box_half_size + 1)]
            
            box_predict = boxes[j, k]
            
            box_predict = cv2.resize(boxes[j, k], (51, 51))
            
            # Predict position of particle with respect to the scanning box
            prediction_wrt_box[j, k] = network.predict(np.reshape(box_predict, (1, 51, 51, 1)))

            prediction_wrt_box[j, k][0] = prediction_wrt_box[j, k][0] * box_half_size + box_half_size
            prediction_wrt_box[j, k][1] = prediction_wrt_box[j, k][1] * box_half_size + box_half_size
            prediction_wrt_box[j, k][2] = prediction_wrt_box[j, k][2] * box_half_size

            prediction_wrt_frame[j, k][0] = prediction_wrt_box[j, k][0] + box_scanning_step * j
            prediction_wrt_frame[j, k][1] = prediction_wrt_box[j, k][1] + box_scanning_step * k
            prediction_wrt_frame[j, k][2] = prediction_wrt_box[j, k][2]

    return (prediction_wrt_box, prediction_wrt_frame, boxes)

def track_video(
    video_file_name,
    network,
    number_frames_to_be_tracked=1,
    box_half_size=25,
    box_scanning_step=5,
    frame_normalize=0,
    frame_enhance=1):
    """Track multiple particles in a video.
    
    Inputs:    
    video_file_name: video file
    network: the pre-trained network
    number_frames_to_be_tracked: number of frames to by analyzed from video begining. If number_frames is equal to 0 then the whole video is tracked.
    box_half_size: half the size of the scanning box. If box_half_size is equal to 0 then a single particle is tracked in a frame.
    box_scanning_step: the size of the scanning step
    frame_normalize: option to normalize the frame before tracking.
    frame_enhance: option to enhance the frame before tracking.
    
    Output:
    frames: frames from video
    predicted_positions_wrt_frame: x, y and r coordinates with respect to the all frames (pixels) 
    predicted_positions_wrt_box: x, y and r coordinates with respect to the boxes for all frames (pixels) 
    boxes_all: the part of the frame corresponding to each box for all frames
    number_frames_to_be_tracked: the number of frames that have been tracked. 
    """
    
    import cv2
    import numpy as np

    # Read the video file and its properties
    video = cv2.VideoCapture(video_file_name)

    if number_frames_to_be_tracked == 0:
        number_frames_to_be_tracked = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Initialize variables
    frames = np.zeros((number_frames_to_be_tracked, video_height, video_width))

   
    box_center_x = np.arange(box_half_size, 
                             video_height - box_half_size, 
                             box_scanning_step)
    box_center_y = np.arange(box_half_size, 
                             video_width - box_half_size, 
                             box_scanning_step)


    boxes_all = np.zeros((number_frames_to_be_tracked, 
                          len(box_center_x),
                          len(box_center_y), 
                          box_half_size * 2 + 1,
                          box_half_size * 2 + 1))

    predicted_positions_wrt_box = np.zeros((number_frames_to_be_tracked,
                                           len(box_center_x), 
                                           len(box_center_y), 
                                           3))
    predicted_positions_wrt_frame = np.zeros((number_frames_to_be_tracked,
                                             len(box_center_x), 
                                             len(box_center_y), 
                                             3))

    # Track the positions of the particles frame by frame
    for i in range(number_frames_to_be_tracked):

        # Read the current frame from the video
        (ret, frame) = video.read()
        
        # Normalize the frame
        if frame_normalize == 1:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        # Convert color image to grayscale.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
        
        frame = frame * frame_enhance

        # Generate the scanning boxes and predict particle position in each box

        (prediction_wrt_box, prediction_wrt_frame, boxes) = track_frame(network, frame, box_half_size, box_scanning_step)


        frames[i] = frame
            
        predicted_positions_wrt_box[i] = prediction_wrt_box
        predicted_positions_wrt_frame[i] = prediction_wrt_frame
        boxes_all[i] = boxes

    # Release the video
    video.release()
    

    return (number_frames_to_be_tracked, frames, predicted_positions_wrt_frame, predicted_positions_wrt_box, boxes_all)

def plot_tracked_scanning_boxes(
    frame_to_be_shown,
    rows_to_be_shown,
    columns_to_be_shown,
    boxes_all,
    predicted_positions_wrt_box,
    box_half_size=25,
    ): 
    """Plot tracked scanning boxes over a range of frames.
    
    Inputs:    
    frame_to_be_shown: the range of frames to be shown
    rows_to_be_shown: the range of rows to be shown 
    columns_to_be_shown: the range of columns to be shown 
    boxes_all: the part of the frame corresponding to each box for all frames
    predicted_positions_wrt_box: x, y and r coordinates with respect to boxes for all frames (pixels)
    box_half_size: half the size of the scanning box
    
    Output: none
    """

    import matplotlib.pyplot as plt
       
    plt.figure(10)

    for i in list(frame_to_be_shown):
        for j in list(rows_to_be_shown):
            for k in list(columns_to_be_shown):

                plt.subplot(1, 2, 1)
                plt.imshow(boxes_all[i, j, k],
                           cmap='gray', 
                           vmin=0, 
                           vmax=1)
                
                plt.plot(predicted_positions_wrt_box[i, j, k, 1],
                         predicted_positions_wrt_box[i, j, k, 0], 
                         'ob')
                plt.xlabel('y (px)', fontsize=16)
                plt.ylabel('x (px)', fontsize=16)
                
                
                plt.subplot(1, 2, 2)

                plt.text(0, .8, 'frame = %1.0f' % i, fontsize=16)
                plt.text(0, .7, 'row = %1.0f' % j, fontsize=16)
                plt.text(0, .6, 'column = %1.0f' % k, fontsize=16)
                
                plt.text(0, .4, 'particle center x = %5.2f px' % predicted_positions_wrt_box[i, j, k, 0], 
                         fontsize=16, color='b')
                plt.text(0, .3, 'particle center y = %5.2f px' % predicted_positions_wrt_box[i, j, k, 1], 
                         fontsize=16, color='b')
                plt.text(0, .2, 'particle radius = %5.2f px' % predicted_positions_wrt_box[i, j, k, 2], 
                         fontsize=16, color='b')
                plt.axis('off')

                
                plt.show()

def centroids(
    particle_positions_x,
    particle_positions_y,
    particle_radial_distance,
    particle_interdistance,
    ):
    """Calculate centroid of the particles by taking the mean x and y positions.
    
    Inputs:    
    x_particle_positions: the predicted x-positions for the particles (many for each particle)
    y_particle_positions: the predicted y-positions for the particles (many for each particle)
    particle_radial_distance: the radial distance of the particle from the center of the scanning box
    particle_max_interdistance: the maximum distance between predicted points for them to belong to the same particle
    
    Output: 
    x_centroid: the x coordinate of the centroid for each particle 
    y_centroid: the y coordinate of the centroid for each particle
    """

    import numpy as np

    particle_number = 0
    particle_index = 0
    particle_numbers = -np.ones(len(particle_positions_x))
    
    # Sort all predicted points to correct particle 
    while particle_numbers[np.argmin(particle_numbers)] == -1:
        particle_index = np.argmin(particle_numbers)
        particle_numbers[particle_index] = particle_number
        particle_number += 1

        for j in range(len(particle_positions_x)):

            if (particle_positions_x[j] - particle_positions_x[particle_index]) ** 2 \
                + (particle_positions_y[j] - particle_positions_y[particle_index]) ** 2 \
                < particle_interdistance ** 2:
                particle_numbers[j] = particle_numbers[particle_index]

    centroid_x = np.zeros(int(np.amax(particle_numbers)) + 1)
    centroid_y = np.zeros(int(np.amax(particle_numbers)) + 1)

    particle_number = 0
    while max(particle_numbers) >= particle_number:
        points_x = particle_positions_x[np.where(particle_numbers == particle_number)]
        points_y = particle_positions_y[np.where(particle_numbers == particle_number)]
        distance_from_center = particle_radial_distance[np.where(particle_numbers == particle_number)]

        # Calculate centroids
        _len = len(points_x)
        centroid_x[particle_number] = sum(points_x) / _len
        centroid_y[particle_number] = sum(points_y) / _len

        particle_number += 1

    return (centroid_x, centroid_y)

def show_tracked_frames(
    particle_radial_distance_threshold,
    particle_maximum_interdistance,
    number_frames_to_be_shown,
    frames,
    predicted_positions_wrt_frame,
    ):
    """Show the frames with the predicted positions and centroid positions.
    
    Inputs:    
    particle_radial_distance: the radial distance of the particle from the center of the scanning box
    particle_maximuminterdistance: the maximum distance between predicted points for them to belong to the same particle
    number_frames_to_be_shown: number of frames to be shown
    frames: frames from video
    predicted_positions_wrt_frame: x, y and r coordinates with respect to frames (pixels) 
    
    Output: none
    """
    
    import numpy as np
    import matplotlib.pyplot as plt

    particle_positions = []
    particle_centroids = []
    particle_radial_distance = []

    for i in range(number_frames_to_be_shown):

        particle_positions_x = []
        particle_positions_y = []

        # Show frame
        plt.figure(figsize=(10, 10))
        plt.imshow(frames[i], cmap='gray', vmin=0, vmax=1)

        # Threshold the radial distance of the predicted points
        for j in range(0, predicted_positions_wrt_frame.shape[1]):
            for k in range(0, predicted_positions_wrt_frame.shape[2]):

                if predicted_positions_wrt_frame[i, j, k, 2] \
                    < particle_radial_distance_threshold:

                    # Plot the predicted points
                    plt.plot(predicted_positions_wrt_frame[i, j, k, 1],
                             predicted_positions_wrt_frame[i, j, k, 0], '.b')
                    particle_positions_x = \
                        np.append(particle_positions_x,
                                  predicted_positions_wrt_frame[i, j, k, 0])
                    particle_positions_y = \
                        np.append(particle_positions_y,
                                  predicted_positions_wrt_frame[i, j, k, 1])
                    particle_radial_distance = \
                        np.append(particle_radial_distance,
                                  predicted_positions_wrt_frame[i, j, k, 2])

        particle_positions.append([])
        particle_positions[i].append(particle_positions_x)
        particle_positions[i].append(particle_positions_y)

        # Calculate the centroid positions
        (centroids_x, centroids_y) = centroids(particle_positions_x,
                                               particle_positions_y, 
                                               particle_radial_distance,
                                               particle_maximum_interdistance)

        particle_centroids.append([])
        particle_centroids[i].append(centroids_x)
        particle_centroids[i].append(centroids_y)

        # Plot the centroid positions
        plt.plot(
            centroids_y,
            centroids_x,
            'o',
            color='#e6661a',
            fillstyle='none',
            markersize=10,
            )
        
    return #(particle_positions, particle_centroids)

def track_video_single_particle(
    video_file_name,
    network,
    number_frames_to_be_tracked=1,
    frame_normalize=0,
    frame_enhance=1):
    """Track single particlee in a video.
    
    Inputs:    
    video_file_name: video file
    network: the pre-trained network
    number_frames_to_be_tracked: number of frames to by analyzed from video begining. If number_frames is equal to 0 then the whole video is tracked.
        
    Output:
    frames: frames from video
    predicted_positions: x, y and r coordinates with respect to the all frames (pixels) 
    """
    
    import cv2
    import numpy as np

    # Read the video file and its properties
    video = cv2.VideoCapture(video_file_name)

    if number_frames_to_be_tracked == 0:
        number_frames_to_be_tracked = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize variables
    frames = np.zeros((number_frames_to_be_tracked, video_height, video_width))

    predicted_positions = np.zeros((number_frames_to_be_tracked, 3))

    # Track the positions of the particles frame by frame
    for i in range(number_frames_to_be_tracked):

        # Read the current frame from the video
        (ret, frame) = video.read()
        
        # Normalize the frame
        if frame_normalize == 1:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        # Convert color image to grayscale.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
        
        # Enhance the frame
        frame = frame * frame_enhance
        
        ### Resize the frame
        frame_resize = cv2.resize(frame, (51, 51))

        predicted_position = network.predict(np.reshape(frame_resize, (1, 51, 51, 1)))

        predicted_position_x = predicted_position[0,0] *  video_width / 2 + video_width / 2
        predicted_position_y = predicted_position[0,1] *  video_height / 2 + video_height / 2
        predicted_position_r = predicted_position[0,2] * (video_height**2 + video_width**2)**0.5 / 2 
 
        
        predicted_positions[i, 0] = predicted_position_x
        predicted_positions[i, 1] = predicted_position_y
        predicted_positions[i, 2] = predicted_position_r
    
       
        if i < 10:
            frames[i] = frame
        
        
    # Release the video
    video.release()

    return (number_frames_to_be_tracked, frames, predicted_positions)

def show_tracked_frames_single_particle(
    number_frames_to_be_shown,
    frames,
    predicted_positions,
    ):
    """Show the frames with the predicted position.
    
    Inputs:    
    number_frames_to_be_shown: number of frames to be shown
    frames: frames from video
    predicted_positions: x, y and r coordinates (pixels) 
    
    Output: none
    """
    
    import numpy as np
    import matplotlib.pyplot as plt


    for i in range(number_frames_to_be_shown):

        # Show frame
        plt.figure(figsize=(10, 10))
        plt.imshow(frames[i], cmap='gray', vmin=0, vmax=1)


        # Plot the predicted points
        plt.plot(predicted_positions[i, 1],
                 predicted_positions[i, 0], 
                 '.',
                 color='#e6661a',
                 #fillstyle='none',
                 markersize=20,
                 )

def particle_positions(
    particle_number=2,
    first_particle_range=0.5,
    other_particle_range=1,
    particle_distance=50,
    ):
    """Generates multiple particle x- and y-coordinates with respect to each other.
    
    Inputs:  
    particle_number: number of particles to generate coordinates for
    first_particle_range: allowed x- and y-range of the centermost particle
    other_particle_range: allowed x- and y-range for all other particles
    particle_distance: particle interdistance
    
    Output:
    particles_center_x: list of x-coordinates for the particles
    particles_center_y: list of y-coordinates for the particles
    """

    from numpy.random import normal
    from numpy.random import uniform
    from numpy import insert
    from numpy import array
    from itertools import combinations
    from numpy import linalg
    
    ### Centermost particle
    target_particle_center_x = uniform(-first_particle_range,
            first_particle_range)
    target_particle_center_y = uniform(-first_particle_range,
            first_particle_range)

    target_center_distance = (target_particle_center_x ** 2
                              + target_particle_center_y ** 2) ** 0.5

    ### Other particles
    while True:
        particles_center_x = uniform(-other_particle_range,
                other_particle_range, particle_number - 1)  
        particles_center_y = uniform(-other_particle_range,
                other_particle_range, particle_number - 1) 

        center_distance = (particles_center_x ** 2 + particles_center_y
                           ** 2) ** 0.5

        ### Force all other particles to be further away from the center than the centermost particle
        if any(t < target_center_distance for t in center_distance):
            continue

        particles_center_x = insert(particles_center_x, 0,
                                    target_particle_center_x)
        particles_center_y = insert(particles_center_y, 0,
                                    target_particle_center_y)

        particle_centers = array([particles_center_x,
                                 particles_center_y])

        ### Force all particles to be a certain distance from each other
        if all(linalg.norm(p - q) > particle_distance for (p, q) in
               combinations(particle_centers, 2)):
            break

    return (particles_center_x, particles_center_y)

def load(saved_network_file_name):
    """Load a pretrained model.
    
    Input:
    saved_network_file_name: name of the file to be loaded
    
    Output:
    network: pretrained network
    """
    
    from keras.models import load_model

    network = load_model(saved_network_file_name)
    
    return network


def credits():
    """Credits for DeepTrack 1.0.
    
    DeepTrack 1.0
    Digital Video Microscopy enhanced with Deep Learning
    version 1.0 - 15 November 2018
    © Saga Helgadottir, Aykut Argun & Giovanni Volpe
    http://www.softmatterlab.org
    """
    
    print(
        '\n' +
        '\033[1mDeepTrack 1.0\033[0m\n' +
        'Digital Video Microscopy enhanced with Deep Learning\n' +
        'version 1.0 - November 2018\n' +
        'Saga Helgadottir, Aykut Argun & Giovanni Volpe\n' +
        'http://www.softmatterlab.org\n')
    
    return