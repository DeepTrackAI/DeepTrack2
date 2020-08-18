# DeepTrack 2.0

DeepTrack is a comprehensive deep learning framework for digital microscopy. 
We provide tools to create physical simulations of customizable optical systems, to generate and train neural network models, and to analyze experimental data.

## Getting started

### Installation

To install, clone the folder 'deeptrack' to your project directory. The ability to install the package through pip is comming soon.

Dependencies:
- tensorflow (>=2.2.0)

Optional dependencies:
- matplotlib
- ffmpeg

### Tutorials

The folder 'tutorials' contains notebooks with common applications. 
These may serve as a useful starting point from which to build a solution. 
The notebooks can be read in any order, but we provide a suggested order to introduce new concepts more naturally. 
This order is as follows:

1. [deeptrack_introduction_tutorial](tutorials/deeptrack_introduction_tutorial.ipynb) gives an overview of how to use DeepTrack 2.0.
2. [tracking_particle_cnn_tutorial](tutorials/tracking_particle_cnn_tutorial.ipynb) demonstrates how to track a point particle with a convolutional neural network (CNN). 
3. [tracking_multiple_particles_unet_tutorial](tutorials/tracking_multiple_particles_unet_tutorial.ipynb) demonstrates how to track multiple particles using a U-net.
4. [characterizing_aberrations_tutorial](tutorials/characterizing_aberrations_tutorial.ipynb) demonstrates how to add and characterize aberrations of an optical device.
5. [distinguishing_particles_in_brightfield_tutorial](tutorials/distinguishing_particles_in_brightfield_tutorial.ipynb) demonstrates how to use a U-net to track and distinguish particles of different sizes in brightfield microscopy.
6. [analyzing_video_tutorial](tutorials/tracking_video_tutorial.ipynb) demonstrates how to create videos and how to train a neural network to analyze them.

### Examples

The examples folder contains notebooks which explains the different modules in more detail. Also these can be read in any order, but we provide a recommended order where more fundamental topics are introduced early.
This order is as follows:

1. [features_example](examples/features_example.ipynb)
2. [properties_example](examples/properties_example.ipynb)
3. [scatterers_example](examples/scatterers_example.ipynb)
4. [optics_example](examples/optics_example.ipynb)
5. [aberrations_example](examples/aberrations_example.ipynb)
6. [noises_example](examples/noises_example.ipynb)
7. [augmentations_example](examples/augmentations_example.ipynb)
6. [image_example](examples/image_example.ipynb)
7. [generators_example](examples/generators_example.ipynb)
8. [models_example](examples/models_example.ipynb)
10. [losses_example](examples/losses_example.ipynb)
11. [utils_example](examples/utils_example.ipynb)
12. [sequences_example](examples/sequences_example.ipynb)
13. [math_example](examples/math_example.ipynb)

### Documentation

The detailed documentation of DeepTrack 2.0 is available at the following link: https://deeptrack-20.readthedocs.io/en/latest/

## Funding
This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511).
