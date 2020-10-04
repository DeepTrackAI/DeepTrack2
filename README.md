<p align="center">
  <img width="450" src=https://github.com/softmatterlab/DeepTrack-2.0/blob/master/assets/logo.png?raw=true>
</p>


DeepTrack is a comprehensive deep learning framework for digital microscopy. 
We provide tools to create physical simulations of customizable optical systems, to generate and train neural network models, and to analyze experimental data.

# Getting started

## Installation

DeepTrack 2.0 requires at least python 3.6

To install DeepTrack 2.0, open a terminal or command prompt and run

    pip install deeptrack
    
## Learning DeepTrack 2.0

Everybody learns in different ways! Depending on your preferences, and what you want to do with DeepTrack, you may want to check out one or more of these resources.

### Fundamentals

First, we have a very general walkthrough of [basic](https://softmatterlab.github.io/DeepTrack-2.0/basics.html) and [advanced](https://softmatterlab.github.io/DeepTrack-2.0/advanced.html) topics. This is a 5-10 minute read, that well get you well on your way to understand the unique interactions available in DeepTrack.

### DeepTrack 2.0 in action

To see DeepTrack in action, we provide six well documented tutorial notebooks that create simulation pipelines and train models:

1. [deeptrack_introduction_tutorial](tutorials/deeptrack_introduction_tutorial.ipynb) gives an overview of how to use DeepTrack 2.0.
2. [tracking_particle_cnn_tutorial](tutorials/tracking_particle_cnn_tutorial.ipynb) demonstrates how to track a point particle with a convolutional neural network (CNN). 
3. [tracking_multiple_particles_unet_tutorial](tutorials/tracking_multiple_particles_unet_tutorial.ipynb) demonstrates how to track multiple particles using a U-net.
4. [characterizing_aberrations_tutorial](tutorials/characterizing_aberrations_tutorial.ipynb) demonstrates how to add and characterize aberrations of an optical device.
5. [distinguishing_particles_in_brightfield_tutorial](tutorials/distinguishing_particles_in_brightfield_tutorial.ipynb) demonstrates how to use a U-net to track and distinguish particles of different sizes in brightfield microscopy.
6. [analyzing_video_tutorial](tutorials/tracking_video_tutorial.ipynb) demonstrates how to create videos and how to train a neural network to analyze them.

Additionally, we have seven more case studies which are less documented, but gives additional insight in how to use DeepTrack with real datasets

1. [MNIST](paper-examples/1-MNIST.ipynb) classifies handwritted digits.
2. [single particle tracking](paper-examples/2-single_particle_tracking.ipynb) tracks experimentally captured videos of a single particle. 
3. [single particle sizing](paper-examples/3-particle_sizing.ipynb) extracts the radius and refractive index of particles.
4. [multi-particle tracking](paper-examples/4-multi-mulecule-tracking.ipynb) detects quantum dots in a low SNR image.
5. [3-dimensional tracking](paper-examples/5-inline_holography_3d_tracking.ipynb) tracks particles in three dimensions.
6. [cell counting](paper-examples/6-cell_counting.ipynb) counts the number of cells in fluorescence images.
7. [GAN image generation](paper-examples/7-GAN_image_generation.ipynb) uses a GAN to create cell image from masks.

### Video Tutorials

[TBA]

### In-depth dives

The examples folder contains notebooks which explains the different modules in more detail. These can be read in any order, but we provide a recommended order where more fundamental topics are introduced early.
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

## Documentation

The detailed documentation of DeepTrack 2.0 is available at the following link: https://softmatterlab.github.io/DeepTrack-2.0/deeptrack.html

## Funding
This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511).
