# DeepTrack 2.0

DeepTrack is comprehensive deep learning framework for digital microscopy. 
We provide tools to create physical simulations of customizable optical systems, to generate and train models, and to analyze experimental data.

## Getting started

### Installation

To install, clone the folder 'deeptrack' to your project directory. The ability to install the package through pip is comming soon.

Dependencies:
- tensorflow (>=1.14)

Optional dependencies:
- matplotlib
- ffmpeg

### Tutorials

The folder 'tutorials' contains notebooks with common applications. 
These may serve as a useful starting point from which to build a solution. 
The notebooks can be read in any order, but we provide a suggested order to introduce new concepts more naturally. 
This order is as follows:

1. deeptrack_introduction_tutorial
2. tracking_particles_cnn_tutorial
3. tracking_multiple_particles_unet_tutorial
4. characterizing_aberrations_tutorial
5. distinguishing_particles_in_brightfield_tutorial
6. tracking_video_tutorial

### Examples

The examples folder contains notebooks which explains the different modules in more detail. Also these can be read in any order, but we provide a recommended order where more fundamental topics are introduced early.
This order is as follows:

1. features_example
2. properties_example
3. scatterers_example
4. optics_example
5. aberrations_example
6. noises_example
7. augmentations_example
6. image_example
7. generators_example
8. models_example
