<p align="center">
  <img width="350" src=https://github.com/softmatterlab/DeepTrack-2.0/blob/master/assets/logo.png?raw=true>
</p>

DeepTrack is a comprehensive deep learning framework for digital microscopy.
We provide tools to create physical simulations of customizable optical systems, to generate and train neural network models, and to analyze experimental data.

If you use DeepTrack 2.0 in your project, please cite our DeepTrack 2.0 article:

```
Benjamin Midtvedt, Saga Helgadottir, Aykut Argun, Jesús Pineda, Daniel Midtvedt, Giovanni Volpe.
"Quantitative Digital Microscopy with Deep Learning."
Applied Physics Reviews 8 (2021), 011310.
https://doi.org/10.1063/5.0034891
```

# Getting started

## Installation

DeepTrack 2.1 requires at least python 3.6

To install DeepTrack 2.1, open a terminal or command prompt and run

    pip install deeptrack
    
If you have a very recent version of python, you may need to install numpy _before_ DeepTrack. This is a known issue with scikit-image.

## Learning DeepTrack 2.0

Everybody learns in different ways! Depending on your preferences, and what you want to do with DeepTrack, you may want to check out one or more of these resources.

### Fundamentals

First, we have a very general walkthrough of [basic](https://softmatterlab.github.io/DeepTrack-2.0/basics.html) and [advanced](https://softmatterlab.github.io/DeepTrack-2.0/advanced.html) topics. This is a 5-10 minute read, that will get you well on your way to understand the unique interactions available in DeepTrack.

Similarly, you may find the [get-started notebooks](examples/get-started) a rewarding way to start learning DeepTrack

## Documentation

The detailed documentation of DeepTrack 2.0 is available at the following link: https://softmatterlab.github.io/DeepTrack-2.0/deeptrack.html

### DeepTrack 2.0 in action

To see DeepTrack in action, we provide six well documented tutorial notebooks that create simulation pipelines and train models:

1. [deeptrack_introduction_tutorial](examples/tutorials/deeptrack_introduction_tutorial.ipynb) gives an overview of how to use DeepTrack 2.0.
2. [tracking_particle_cnn_tutorial](examples/tutorials/tracking_particle_cnn_tutorial.ipynb) demonstrates how to track a point particle with a convolutional neural network (CNN).
3. [tracking_multiple_particles_unet_tutorial](examples/tutorials/tracking_multiple_particles_unet_tutorial.ipynb) demonstrates how to track multiple particles using a U-net.
4. [characterizing_aberrations_tutorial](examples/tutorials/characterizing_aberrations_tutorial.ipynb) demonstrates how to add and characterize aberrations of an optical device.
5. [distinguishing_particles_in_brightfield_tutorial](examples/tutorials/distinguishing_particles_in_brightfield_tutorial.ipynb) demonstrates how to use a U-net to track and distinguish particles of different sizes in brightfield microscopy.
6. [analyzing_video_tutorial](examples/tutorials/analyzing_video_tutorial.ipynb) demonstrates how to create videos and how to train a neural network to analyze them.

Additionally, we have seven more case studies which are less documented, but gives additional insight in how to use DeepTrack with real datasets

1. [MNIST](examples/paper-examples/1_MNIST.ipynb) classifies handwritted digits.
2. [single particle tracking](examples/paper-examples/2-single_particle_tracking.ipynb) tracks experimentally captured videos of a single particle. (Requires opencv-python compiled with ffmpeg to open and read a video.)
3. [single particle sizing](examples/paper-examples/3-particle_sizing.ipynb) extracts the radius and refractive index of particles.
4. [multi-particle tracking](examples/paper-examples/4-multi-molecule-tracking.ipynb) detects quantum dots in a low SNR image.
5. [3-dimensional tracking](examples/paper-examples/5-inline_holography_3d_tracking.ipynb) tracks particles in three dimensions.
6. [cell counting](examples/paper-examples/6-cell_counting.ipynb) counts the number of cells in fluorescence images.
7. [GAN image generation](examples/paper-examples/7-GAN_image_generation.ipynb) uses a GAN to create cell image from masks.

### Model-specific examples

We also have examples that are specific for certain models. This includes 
- [*LodeSTAR*](examples/LodeSTAR) for label-free particle tracking.
- [*MAGIK*](examples/MAGIK) for graph-based particle linking a trace characterization.

### Video Tutorials

Videos are currently being updated to match with the current version of DeepTrack.

### In-depth dives

The examples folder contains notebooks which explains the different modules in more detail. These can be read in any order, but we provide a recommended order where more fundamental topics are introduced early.
This order is as follows:

1. [features_example](examples/module-examples/features_example.ipynb)
2. [properties_example](examples/module-examples/properties_example.ipynb)
3. [scatterers_example](examples/module-examples/scatterers_example.ipynb)
4. [optics_example](examples/module-examples/optics_example.ipynb)
5. [aberrations_example](examples/module-examples/aberrations_example.ipynb)
6. [noises_example](examples/module-examples/noises_example.ipynb)
7. [augmentations_example](examples/module-examples/augmentations_example.ipynb)
8. [image_example](examples/module-examples/image_example.ipynb)
9. [generators_example](examples/module-examples/generators_example.ipynb)
10. [models_example](examples/module-examples/models_example.ipynb)
11. [losses_example](examples/module-examples/losses_example.ipynb)
12. [utils_example](examples/module-examples/utils_example.ipynb)
13. [sequences_example](examples/module-examples/sequences_example.ipynb)
14. [math_example](examples/module-examples/math_example.ipynb)



## Cite us!

If you use DeepTrack 2.0 in your project, please cite us here:

```
Benjamin Midtvedt, Saga Helgadottir, Aykut Argun, Jesús Pineda, Daniel Midtvedt, Giovanni Volpe.
"Quantitative Digital Microscopy with Deep Learning."
Applied Physics Reviews 8 (2021), 011310.
https://doi.org/10.1063/5.0034891
```

See also:

```
Jesús Pineda, Benjamin Midtvedt, Harshith Bachimanchi, Sergio Noé, Daniel  Midtvedt, Giovanni Volpe,1 and  Carlo  Manzo
"Geometric deep learning reveals the spatiotemporal fingerprint ofmicroscopic motion."
arxiv (2022).
 arXiv:2202.06355v1
```

```
Saga Helgadottir, Aykut Argun, and Giovanni Volpe.
"Digital video microscopy enhanced by deep learning."
Optica 6.4 (2019): 506-513.
https://doi.org/10.1364/OPTICA.6.000506
```

```
Saga Helgadottir, Aykut Argun, and Giovanni Volpe.
"DeepTrack." (2019)
https://github.com/softmatterlab/DeepTrack.git
```

## Funding

This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511).
