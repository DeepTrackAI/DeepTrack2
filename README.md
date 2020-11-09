<p align="center">
  <img width="350" src=https://github.com/softmatterlab/DeepTrack-2.0/blob/master/assets/logo.png?raw=true>
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
6. [analyzing_video_tutorial](tutorials/analyzing_video_tutorial.ipynb) demonstrates how to create videos and how to train a neural network to analyze them.

Additionally, we have seven more case studies which are less documented, but gives additional insight in how to use DeepTrack with real datasets

1. [![](https://colab.research.google.com/assets/colab-badge.svg) MNIST](https://colab.research.google.com/drive/1dRehGzf9DNpz7Jo2dw4U6vSyE4STZgpF?usp=sharing) classifies handwritted digits.
2. [![](https://colab.research.google.com/assets/colab-badge.svg) single particle tracking](https://colab.research.google.com/drive/1rh46w8TuJDF0mnvLpo6dlWkLiLr7MmQ9?usp=sharing) tracks experimentally captured videos of a single particle. (Requires opencv-python compiled with ffmpeg to open and read a video.)
3. [![](https://colab.research.google.com/assets/colab-badge.svg) single particle sizing](https://colab.research.google.com/drive/1U12f3m3oLKCGp-BAERGrwjMhEZdkWvT5?usp=sharing) extracts the radius and refractive index of particles.
4. [![](https://colab.research.google.com/assets/colab-badge.svg) multi-particle tracking](https://colab.research.google.com/drive/1TpNZ6ytoDXSZvGDFAFWrSjNs4SXGZmBw?usp=sharing) detects quantum dots in a low SNR image.
5. [![](https://colab.research.google.com/assets/colab-badge.svg) 3-dimensional tracking](https://colab.research.google.com/drive/1QJXPxsVeDt1ZW1685D5VANsME69s3mqi?usp=sharing) tracks particles in three dimensions.
6. [![](https://colab.research.google.com/assets/colab-badge.svg) cell counting](https://colab.research.google.com/drive/1C2Gn1Ym8etycOYW9yfDB_WiKlEeyvLtp?usp=sharing) counts the number of cells in fluorescence images.
7. [![](https://colab.research.google.com/assets/colab-badge.svg) GAN image generation](https://colab.research.google.com/drive/1rfFbeE-qkg3PxHBEa_r7Q9wXq0vdueEC?usp=sharing) uses a GAN to create cell image from masks.

### Video Tutorials

DeepTrack 2.0 introduction tutorial video: https://youtu.be/hyfaxF8q6VE  
<a href="http://www.youtube.com/watch?feature=player_embedded&v=hyfaxF8q6VE
" target="_blank"><img src="https://img.youtube.com/vi/hyfaxF8q6VE/maxresdefault.jpg" 
alt="Tutorial" width="384" height="216" border="10" /></a>

DeepTrack 2.0 single particle tracking tutorial video: https://youtu.be/6Cntik6AfBI   
<a href="http://www.youtube.com/watch?feature=player_embedded&v=6Cntik6AfBI
" target="_blank"><img src="https://img.youtube.com/vi/6Cntik6AfBI/maxresdefault.jpg" 
alt="Tutorial" width="384" height="216" border="10" /></a>


DeepTrack 2.0 multiple particle tracking tutorial video: https://youtu.be/wFV2VqzpeZs  
<a href="http://www.youtube.com/watch?feature=player_embedded&v=wFV2VqzpeZs
" target="_blank"><img src="https://img.youtube.com/vi/wFV2VqzpeZs/maxresdefault.jpg" 
alt="Tutorial" width="384" height="216" border="10" /></a>




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

## Graphical user interface

DeepTrack 2.0 provides a completely stand-alone  [graphical  user  interface](https://github.com/softmatterlab/DeepTrack-2.0-app),  which  delivers  all  the power of DeepTrack without requiring programming knowledge.

[![InterfaceDemo](https://i.imgur.com/lTy2vhz.gif)](https://i.imgur.com/lTy2vhz.gif)

## Documentation

The detailed documentation of DeepTrack 2.0 is available at the following link: https://softmatterlab.github.io/DeepTrack-2.0/deeptrack.html

## Cite us!

If you use DeepTrack 2.0 in your project, please cite us here:

    Benjamin Midtvedt, Saga Helgadottir, Aykut Argun, Jesús Pineda, Daniel Midtvedt, Giovanni Volpe. "Quantitative Digital Microscopy with Deep Learning." [arXiv:2010.08260](https://arxiv.org/abs/2010.08260)

See also:

    Saga Helgadottir, Aykut Argun, and Giovanni Volpe. "Digital video microscopy enhanced by deep learning." Optica 6.4 (2019): 506-513. [10.1364/OPTICA.6.000506](https://doi.org/10.1364/OPTICA.6.000506)

    Saga Helgadottir, Aykut Argun, and Giovanni Volpe. "DeepTrack." https://github.com/softmatterlab/DeepTrack.git (2019).

## Funding
This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511).
