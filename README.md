<p align="center">
  <img width="350" src=https://github.com/softmatterlab/DeepTrack2/blob/develop/assets/logo.png?raw=true>
</p>

<h3 align="center">A comprehensive framework for digital microscopy.</h3>
<p align="center">
  <a href="/LICENSE" alt="licence"><img src="https://img.shields.io/github/license/softmatterlab/DeepTrack-2.0"></a>
  <a href="https://badge.fury.io/py/deeptrack"><img src="https://badge.fury.io/py/deeptrack.svg" alt="PyPI version"></a>
  <a href="https://deeptrackai.github.io/DeepTrack2"><img src="https://img.shields.io/badge/docs-passing-green" alt="PyPI version"></a>
  <a href="https://badge.fury.io/py/deeptrack"><img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python version"></a>
  <a href="https://doi.org/10.1063/5.0034891" alt="DOI">
    <img src="https://img.shields.io/badge/DOI-10.1063%2F5.0034891-blue">
  </a>
</p>
<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#examples-of-applications-using-deeptrack"> Examples</a> •
  <a href="#getting-started-guides">Basics</a> •
  <a href="#cite-us">Cite us</a> •
  <a href="/LICENSE">License</a> 
</p>

DeepTrack2 is a modular Python library for generating, manipulating, and analyzing image data pipelines for machine learning and experimental imaging.

<b>TensorFlow Compatibility Notice:</b> 
DeepTrack2 version 2.0 and subsequent do not support TensorFlow. If you need TensorFlow support, please install the legacy version 1.7.

# Quick Start Guide

The following quick start guide is intended for complete beginners to understand how to use DeepTrack2, from installation to training your first model. Let's get started!

## Installation

DeepTrack2 2.0 requires at least python 3.9.

To install DeepTrack2, open a terminal or command prompt and run:
```bash
pip install deeptrack
```

## Getting-started guides

We have a set of four notebooks which aims to teach you all you need to know to use DeepTrack to its fullest with a focus on the application.

1. <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/examples/get-started/01.%20deeptrack_introduction_tutorial.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> deeptrack_introduction_tutorial </a>  Gives an overview of how to use DeepTrack 2.1.
2. <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/examples/tutorials/01.%20tracking_particle_cnn_tutorial.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> tracking_particle_cnn_tutorial </a> Demonstrates how to track a point particle with a convolutional neural network (CNN).
3. <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/examples/tutorial/02.%20tracking_multiple_particles_unet_tutorial.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> tracking_multiple_particles_unet_tutorial </a> Demonstrates how to track multiple particles using a U-net.
4. <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/examples/tutorials/03.%20distinguishing_particles_in_brightfield_tutorial.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg"> distinguishing_particles_in_brightfield_tutorial </a> Demonstrates how to use a U-net to track and distinguish particles of different sizes in brightfield microscopy. 




## DeepTrack 2.1 in action

Additionally, we have six more case studies which are less documented, but gives additional insight in how to use DeepTrack with real datasets

1. [Single Particle Tracking](examples/paper-examples/2-single_particle_tracking.ipynb) Tracks experimental videos of a single particle. (Requires opencv-python compiled with ffmpeg)
2. [Multi-Particle tracking](examples/paper-examples/4-multi-molecule-tracking.ipynb) Detect quantum dots in a low SNR image.
3. [Particle Feature Extraction](examples/paper-examples/3-particle_sizing.ipynb) Extract the radius and refractive index of particles.
4. [Cell Counting](examples/paper-examples/6-cell_counting.ipynb) Count the number of cells in fluorescence images.
5. [3D Multi-Particle tracking](examples/paper-examples/5-inline_holography_3d_tracking.ipynb)
6. [GAN image generation](examples/paper-examples/7-GAN_image_generation.ipynb) Use a GAN to create cell image from masks.

## Model-specific examples

We also have examples that are specific for certain models. This includes 
- [*LodeSTAR*](examples/LodeSTAR) for label-free particle tracking.
- [*MAGIK*](examples/MAGIK) for graph-based particle linking and trace characterization.

## Cite us!
If you use DeepTrack 2.1 in your project, please cite us here:

```
Benjamin Midtvedt, Saga Helgadottir, Aykut Argun, Jesús Pineda, Daniel Midtvedt, Giovanni Volpe.
"Quantitative Digital Microscopy with Deep Learning."
Applied Physics Reviews 8 (2021), 011310.
https://doi.org/10.1063/5.0034891
```

See also:

<https://www.nature.com/articles/s41467-022-35004-y>:
```
Midtvedt, B., Pineda, J., Skärberg, F. et al. 
"Single-shot self-supervised object detection in microscopy." 
Nat Commun 13, 7492 (2022).
```

<https://arxiv.org/abs/2202.06355>:
```
Jesús Pineda, Benjamin Midtvedt, Harshith Bachimanchi, Sergio Noé, Daniel  Midtvedt, Giovanni Volpe,1 and  Carlo  Manzo
"Geometric deep learning reveals the spatiotemporal fingerprint ofmicroscopic motion."
arXiv 2202.06355 (2022).
```

<https://doi.org/10.1364/OPTICA.6.000506>:
```
Saga Helgadottir, Aykut Argun, and Giovanni Volpe.
"Digital video microscopy enhanced by deep learning."
Optica 6.4 (2019): 506-513.
```

<https://github.com/softmatterlab/DeepTrack.git>:
```
Saga Helgadottir, Aykut Argun, and Giovanni Volpe.
"DeepTrack." (2019)
```

## Getting Started

Here you find a series of notebooks that give you an overview of the core features of DeepTrack2 and how to use them:

- GS101 **[Introduction to DeepTrack2](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/getting-started/GS101_intro.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/getting-started/GS101_intro.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Creating images combining DeepTrack2 features, extracting properties, and using them to train a neural network.

- GS111 **[Loading Image Files Using Sources](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/getting-started/GS111_datafiles.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/getting-started/GS111_datafiles.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Using sources to load image files and to train a neural network.

## Examples


## Developer Tutorials

Here you find a series of notebooks tailored for DeepTrack2's developers:

- DT111 **[Style Guide](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/developers/DT111_style.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/developers/DT111_style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

## Documentation

The detailed documentation of DeepTrack2 is available at the following link: [https://deeptrackai.github.io/DeepTrack2](https://deeptrackai.github.io/DeepTrack2)

## Funding

This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511), the ERC Starting Grant MAPEI (101001267), and the Knut and Alice Wallenberg Foundation.
