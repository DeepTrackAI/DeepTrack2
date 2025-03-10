<!-- GH_ONLY_START -->
<p align="center">
  <img width="350" src=https://github.com/DeepTrackAI/DeepTrack2/blob/develop/assets/DeepTrack2-logo.png?raw=true>
</p>
<!-- GH_ONLY_END -->

<h3 align="center">DeepTrack2 - A comprehensive deep learning framework for digital microscopy.</h3>
<p align="center">
  <a href="/LICENSE" alt="licence">
    <img src="https://img.shields.io/github/license/DeepTrackAI/DeepTrack2">
  </a>
  <a href="https://badge.fury.io/py/deeptrack">
    <img src="https://badge.fury.io/py/deeptrack.svg" alt="PyPI version">
  </a>
  <a href="https://deeptrackai.github.io/DeepTrack2">
    <img src="https://img.shields.io/badge/docs-available-blue?logo=readthedocs">
  </a>
  <a href="https://badge.fury.io/py/deeptrack">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python version">
  </a>
  <a href="https://doi.org/10.1063/5.0034891">
    <img src="https://img.shields.io/badge/cite us-10.1063%2F5.0034891-blue">
  </a>
</p>
<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#examples">Examples</a> •
  <a href="#advanced-tutorials">Advanced Tutorials</a> •
  <a href="#developer-tutorials">Developer Tutorials</a> •
  <a href="#cite-us">Cite us</a> •
  <a href="/LICENSE">License</a> 
</p>

DeepTrack2 is a modular Python library for generating, manipulating, and analyzing image data pipelines for machine learning and experimental imaging.

<b>TensorFlow Compatibility Notice:</b> 
DeepTrack2 version 2.0 and subsequent do not support TensorFlow. If you need TensorFlow support, please install the legacy version 1.7.

The following quick start guide is intended for complete beginners to understand how to use DeepTrack2, from installation to training your first model. Let's get started!

# Installation

DeepTrack2 2.0 requires at least python 3.9.

To install DeepTrack2, open a terminal or command prompt and run:
```bash
pip install deeptrack
```
or
```bash
python -m pip install deeptrack
```
This will automatically install the required dependencies.

# Getting Started

Here you find a series of notebooks that give you an overview of the core features of DeepTrack2 and how to use them:

- DTGS101 **[Introduction to DeepTrack2](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS101_intro.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS101_intro.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Overview of how to use DeepTrack 2. Creating images combining DeepTrack2 features, extracting properties, and using them to train a neural network.

- DTGS111 **[Loading Image Files Using Sources](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS111_datafiles.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS111_datafiles.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Using sources to load image files and to train a neural network.

- DTGS121 **[Tracking a Point Particle with a CNN](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS121_tracking_particle_cnn.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS121_tracking_particle_cnn.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Tracking a point particle with a convolutional neural network (CNN) using simulated images in the training process.

- DTGS131 **[Tracking Multiple Particles with a U-Net](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS131_tracking_multiple_particles_unet.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS131_tracking_multiple_particles_unet.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Tracking multiple particles using a U-net trained on simulated images.

- DTGS141 **[Distinguishing Particles with a U-Net](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS141_distinguishing_particles_in_brightfield.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS141_distinguishing_particles_in_brightfield.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Tracking and distinguishing particles of different sizes in brightfield microscopy using a U-net trained on simulated images.

- DTGS151 **[Unsupervised Object Detection](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS151_unsupervised_object_detection_with_lodestar.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS151_unsupervised_object_detection_with_lodestar.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Single-shot unsupervised object detection a using LodeSTAR.

# Examples

These are examples of how DeepTrack2 can be used on real datasets:

- DTEx211 **[MNIST](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx201_MNIST.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx201_MNIST.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Training a fully connected neural network to identify handwritten digits using MNIST dataset.

- DTEx212 **Single Particle Tracking**

  Tracks experimental videos of a single particle. (Requires opencv-python compiled with ffmpeg)

  <!-- GH_ONLY_START -->
  <p align="left">
    <img width="300" src=/assets/SPT-ideal.gif?raw=true>
    <img width="300" src=/assets/SPT-noisy.gif?raw=true>
    <br/>
    <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx202_single_particle_tracking.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://doi.org/10.1364/OPTICA.6.000506" alt="DeepTrack article">
      <img src="https://img.shields.io/badge/article-10.1364/OPTICA.6.000506-blue">
    </a> 
    <br/>
    <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231B_LodeSTAR_tracking_particles_of_various_shapes.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://doi.org/10.1038/s41467-022-35004-y" alt="LodeSTAR article">
      <img src="https://img.shields.io/badge/article-10.1038%2Fs41467--022--35004--y-blue">
    </a> 
  </p>
  <!-- GH_ONLY_END -->

- DTEx213 **Multi-Particle tracking**
- 
  Detecting quantum dots in a low SNR image.

  <!-- GH_ONLY_START -->
  <p align="left">
    <img width="600" src=/assets/MPT-packed.gif?raw=true>
    <br/>
    <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx203_particle_sizing.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://doi.org/10.1063/5.0034891" alt="LodeSTAR article">
      <img src="https://img.shields.io/badge/article-10.1063/5.0034891-blue">
    </a> 
    <br/>
    <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231A_LodeSTAR_autotracker_template.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://doi.org/10.1038/s41467-022-35004-y" alt="Article LodeSTAR">
      <img src="https://img.shields.io/badge/article-10.1038%2Fs41467--022--35004--y-blue">
    </a>
  </p>
  <!-- GH_ONLY_END -->

- DTEx214 **Particle Feature Extraction**
- 
  Extracting the radius and refractive index of particles.

- DTEx215 **Cell Counting**

  Counting the number of cells in fluorescence images.

- DTEx216 **3D Multi-Particle tracking**

  Tracking multiple particles in 3D for holography.

- DTEx217 **GAN image generation**

  Using a GAN to create cell image from masks.

Specific examples for label-free particle tracking using **LodeSTAR**:

- DTEx231A **LodeSTAR Autotracker Template**

- DTEx231B **LodeSTAR Detecting Particles of Various Shapes**

- DTEx231C **LodeSTAR Measuring the Mass of Particles in Holography**

- DTEx231D **LodeSTAR Detecting the Cells in the BF-C2DT-HSC Dataset**

- DTEx231E **LodeSTAR Detecting the Cells in the Fluo-C2DT-Huh7 Dataset**
  
- DTEx231F **LodeSTAR Detecting the Cells in the PhC-C2DT-PSC Dataset**
  
- DTEx231G **LodeSTAR Detecting Plankton**
  
- DTEx231H **LodeSTAR Detecting in 3D Holography**

- DTEx231I **LodeSTAR Measuring the Mass of Simulated Particles**
  
- DTEx231J **LodeSTAR Measuring the Mass of Cells**

Specific examples for graph-neural-network-based particle linking and trace characterization using **MAGIK**:

- DTEx241A **MAGIK Tracing Migrating Cells**

  <!-- GH_ONLY_START -->
  <p align="left">
    <img width="600" src=/assets/Tracing.gif?raw=true>
    <br/>
    <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx241A_MAGIK_cell_migration_analysis.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://doi.org/10.1038/s42256-022-00595-0" alt="Article MAGIK">
      <img src="https://img.shields.io/badge/article-10.1038/s42256--022--00595--0-blue">
    </a>  
  </p>
  <!-- GH_ONLY_END -->

- DTEx241B **MAGIK to Track HeLa Cells**

# Advanced Tutorials

This section provides a list of advanced topic tutorials. The primary focus of these tutorials is to demonstrate the functionalities of individual modules and how they work in relative isolation, helping to provide a better understanding of them and their roles in DeepTrack2.

- DTAT301 **[deeptrack.features](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT301_features.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT301_features.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT306 **[deeptrack.properties](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT306_properties.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT306_properties.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT311 **[deeptrack.image](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT311_image.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT311_image.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT321 **[deeptrack.scatterers](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT321_scatterers.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT321_scatterers.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT323 **[deeptrack.optics](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT323_optics.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT323_optics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT324 **[deeptrack.holography](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT324_holography.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT324_holography.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT325 **[deeptrack.aberrations](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT325_aberrations.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT325_aberrations.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT327 **[deeptrack.noises](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT327_noises.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT327_noises.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT329 **[deeptrack.augmentations](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT329_augmentations.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT329_augmentations.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT341 **[deeptrack.sequences](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT341_sequences.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT341_sequences.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT381 **[deeptrack.math](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT381_math.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT381_math.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT383 **[deeptrack.utils](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT383_utils.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT383_utils.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT385 **[deeptrack.statistics](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT385_statistics.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT385_statistics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT387 **[deeptrack.types](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT_387_types.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT_387_types.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT389 **[deeptrack.elementwise](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT389_elementwise.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT389_elementwise.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT391A **[deeptrack.sources.base](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT391A_sources.base.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT391A_sources.base.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT391B **[deeptrack.sources.folder](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT391B_sources.folder.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT391B_sources.folder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT391C **[deeptrack.sources.rng](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT391C_sources.rng.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT391C_sources.rng.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT393A **[deeptrack.pytorch.data](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT393A_pytorch.features.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT393A_pytorch.features.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT393B **[deeptrack.pytorch.features](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT393B_pytorch.data.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT393B_pytorch.data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT395 **[deeptrack.extras.radialcenter](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT395_extras.radialcenter.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT395_extras.radialcenter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- DTAT399A **[deeptrack.backend.core](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399A_backend.core.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399A_backend.core.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  
- DTAT399B **[deeptrack.backend.pint_definition](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399B_backend.pint_definition.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399B_backend.pint_definition.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  
- DTAT399C **[deeptrack.backend.units](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399C_backend.units.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399C_backend.units.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  
- DTAT399D **[deeptrack.backend.polynomials](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399D_backend.polynomials.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399D_backend.polynomials.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  
- DTAT399E **[deeptrack.backend.mie](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399E_backend.mie.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399E_backend.mie.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  
- DTAT399F **[deeptrack.backend._config](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399F_backend._config.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT399F_backend._config.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Developer Tutorials

Here you find a series of notebooks tailored for DeepTrack2's developers:

- DTDV401 **[Overview of Code Base](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/4-developers/DTDV401_overview.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/4-developers/DTDV401_overview.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTDV411 **[Style Guide](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/4-developers/DTDV411_style.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/4-developers/DTDV411_style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# Documentation

The detailed documentation of DeepTrack2 is available at the following link: [https://deeptrackai.github.io/DeepTrack2](https://deeptrackai.github.io/DeepTrack2)

# Cite us!
If you use DeepTrack 2.1 in your project, please cite us:

<https://pubs.aip.org/aip/apr/article/8/1/011310/238663>
```
"Quantitative Digital Microscopy with Deep Learning."
Benjamin Midtvedt, Saga Helgadottir, Aykut Argun, Jesús Pineda, Daniel Midtvedt & Giovanni Volpe.
Applied Physics Reviews, volume 8, article number 011310 (2021).
```

See also:

<https://nostarch.com/deep-learning-crash-course>
```
Deep Learning Crash Course
Benjamin Midtvedt, Jesús Pineda, Henrik Klein Moberg, Harshith Bachimanchi, Joana B. Pereira, Carlo Manzo & Giovanni Volpe.
2025, No Starch Press (San Francisco, CA)
ISBN-13: 9781718503922
```


<https://www.nature.com/articles/s41467-022-35004-y>
```
"Single-shot self-supervised object detection in microscopy." 
Benjamin Midtvedt, Jesús Pineda, Fredrik Skärberg, Erik Olsén, Harshith Bachimanchi, Emelie Wesén, Elin K. Esbjörner, Erik Selander, Fredrik Höök, Daniel Midtvedt & Giovanni Volpe
Nature Communications, volume 13, article number 7492 (2022).
```

<https://www.nature.com/articles/s42256-022-00595-0>
```
"Geometric deep learning reveals the spatiotemporal fingerprint of microscopic motion."
Jesús Pineda, Benjamin Midtvedt, Harshith Bachimanchi, Sergio Noé, Daniel Midtvedt, Giovanni Volpe & Carlo Manzo
Nature Machine Intelligence volume 5, pages 71–82 (2023).
```

<https://doi.org/10.1364/OPTICA.6.000506>
```
"Digital video microscopy enhanced by deep learning."
Saga Helgadottir, Aykut Argun & Giovanni Volpe.
Optica, volume 6, pages 506-513 (2019).
```

# Funding

This work was supported by the ERC Starting Grant ComplexSwimmers (Grant No. 677511), the ERC Starting Grant MAPEI (101001267), and the Knut and Alice Wallenberg Foundation.
