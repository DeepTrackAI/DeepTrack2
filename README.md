<!-- GH_ONLY_START -->
<p align="center">
  <img width="350" src=https://github.com/DeepTrackAI/DeepTrack2/blob/develop/assets/logo.png?raw=true>
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

- DTGS131 **[Tracking a Multiple Particles with a U-Net](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS131_tracking_multiple_particles_unet.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS131_tracking_multiple_particles_unet.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Tracking multiple particles using a U-net trained on simulated images.

- DTGS141 **[Tracking a Multiple Particles with a U-Net](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS141_distinguishing_particles_in_brightfield.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS141_distinguishing_particles_in_brightfield.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Tracking and distinguishing particles of different sizes in brightfield microscopy using a U-net trained on simulated images.

- DTGS151 **[Unsupervised Object Detection](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS151_unsupervised_object_detection_with_lodestar.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/1-getting-started/DTGS151_unsupervised_object_detection_with_lodestar.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Single-shot unsupervised object detection a using LodeSTAR.

# Examples

These are examples of how DeepTrack2 can be used on real datasets:

- DTEx201 **[MNIST](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx201_MNIST.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx201_MNIST.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Training a fully connected neural network to identify handwritten digits using MNIST dataset.

- DTEx202 **[Single Particle Tracking](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx202_single_particle_tracking.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx202_single_particle_tracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

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

- DTEx203 **[Multi-Particle tracking](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx203_particle_sizing.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx203_particle_sizing.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

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

- DTEx204 **[Particle Feature Extraction](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx204_multi_molecule_tracking.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx204_multi_molecule_tracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Extracting the radius and refractive index of particles.

- DTEx205 **[Cell Counting](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx205_inline_holography_3d_tracking.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx205_inline_holography_3d_tracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Counting the number of cells in fluorescence images.

- DTEx206 **[3D Multi-Particle tracking](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx206_cell_counting.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx206_cell_counting.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Tracking multiple particles in 3D for holography.

- DTEx207 **[GAN image generation](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx207_GAN_image_generation.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx207_GAN_image_generation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

  Using a GAN to create cell image from masks.

Specific examples for label-free particle tracking using **LodeSTAR**:

- DTEx231A **[LodeSTAR Autotracker Template](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231A_LodeSTAR_autotracker_template.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231A_LodeSTAR_autotracker_template.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231B **[LodeSTAR Detecting Particles of Various Shapes](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231B_LodeSTAR_tracking_particles_of_various_shapes.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231B_LodeSTAR_tracking_particles_of_various_shapes.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231C **[LodeSTAR Measuring the Mass of Particles in Holography](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231C_LodeSTAR_measure_mass_experimental.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231C_LodeSTAR_measure_mass_experimental.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231D **[LodeSTAR Detecting the Cells in the BF-C2DT-HSC Dataset](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231D_LodeSTAR_track_BF-C2DT-HSC.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231D_LodeSTAR_track_BF-C2DT-HSC.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231E **[LodeSTAR Detecting the Cells in the Fluo-C2DT-Huh7 Dataset](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231E_LodeSTAR_track_Fluo-C2DT-Huh7.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231E_LodeSTAR_track_Fluo-C2DT-Huh7.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231F **[LodeSTAR Detecting the Cells in the PhC-C2DT-PSC Dataset](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231F_LodeSTAR_track_PhC-C2DT-PSC.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231F_LodeSTAR_track_PhC-C2DT-PSC.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231G **[LodeSTAR Detecting Plankton](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231G_LodeSTAR_track_plankton.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231G_LodeSTAR_track_plankton.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231H **[LodeSTAR Detecting in 3D Holography](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231H_LodeSTAR_track_3D_holography.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231H_LodeSTAR_track_3D_holography.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231I **[LodeSTAR Measuring the Mass of Simulated Particles](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231I_LodeSTAR_measure_mass_simulated.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx231I_LodeSTAR_measure_mass_simulated.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTEx231J **[LodeSTAR Measuring the Mass of Cells](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/examples/DTEx231J_LodeSTAR_measure_mass_cell.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/examples/DTEx231J_LodeSTAR_measure_mass_cell.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Specific examples for graph-neural-network-based particle linking and trace characterization using **MAGIK**:

- DTEx241A **[MAGIK Tracing Migrating Cells](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx241A_MAGIK_cell_migration_analysis.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx241A_MAGIK_cell_migration_analysis.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

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

- DTEx241B **[MAGIK to Track HeLa Cells](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx241B_MAGIK_tracking_hela_cells.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/2-examples/DTEx241B_MAGIK_tracking_hela_cells.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# Advanced Tutorials

- DTAT301 **[Features](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT301_features.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT301_features.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT306 **[Properties](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT306_properties.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT306_properties.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT311 **[Image](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT311_image.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT311_image.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT321 **[Scatterers](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT321_scatterers.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT321_scatterers.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT323 **[Optics](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT323_optics.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT323_optics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT325 **[Aberrations](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT325_aberrations.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT325_aberrations.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT327 **[Noises](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT327_noises.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT327_noises.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT329 **[Augmentations](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT329_augmentations.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT329_augmentations.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT341 **[Sequences](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT341_sequences.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT341_sequences.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT381 **[Math](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT381_math.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT381_math.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

- DTAT383 **[Utils](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/3-advanced-topics/DTAT383_utils.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials/exadvanced-topicsamples/DTAT383_utils.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>


# Developer Tutorials

Here you find a series of notebooks tailored for DeepTrack2's developers:

- DTDV401 **[Overview of Code Base]**

- DTDV411 **[Style Guide](https://github.com/DeepTrackAI/DeepTrack2/blob/develop/tutorials/4-developers/DTDV111_style.ipynb)** <a href="https://colab.research.google.com/github/DeepTrackAI/DeepTrack2/blob/develop/tutorials4-developers/DTDV411_style.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>

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
"Geometric deep learning reveals the spatiotemporal fingerprint ofmicroscopic motion."
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
