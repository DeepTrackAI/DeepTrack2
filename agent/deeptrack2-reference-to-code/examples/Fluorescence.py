import numpy as np
import deeptrack as dt
from matplotlib import pyplot as plt

# ## Fluorescence
# Fluorescence images in `DeepTrack2` are simulated with discrete volumes to act as light sources (fluorophores) which emit light from a scatterer.
# 
# Simulating fluorescence images are easy and straightforward to implement, we start by defining our microscope.
# Create a fluorescence microscope for a given wavelength.

IMAGE_SIZE = 150

particle = dt.Sphere(
    position=(IMAGE_SIZE//8, IMAGE_SIZE//4, 2) * dt.units.pixel, #put particles in the middle
    radius= 0.3e-6,
    position_unit="pixel",
    refractive_index=1.42,
)

fluorescence_optics = dt.Fluorescence(
    NA=1.4,
    resolution=1e-6,
    magnification=10,
    wavelength=600e-9,
    padding=(32, 32, 32, 32),
    output_region=(0, 0, IMAGE_SIZE//4, IMAGE_SIZE//2),
)

# Image the particle.
imaged_particle = fluorescence_optics(particle)

# Add background noise.
fluorescence_image = (imaged_particle >> dt.Gaussian(0, 0.001)).resolve()

plt.imshow(fluorescence_image, cmap="gray")
plt.show()