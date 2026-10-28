import numpy as np
import deeptrack as dt
from matplotlib import pyplot as plt

# ## Brightfield
# 
# In `DeepTrack2`, the brightfield microscope model simulates illumination using a single wavelength of light. This is equivalent to a brightfield microscope operating with a monochromatic light source.
# 
# In experimental setups however, a brightfield microscope typically uses a broad spectrum of wavelengths (white light).
# 
# To achieve a more realistic simulation of white light, we will need to simulate multiple brightfield images (sampling different parts of the visible light spectrum) and then averaging the contributions of these images. 
# 
# We start by generating a spectrum of wavelengths and a particle scatterer object. 

IMAGE_SIZE = 150

# Generate a spectrum of wavelengths to sample from, 5 will be enough.
wavelengths = np.linspace(450e-9, 700e-9, 5)

particle = dt.Sphere(
    position=(IMAGE_SIZE//2, IMAGE_SIZE//4, 2) * dt.units.pixel, #put particles in the middle
    radius= 0.3e-6,
    position_unit="pixel",
    refractive_index=1.42,
)


# Image the particle by sampling the spectrum.

# Sample the wavelengths.
imaged_particle_list = []
for wavelength in wavelengths:
    # Create a brightfield microscope for a given wavelength.
    single_wavelength_optics = dt.Brightfield(
        resolution=1e-6,
        magnification=10,
        wavelength=wavelength,
        padding=(32, 32, 32, 32),
        output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE//2),
    )

    # Image the particle.
    imaged_particle = single_wavelength_optics(particle)

    # Add background noise.
    imaged_particle = imaged_particle >> dt.Gaussian(0, 0.01)

    # Append to list.
    imaged_particle_list.append(imaged_particle)

# Take the average of the images in the list.
brightfield_image = (
    sum(imaged_particle_list) / len(imaged_particle_list)
).resolve()
plt.imshow(brightfield_image, cmap="gray")
plt.show()