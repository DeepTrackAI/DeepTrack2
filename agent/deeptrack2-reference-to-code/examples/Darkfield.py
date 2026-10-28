import numpy as np
import deeptrack as dt
from matplotlib import pyplot as plt

# ## Darkfield
# We will image a particle in a darkfield modality. This requires the scatterer to be defined as a Mie scatterer object.
# 

IMAGE_SIZE = 150

# Generate a spectrum of wavelengths to sample from, 5 will be enough.
wavelengths = np.linspace(450e-9, 700e-9, 5)

mie_scatterer = dt.MieSphere(
    position=(IMAGE_SIZE//4, IMAGE_SIZE//2, 2) * dt.units.pixel, #put particles in the middle
    radius= 2e-6,
    position_unit="pixel",
    refractive_index=1.42,
    L=10,
)


# Then, we define our darkfield microscope and image the particle through this.


optics = dt.Darkfield(
    resolution=1e-6,
    magnification=10,
    wavelength=600e-9,
    padding=(32, 32, 32, 32),
    output_region=(0, 0, IMAGE_SIZE//2, IMAGE_SIZE),
)

image = optics(mie_scatterer)

# Sample the wavelengths.
imaged_particle_list = []
for wavelength in wavelengths:

    # Create a darkfield microscope for a given wavelength.
    single_wavelength_optics = dt.Darkfield(
        resolution=1e-6,
        magnification=10,
        wavelength=wavelength,
        padding=(32, 32, 32, 32),
        output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
    )

    # Image the particle.
    imaged_particle = single_wavelength_optics(mie_scatterer)

    # Add background noise.
    imaged_particle = imaged_particle >> dt.Gaussian(0, 0.00015)

    # Append to list.
    imaged_particle_list.append(imaged_particle)

# Take the average of the images in the list.
darkfield_image = (
    sum(imaged_particle_list) / len(imaged_particle_list)
).resolve()
plt.imshow(darkfield_image, cmap="gray")
plt.show()