import numpy as np
import deeptrack as dt
from matplotlib import pyplot as plt


IMAGE_SIZE = 150

# ## Holographic microscope
# The Holographic microscope returns an image with real and imaginary values, so we will need to define a helper function that converts these into floats so we can visualize the values with a plot.


def complex_to_float_f():
    """Converts a complex image to a float image, needed for Holography."""
    def inner(image):
        image = image - 1
        output = np.zeros((*image.shape[:2], 2))
        output[..., 0:1] = np.real(image)
        output[..., 1:2] = np.imag(image)
        return output
    return inner
complex_to_float = dt.Lambda(complex_to_float_f)

# Define the holography microscope and resolve a particle.

mie_scatterer = dt.MieSphere(
    position=(IMAGE_SIZE//4, IMAGE_SIZE//2, 2) * dt.units.pixel, #put particles in the middle
    radius= 2e-6,
    position_unit="pixel",
    refractive_index=1.42,
    L=10,
)

holography_microscope = dt.Holography(
    resolution=1e-6,
    magnification=10,
    wavelength=600e-9,
    padding=(32, 32, 32, 32),
    output_region=(0, 0, IMAGE_SIZE//2, IMAGE_SIZE),
    return_field=True,
)
# Get complex-valued image.
holography_image = holography_microscope(mie_scatterer).resolve()
print(f"Image shape = {holography_image.shape}")

# Print a value in the image.
print(f"Pixel value of (0, 0) = {holography_image[0, 0, 0]}")

# Convert to floats.
converted_image = complex_to_float(holography_image)
holography_real = converted_image[:, :, 0]
holography_imag = converted_image[:, :, 1]
holography_combined = holography_real + holography_imag

fig, ax = plt.subplots(1, 3)
ax[0].imshow(holography_real, cmap="gray")
ax[1].imshow(holography_imag, cmap="gray")
ax[2].imshow(holography_combined, cmap="gray")

plt.show()