import numpy as np
import deeptrack as dt
from matplotlib import pyplot as plt


IMAGE_SIZE = 150

mie_scatterer = dt.MieSphere(
    position=(IMAGE_SIZE//2, IMAGE_SIZE//4, 2) * dt.units.pixel, #put particles in the middle
    radius= 2e-6,
    position_unit="pixel",
    refractive_index=1.42,
    L=10,
)

iscat_microscope = dt.ISCAT(
    resolution=1e-6,
    magnification=10,
    wavelength=600e-9,
    padding=(32, 32, 32, 32),
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE//2),
)

iscat_image = iscat_microscope(mie_scatterer).resolve()

plt.imshow(iscat_image, cmap="gray")
plt.show()