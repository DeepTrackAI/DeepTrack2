import deeptrack as dt
import numpy as np
from matplotlib import pyplot as plt


# Define the new operator fo channel average
def channel_average(axis=0):
    def sny_function(image):
        return np.mean(image, axis=axis)
    return sny_function

# Wraper it using dt.Lambad, make it possilbe to ba chained using `>>`, `^`, and `&`
syn_feature = dt.Lambda(function=channel_average)

# Treat all input in one operation 
syn_feature.__distributed__ = False


IMAGE_SIZE = 150

# Define 2 images channel under two wavelength
particle = dt.Sphere(
    position= lambda: (
        np.array([
            0.1*IMAGE_SIZE + 0.7*IMAGE_SIZE*np.random.random(),
            0.1*IMAGE_SIZE + 0.7*IMAGE_SIZE*np.random.random(),
        ]) * dt.units.pixel
    ), #put particles randomly
    radius= 0.3e-6,
    position_unit="pixel",
    refractive_index=1.42,
)


fluorescence_optics_1 = dt.Fluorescence(
    NA=1.4,
    resolution=1e-6,
    magnification=10,
    wavelength=600e-9,
    padding=(32, 32, 32, 32),
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
)


fluorescence_optics_2 = dt.Fluorescence(
    NA=1.4,
    resolution=1e-6,
    magnification=10,
    wavelength=700e-9,
    padding=(32, 32, 32, 32),
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
)

# Image the particle.
imaged_particle_1 = fluorescence_optics_1(particle) >> dt.NormalizeMinMax()
imaged_particle_2 = fluorescence_optics_2(particle) >> dt.NormalizeMinMax()


# Add background noise.
fluorescence_image = ((imaged_particle_1 & imaged_particle_2) 
                      >> syn_feature >> dt.Gaussian(0, 0.001) >> dt.NormalizeMinMax()).update().resolve()

plt.imshow(fluorescence_image)
plt.show()