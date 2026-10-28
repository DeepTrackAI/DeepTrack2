import numpy as np
import deeptrack as dt
from matplotlib import pyplot as plt

IMAGE_SIZE = 256

# Define the optical setup

brightfield_microscope = dt.Brightfield(
    NA=0.9,
    resolution=1e-6,
    magnification=10,
    wavelength=680e-9,
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
    upscale=1,
    padding=(32, 32, 32, 32),
)

# Define the particle with random position

particle = dt.MieSphere(
    position=lambda: (
        np.array([
            0.1*IMAGE_SIZE + 0.7*IMAGE_SIZE*np.random.random(),
            0.1*IMAGE_SIZE + 0.7*IMAGE_SIZE*np.random.random(),
        ])
    ),
    radius= lambda: (0.25 + 0.25*np.random.random())*1e-6,
    intensity=10,
)

imaged_particle_with_random_position = brightfield_microscope(
    particle
)

# Normalized Operation

normalized_image_of_particle = (
    imaged_particle_with_random_position >> dt.NormalizeMinMax(0, 1)
)

output_image = imaged_particle_with_random_position()

# Visualization
plt.imshow(np.squeeze(output_image), cmap="gray")
plt.show()

# Way1: Get Vector-based Ground-Truth
position_of_particle = particle.position()

plt.imshow(np.squeeze(output_image), cmap="gray")
plt.scatter(position_of_particle[1], position_of_particle[0])
plt.show()

# Way2: Get Image-based Ground-Truth
CIRCLE_RADIUS = 3

def get_target_image(image, image_pipeline=normalized_image_of_particle):
    """Create a binary image with the circles in the particle positions."""

    target_image = np.zeros(image.shape)
    x, y = np.meshgrid(
        np.arange(0, image.shape[0]),
        np.arange(0, image.shape[1]),
    )

    positions = dt.TakeProperties(image_pipeline, "position").resolve()
    positions = np.reshape(
        positions, (-1, 2)
    )  # ensure (N, 2) shape, even for a single particle

    for position in positions:
        distance_map = (x - position[1]) ** 2 + (y - position[0]) ** 2
        target_image[distance_map < CIRCLE_RADIUS**2] = 1

    return target_image

target_image = normalized_image_of_particle >> get_target_image

# Visualization
plt.imshow(target_image(), cmap="gray")
plt.show()

# Move Axis for Torch (Channel-first)
torch_data_pipeline = (normalized_image_of_particle & (normalized_image_of_particle >> get_target_image)) >> dt.MoveAxis(2, 0)