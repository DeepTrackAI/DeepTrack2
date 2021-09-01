# Add deeptrack to path
import sys

sys.path.append(".")

import deeptrack as dt
import numpy as np

# Centered particle with random radius
particle = dt.Sphere(
    position=(14, 14), radius=lambda: (3 + np.random.rand() * 3) * dt.units.px
)

optics = dt.Fluorescence(output_region=(0, 0, 28, 28))

data = optics(particle)


# Create model
model = dt.models.AutoTracker(input_shape=(28, 28, 1), symmetries=1, mode="tracking")

h = model.fit(data, epochs=20, batch_size=16)

assert (
    h.history["loss"][0] / h.history["loss"][-1] > 2
), "Loss did not improve substantially enough."
