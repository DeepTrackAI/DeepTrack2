# Add deeptrack to path
import sys

sys.path.append(".")

import deeptrack as dt
import numpy as np

# Centered particle with random radius
particle = dt.Ellipsoid(
    position=(25, 25),
    radius=(10, 5) * dt.units.px,
)

optics = dt.Fluorescence(output_region=(0, 0, 50, 50))

data = optics(particle)

# Create model
model = dt.models.AutoTracker(input_shape=(50, 50, 1), symmetries=2, mode="orientation")

h = model.fit(data, epochs=10, batch_size=8)

assert (
    h.history["loss"][0] / h.history["loss"][-1] > 2
), "Loss did not improve substantially enough."
