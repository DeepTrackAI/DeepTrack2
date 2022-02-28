# Add deeptrack to path
import sys

sys.path.append(".")

import deeptrack as dt
import numpy as np

# Centered particle with random radius
particle = dt.Sphere(position=lambda: np.random.rand(2) * 32, radius=5 * dt.units.px)

optics = dt.Fluorescence(output_region=(0, 0, 32, 32))

data = optics(particle)
label = particle >> dt.SampleToMasks(
    lambda: lambda x: np.any(x, axis=-1, keepdims=True),
    output_region=optics.output_region,
    merge_method="or",
)

# Pipeline that resolves data and label
dataset = data & label


# Create model
model = dt.models.UNet(
    input_shape=(None, None, 1),
    number_of_outputs=1,
    output_activation="sigmoid",
    loss="binary_crossentropy",
)

h = model.fit(dataset, epochs=20, batch_size=16)

assert (
    h.history["loss"][0] / h.history["loss"][-1] > 2
), "Loss did not improve substantially enough."
