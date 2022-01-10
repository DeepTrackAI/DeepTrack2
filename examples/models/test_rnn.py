# Add deeptrack to path
import sys

sys.path.append(".")

import deeptrack as dt
import numpy as np

# Centered particle with random radius
particle = dt.Sphere(position=(14, 14), radius=5 * dt.units.px, z=0)

particle = dt.Sequential(
    particle, z=lambda previous_value: previous_value + np.random.randn() * 3
)

optics = dt.Fluorescence(output_region=(0, 0, 28, 28))

data = optics(particle)
label = particle.z

# Pipeline that resolves data and label
dataset = data & label

dataset = dt.Sequence(dataset, sequence_length=10)

# Create model
model = dt.models.RNN(
    input_shape=(None, 28, 28, 1),
    number_of_outputs=1,
    return_sequences=True,
    loss="mae",
)

h = model.fit(dataset, epochs=20, batch_size=8)

assert (
    h.history["loss"][0] / h.history["loss"][-1] > 2
), "Loss did not improve substantially enough."
