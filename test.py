#%%

import deeptrack as dt
import numpy as np

#%%
optics = dt.Fluorescence(
    resolution=2 * dt.units.um,
    wavelength=660 * dt.units.nm,
)
particle = dt.Sphere(position=lambda: np.random.uniform(0, 128, 2))
image = optics(particle ^ 2)

image.update().plot()
# %%
