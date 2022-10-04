#%%

import deeptrack as dt

particle = dt.MieSphere(position=(32, 32), refractive_index=1.58, radius=228e-9)

optics = dt.Brightfield(
    magnification=1,
    resolution=0.115e-6,
    NA=0.8,
    wavelength=532e-9,
    output_region=(0, 0, 64, 64),
)

image_feature = optics(particle)

image = image_feature.resolve().get()
type(image.dtype)

import matplotlib.pyplot as plt

plt.imshow(image.real, cmap="gray")
plt.show()

plt.imshow(image.imag, cmap="gray")
plt.show()
# #%%

# import numpy as np
# import matplotlib.pyplot as plt

# positions = np.linspace(-200, 200, 10)
# center_phase = []
# for position in positions:
#     image = image_feature(position_objective=(position * 0.115e-6,) * 2)
#     image = np.angle(image)

#     plt.imshow(image)
#     plt.show()
#     center_phase.append(np.max(image))

# plt.plot(positions, center_phase)
# # %%
