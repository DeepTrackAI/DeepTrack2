


import sys
sys.path.append("./")

from DeepTrack.DataGeneration.Generator import Generator
from DeepTrack.DataGeneration.Particles import SphericalParticle
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

G = Generator(
    shape = (256,256),
    NA = 0.7,
    pixel_size=0.1,
    wavelength=0.68
)
G.add_particle(SphericalParticle(radius = 0.1, intensity = 0.5))


start = timer()

for i in range(100):
    image, position = G.get()

end = timer()

print("Generates (256,256) particles at {0}s per image".format((end - start)/100))

for i in range(1):
    image, position = G.get()
    plt.imshow(image)
    plt.scatter(position[0], position[1], 2)
    plt.show()

