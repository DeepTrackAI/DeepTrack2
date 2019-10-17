


import sys
sys.path.append("../DeepTrack")

from DeepTrack.Generators import Generator
from DeepTrack.Particles import PointParticle
from DeepTrack.Distributions import uniform_random
from DeepTrack.Noise import Gaussian, Offset, Poisson
from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Backend.Image import Image
from DeepTrack.Callbacks import Storage
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

'''
    Simple example showcasing current possiblites.
    
    A generator is created with certai optical properties. To the generator two particles are added.
 '''


Optics = BaseOpticalDevice2D(
    shape=(64,64),          # Desired output shape of the generator.
    NA=0.7,                 # The NA of the optical system.
    pixel_size=0.1e-6,      # The pixel_size of the optical system (m^-1).
    wavelength=0.68e-6,     # The wavelength of the illuminating source (m).
    upscale=4               # Finetunes the resolution of the pupil.
)

G = Generator(Optics)

P = PointParticle(                                         # Radius of the generated particles
    intensity=np.linspace(50,100),                           # Peak intensity of the generated particle
    position=uniform_random((64,64,20))           # The distrbution from which to draw the position of the particle
)

N1 = Poisson(
    SNr=np.linspace(10,20)
)

N2 = Offset(
    offset=np.linspace(0,0.2)
)


S = Storage("./Tests/Storage/Particle")
# Time the average generation time for 100 particles
start = timer()
images = next(G.generate(Optics(P + P) + N2 + N2, [], shape=(64,64), batch_size=100, callbacks=[S]))
end = timer()
print("Generates a {0} batch in {1}s".format(images[0].shape, (end - start)))

start = timer()
images = next(G.generate("./Tests/Storage", [], shape=(64,64), batch_size=100, callbacks=[S]))
end = timer()
print("Loads a {0} batch in {1}s".format(images[0].shape, (end - start)))


# Show one typical particle
plt.gray()
for i in range(1):
    Image = G.get((64,64), Optics(P) + N1)
    plt.imshow(np.abs(Image))
    plt.show()

