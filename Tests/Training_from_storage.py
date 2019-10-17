


import sys
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D as Conv
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D as Pool

sys.path.append("../DeepTrack")

from DeepTrack.Augmentation import FlipLR, FlipUD, NormalizeMinMax, Transpose
from DeepTrack.Distributions import uniform_random
from DeepTrack.Backend.Image import Image
from DeepTrack.Labels import Label
from DeepTrack.Callbacks import Storage
from DeepTrack.Generators import Generator
from DeepTrack.Models import DeepTrackNetwork
from DeepTrack.Noise import Gaussian, Offset
from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Particles import PointParticle
from DeepTrack.Features import Load





'''
    Simple example showcasing current possiblites.
    
    A generator is created with certain optical properties. A simple tracker is trained on particles with added gaussian noise.
 '''


Optics = BaseOpticalDevice2D(
    shape=(64,64),      # Desired output shape of the generator.
    NA=0.7,             # The NA of the optical system.
    pixel_size=0.1e-6,     # The pixel_size of the optical system (m^-1).
    wavelength=0.68e-6     # The wavelength of the illuminating source (m).
)

G = Generator(Optics)

P = PointParticle(                                         # Radius of the generated particles
    intensity=np.linspace(50,100),                           # Peak intensity of the generated particle
    position=uniform_random((64,64,20))           # The distrbution from which to draw the position of the particle
)

N = Gaussian(
    sigma=np.linspace(0,0.001),
)

S = Storage("./Tests/Storage/Particle_Batch.npy", overwrite=False)
L = Label()["position"][0:2]
storage_generator =   G.generate(Optics(P), L, callbacks=[S], batch_size=100)

# Prefill the storage with 20 batches.
for _ in range(20):
    next(storage_generator)


model = DeepTrackNetwork(input_shape=(64,64,1), number_of_outputs=2)
model.compile(keras.optimizers.Adam(), loss="mse")

training_generator =   G.generate(Load("./Tests/Storage/") + N, L, augmentation=[NormalizeMinMax(), FlipLR(), FlipUD(), Transpose()], batch_size=100)
validation_generator = G.generate(Load("./Tests/Storage/") + N, L, augmentation=[NormalizeMinMax(), FlipLR(), FlipUD(), Transpose()], batch_size=10)

model.fit_generator(training_generator,  
                        steps_per_epoch=32,
                        epochs=50,
                        workers=1,
                        use_multiprocessing=False)

test_batch, labels = next(validation_generator)
test_prediction = model.predict(test_batch)


plt.gray()
for i in range(9):
    plt.subplot(331 + i)
    plt.imshow(np.squeeze(test_batch[i,:,:,0]))
    plt.scatter(labels[i,0], labels[i,1], 20, 'g')
    plt.scatter(test_prediction[i,0], test_prediction[i,1], 20, 'b')
plt.show()
