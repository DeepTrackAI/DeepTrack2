


import sys
sys.path.append("../DeepTrack")

from DeepTrack.Generators import Generator
from DeepTrack.Particles import PointParticle
from DeepTrack.Backend.Distributions import uniform_random
from DeepTrack.Noise import Gaussian, Offset
from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Backend.Image import Image
from DeepTrack.Callbacks import Storage
from DeepTrack.Augmentation import FlipLR, FlipUD, NormalizeMinMax

from tensorflow import keras
from tensorflow.keras.layers import Conv2D as Conv, MaxPooling2D as Pool, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

'''
    Simple example showcasing current possiblites.
    
    A generator is created with certain optical properties. A simple tracker is trained.
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
    position_distribution=uniform_random((64,64,20))           # The distrbution from which to draw the position of the particle
)


# Create a model. This model is likely too simple to achieve subpixel resolution.
model = keras.models.Sequential()
model.add(Conv(16, kernel_size=3, activation="relu", padding="same", input_shape=(64,64,1)))
model.add(Pool(2))
model.add(Conv(32, kernel_size=3, activation="relu", padding="same"))
model.add(Pool(2))
model.add(Conv(64, kernel_size=3, activation="relu", padding="same"))
model.add(Conv(2, kernel_size=3, activation="relu", padding="same"))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(2))
model.compile(keras.optimizers.Adam(), loss="mae")


# Create your generators. (Features to generate, Labels to extract, augmentations, batch_size)
training_generator = G.generate(P, ["x", "y"], augmentation=[NormalizeMinMax(), FlipLR(), FlipUD()], batch_size=100)
validation_generator = G.generate(P, ["x", "y"], augmentation=[NormalizeMinMax(), FlipLR(), FlipUD()], batch_size=4)

model.fit_generator(training_generator,  
                        steps_per_epoch=4,
                        epochs=100, 
                        validation_data=validation_generator,
                        validation_steps=1,
                        workers=1)

test_batch, labels = next(validation_generator)
test_prediction = model.predict(test_batch)


plt.gray()
for i in range(4):
    plt.subplot(221 + i)
    plt.imshow(np.squeeze(test_batch[i,:,:,0]))
    plt.scatter(labels[i,0], labels[i,1], 20, 'g')
    plt.scatter(test_prediction[i,0], test_prediction[i,1], 20, 'b')
plt.show()


