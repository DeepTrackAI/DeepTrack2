#%%
import sys 
sys.path.insert(0, '../../..')

import numpy as np
import matplotlib.pyplot as plt

import deeptrack as dt
import deeptrack.extras
deeptrack.extras.datasets.load("ParticleTracking")


#%%

"""
This example shows how to train a convolutional neural network to track a particle in a 2D image 
without knowing the particle's position in advance. First we load two videos of a particle moving
in a 2D image.
"""


import cv2

path_to_ideal_video = "datasets/ParticleTracking/ideal.avi"
path_to_bad_video = "datasets/ParticleTracking/bad.avi"

def load_video(path):
    cap =   cv2.VideoCapture(path)
    output_buffer = np.zeros((
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ))

    for i in range(output_buffer.shape[0]):
        ret, frame = cap.read()
        output_buffer[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return output_buffer

ideal_video = load_video(path_to_ideal_video)
bad_video = load_video(path_to_bad_video)

#%%

"""
We can now plot the first frames of the videos to see what they look like.
"""

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(ideal_video[i], cmap="gray")
    plt.axis("off")

for i in range(5):
    plt.subplot(2, 5, i+6)
    plt.imshow(bad_video[i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Ideal video shape:", ideal_video.shape)
print("Bad video shape:", bad_video.shape)

#%%

"""
Let us define a convolutional neural network that takes an image as input and outputs the position
of the particle in the image.
"""

model = dt.models.Convolutional(
    input_shape=(120, 120, 1),
    output_shape=(2,),
)

#%%

"""
We define a custom loss function that will allow us to train our neural network
without knowing the particle's position in advance.
"""
import tensorflow.keras.backend as K
def loss_function(y_true, y_pred):

    difference = y_true - y_pred

    # Assert that the standard deviation of the difference is small.
    return K.std(difference)

#%%

"""
We can now train our model on the ideal video. We will use the first 100 frames of the video.
"""

model.compile(loss=loss_function)

# we will train in a loop.
epochs = 100
batch_size = 8

for epoch in range(100):
    for image_index in range(100):
        image = ideal_video[image_index]
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        for batch_index in range(0, image.shape[0], batch_size):
            batch = image[batch_index:batch_index+batch_size]
            model.train_on_batch(batch, batch)

        # We will use the particle's position as the target.
        target = np.array([60, 60])

        model.train_on_batch(image, target)
    