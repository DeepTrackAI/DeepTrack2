# This file contains the different neural networks that proved succesful during testing stages of tracking plankton
from deeptrack.models import unet
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as C
import os  
from deeptrack.features import LoadImage
from deeptrack.math import NormalizeMinMax
import cv2

def softmax_categorical(T, P):
    eps = 1e-6
    classwise_weight = K.mean(1 - T, axis=(1, 2), keepdims=True)
    true_error = K.mean(T * K.log(P + eps) * classwise_weight, axis=-1)
    return -K.mean(true_error)

def generate_unet(im_size_height, im_size_width, no_of_inputs, no_of_outputs):
#     image_size
    model = unet(
    (im_size_height, im_size_width, no_of_inputs), 
    conv_layers_dimensions=[16, 32, 64, 128],
    base_conv_layers_dimensions=[32, 32],
    number_of_outputs=no_of_outputs,
    output_activation="softmax",
    loss=softmax_categorical
    )
    return model

def train_model_early_stopping(model, generator, patience=15, epochs=500, steps_per_epoch=10):
    callback = C.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

    with generator:
        model.fit(generator,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              callbacks=[callback],
              max_queue_size=0,
              workers=0)
    return model