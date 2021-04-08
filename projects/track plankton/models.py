# This file contains the different neural networks that proved succesful during testing stages of tracking plankton
from deeptrack.models import unet
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



def model_predict_one_image(model, folder_path, frame, ):
    list_paths = os.listdir(folder_path)
    first_file_format = list_paths[0][-3:]
    for i in range(len(list_paths)):
        if list_paths[i][-3:] != first_file_format:
            print('Only the images to be analyzed can be in the folder with the images.')
            break
    input_shape = model.input_shape[1:]
    img = cv2.imread(folder_path +'\\' + list_paths[frame], 0)
    res = cv2.resize(img, dsize=(input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
    
    
    # return model.predict(im_stack)