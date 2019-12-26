from deeptrack.features import Feature
from deeptrack.image import Image
import numpy as np

class Clip(Feature):
    def __init__(self, *args, min=-np.inf, max=+np.inf, **kwargs):
        super().__init__(*args, min=min, max=max, **kwargs)


    def get(self, image, min=None, max=None, **kwargs):
        np.clip(image, min, max, image)
        return image 


class NormalizeMinMax(Feature):
    def __init__(self, *args, min=0, max=1, **kwargs):
        super().__init__(*args, min=min, max=max, **kwargs)


    def get(self, image, min=None, max=None, **kwargs):
        image = image / np.max(image) * (max - min)
        image = image - np.min(image) + min 
        return image

class Concatenate(Feature):

    __distributed__ = False

    def __init__(self, *args, features=None, axis=-1):

        

        super().__init__(*args, features=features, axis=axis)
    
    def get(self, image, features=None, axis=None):


        image_list = [feature.resolve(image) for feature in features]

        merged_image = Image(np.concatenate(image_list, axis=axis))
        
        image = Image(image)
        num_properties = len(image.properties)
        
        merged_properties = image.properties

        for im in image_list:
            merged_properties += im.properties[num_properties:]

        merged_image.properties = merged_properties

        return merged_image

