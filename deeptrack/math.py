from deeptrack.features import Feature
import numpy as np

class Clip(Feature):
    def __init__(self, min=-np.inf, max=+np.inf, **kwargs):
        super().__init__(min=min, max=max, **kwargs)


    def get(self, image, min=None, max=None, **kwargs):
        np.clip(image, min, max, image)
        return image 


class NormalizeMinMax(Feature):
    def __init__(self, min=0, max=1, **kwargs):
        super().__init__(min=min, max=max, **kwargs)


    def get(self, image, min=None, max=None, **kwargs):
        image = image / np.max(image) * (max - min)
        image = image - np.min(image) + min 
        return image