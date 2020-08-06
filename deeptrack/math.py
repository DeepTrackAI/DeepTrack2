''' Mathematical oprations and structures

Classses
--------
Clip
    Clip the input within a minimum and a maximum value.
NormalizeMinMax
    Min-max image normalization.
'''

from deeptrack.features import Feature
from deeptrack.image import Image
import numpy as np

class Average(Feature):
    ''' Average of input images

    If `features` is not None, it instead resolves all features
    in the list and averages the result.

    Parameters
    ----------
    axis : int or tuple of ints
        Axis along which to average
    features : list of features, optional
    '''
    
    __distributed__ = False

    def __init__(self, features=None, axis=0, **kwargs):
        super().__init__(axis=axis, features=features, **kwargs)
    
    def get(self, images, axis, features, **kwargs):
        if features is not None:
            images = [feature.resolve() for feature in features]
        result = Image(np.mean(images, axis=axis))

        for image in images:
            result.merge_properties_from(image)
        
        return result

class Clip(Feature):
    '''Clip the input within a minimum and a maximum value.

    Parameters
    ----------
    min : float
        Clip the input to be larger than this value.
    max : float
        Clip the input to be smaller than this value.
    '''

    def __init__(self, min=-np.inf, max=+np.inf, **kwargs):
        super().__init__(min=min, max=max, **kwargs)



    def get(self, image, min=None, max=None, **kwargs):
        np.clip(image, min, max, image)
        return image 


    
class NormalizeMinMax(Feature):
    '''Image normalization.
    
    Transforms the input to be between a minimum and a maximum value using a linear transformation.

    Parameters
    ----------
    min : float
        The minimum of the transformation.
    max : float
        The maximum of the transformation.
    '''

    def __init__(self, min=0, max=1, **kwargs):
        super().__init__(min=min, max=max, **kwargs)



    def get(self, image, min, max, **kwargs):
        image = image / (np.max(image) - np.min(image)) * (max - min)
        image = image - np.min(image) + min 
        image[np.isnan(image)] = 0
        return image
