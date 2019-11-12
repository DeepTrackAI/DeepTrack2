

from deeptrack.features import Feature
from deeptrack.image import Image
import copy
import random

from typing import List, Tuple, Dict, TextIO

import os
import numpy as np
from tensorflow import keras

from timeit import default_timer as timer


class Generator(keras.utils.Sequence):
    '''
    Base class for a generator.

    Generators combine a set of particles, an optical system and a ruleset
    to continuously create random images of particles.

    This base class convolves the intensity map of the particle with an optical pupil
    to simulate particles.

    Input arguments:
        shape           Shape of the output (tuple)
        wavelength      wavelength of the illumination source in microns (number)
        pixel_size      size of the pixels in microns (number)
        NA              the effective NA of the optical systen (number)          
    '''
    
    
    def get(self, shape, features:Feature or List[Feature]) -> Image:
        '''
        Resolves a single set of images given an input set of features before
        clearing the cache.

        If the input is a list, the function will iterate through them all
        before clearing the cache. 
        '''
    
        if isinstance(features, List):
            for f in features:
                f.update()
            return [feature.resolve(Image(np.zeros(shape))) for feature in reversed(features)]

        else:
            features.update()
            return features.resolve(Image(np.zeros(shape)))
            
        

    def generate(self,
                    features,
                    label_function = None,
                    shape=(64,64),
                    batch_size=1,
                    callbacks=None,
                    augmentation=None,
                    shuffle_batch=True):

        get_one = self._get_from_map(shape, features)
        while True:
            batch = []
            labels = []

            for _ in range(batch_size):
                image = next(get_one)
                batch.append(image)
                if not label_function is None:
                    labels.append(label_function(image))

            if shuffle_batch:
                self.shuffle(batch,labels)

            batch = np.array(batch)
            labels = np.array(labels)

            batch = np.expand_dims(batch, axis=-1)

            if not label_function is None:
                yield batch, labels
            else: 
                yield batch

        
    def shuffle(self,a,b):
        import random
        # assert len(a) == len(b)
        start_state = random.getstate()
        random.shuffle(a)
        random.setstate(start_state)
        random.shuffle(b)
    # Placeholder



    def _get_from_map(self, shape, features):
        while True:
            yield self.get(shape, features)
                

