import numpy as np

from typing import List
from tensorflow import keras
from deeptrack.features import Feature
from deeptrack.image import Image




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


    def get(self, features: Feature or List[Feature]) -> Image:
        '''
        Resolves a single set of images given an input set of features before
        clearing the cache.

        If the input is a list, the function will iterate through them all
        before clearing the cache.
        '''

        if isinstance(features, List):
            for feature in features:
                feature.update()
            return [feature.resolve() for feature in reversed(features)]

        else:
            features.update()
            return features.resolve()



    def generate(self,
                 features,
                 label_function=None,
                 batch_size=1,
                 shuffle_batch=True):

        get_one = self._get_from_map(features)
        while True:
            batch = []
            labels = []

            for _ in range(batch_size):
                image = next(get_one)
                batch.append(image)
                if not label_function is None:
                    labels.append(label_function(image))

            if shuffle_batch:
                self.shuffle(batch, labels)

            batch = np.array(batch)
            labels = np.array(labels)

            if not label_function is None:
                yield batch, labels
            else: 
                yield batch


    def shuffle(self, x, y):
        import random
        start_state = random.getstate()
        random.shuffle(x)
        random.setstate(start_state)
        random.shuffle(y)


    def _get_from_map(self, features):
        while True:
            yield self.get(features)
