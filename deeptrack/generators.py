import numpy as np

from typing import List
from tensorflow import keras
from deeptrack.features import Feature
from deeptrack.image import Image




class Generator(keras.utils.Sequence):
    '''
    Base class for a generator.
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
                 shuffle_batch=True,
                 ndim=4):

        get_one = self._get_from_map(features)
        while True:
            batch = []
            labels = []

            for _ in range(batch_size):
                image = next(get_one)
                batch.append(image)
                if label_function:
                    labels.append(label_function(image))



            if shuffle_batch:
                self.shuffle(batch, labels)

            batch = np.array(batch)
            labels = np.array(labels)

            if batch.ndim > ndim:
                dims_to_remove = batch.ndim - ndim
                batch = np.reshape(batch, (-1, *batch.shape[dims_to_remove + 1:]))
                labels = np.reshape(labels, (-1, *labels.shape[dims_to_remove + 1:]))

            if label_function:
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
