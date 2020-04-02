''' Generators to continusously resolve features.

Classes
-------
Generator
    Base class for a generator.
'''

import numpy as np

from typing import List
from tensorflow import keras
from deeptrack.features import Feature
from deeptrack.image import Image



class Generator(keras.utils.Sequence):
    '''Base class for a generator.

    Generators continously update and resolve features, and allow other
    frameworks to continuously access new features.
    '''

    def generate(self,
                 feature,
                 label_function=None,
                 batch_size=1,
                 shuffle_batch=True,
                 ndim=4):
        ''' Create generator instance.
        
        Parameters
        ----------
        feature : Feature
            The feature to resolve images from.
        label_function : Callable[Image] -> array_like
            Function that returns the label corresponding to an image.
        batch_size : int
            Number of images per batch.
        shuffle_batch : bool
            If True, the batches are shuffled before outputting.
        ndim : int
            Expected number of dimensions of the output.
        '''

        get_one = self._get_from_map(feature)
        while True:
            batch = []
            labels = []
            # Yield batch_size results
            for _ in range(batch_size):
                image = next(get_one)
                batch.append(image)
                if label_function:
                    labels.append(label_function(image))

            if shuffle_batch:
                self._shuffle(batch, labels)

            batch = np.array(batch)
            labels = np.array(labels)

            # Console found batch_size with results
            if batch.ndim > ndim:
                dims_to_remove = batch.ndim - ndim
                batch = np.reshape(batch, (-1, *batch.shape[dims_to_remove + 1:]))
                labels = np.reshape(labels, (-1, *labels.shape[dims_to_remove + 1:]))
            elif batch.ndim < ndim:
                Warning("Incorrect number of dimensions. Found {0} with {1} dimensions, expected {2}.".format(batch.shape, batch.ndim, ndim))

            if label_function:
                yield batch, labels
            else:
                yield batch


    def _get(self, features: Feature or List[Feature]) -> Image:
        # Updates and resolves a feature or list of features.
        if isinstance(features, List):
            for feature in features:
                feature.update()
            return [feature.resolve() for feature in reversed(features)]
        else:
            features.update()
            return features.resolve()


    def _shuffle(self, x, y):
        # Shuffles the batch and labels equally along the first dimension
        import random
        start_state = random.getstate()
        random.shuffle(x)
        random.setstate(start_state)
        random.shuffle(y)


    def _get_from_map(self, features):
        # Continuously yield the output of _get
        while True:
            yield self._get(features)
