

from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Particles import Particle
from DeepTrack.Noise import Noise
from DeepTrack.Backend.Distributions import draw
from DeepTrack.Backend.Image import Label
import random

from typing import List, Tuple, Dict, TextIO

import os
import numpy as np
from tensorflow import keras

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
class Generator(keras.utils.Sequence):
    def __init__(self,
        Optics
    ):
        self.Optics = Optics
        self.Particles = []
        self.Noise = []

    # Adds a particle to the set of particles that can be generated
    def add_particle(self, P):
        assert isinstance(P, Particle), "Argument supplied to add_particle is not an instance of Particle"
        
        self.Particles.append(P)

    def add_noise(self, N):
        assert isinstance(N, Noise), "Argument supplied to add_particle is not an instance of Noise"
        
        self.Noise.append(N)
    
    # Generates a single random image.
    def get(self, Features):
        
        Image = Features.resolve(self.Optics)
        
        return Image

    def get_epoch(self):
        return self.epoch
    def generate(self,
                    Features,
                    Labels,
                    batch_size=1,
                    callbacks=None,
                    augmentation=None,
                    shuffle_batch=True):
        
        self.epoch = 0
        # If string, handle as path
        if isinstance(Features, str):
            assert os.path.exists(Features), "Path does not exist"

            if os.path.isdir(Features):
                Features = [os.path.join(Features,file) for file in os.listdir(Features) if os.path.isfile(os.path.join(Features,file) )]
            else:
                Features = [Features]
            
            get_one = self._get_from_path(Features)
        else:
            get_one = self._get_from_map(Features)


        while True:
            batch = []
            labels = []
            for _ in range(batch_size):
                
                Image = next(get_one)

                for augmented_image in self.augment(Image, augmentation):
                    
                    Label = self.get_labels(augmented_image, Labels)
                    batch.append(augmented_image)
                    labels.append(np.array(Label))

            if shuffle_batch:
                self.shuffle(batch,labels)

            for i0 in range(0, len(batch), batch_size):
                sub_batch =  batch[i0:i0+batch_size]
                sub_labels = labels[i0:i0+batch_size]
                

                if callbacks is not None:
                    if not isinstance(callbacks, List):
                        callbacks = [callbacks]
                    
                    for c in callbacks:
                        c(self, sub_batch)

                yield (np.array(sub_batch), np.array(sub_labels))
                self.epoch += 1

                
                

                

    # Placeholder
    def augment(self, Image, Augmentations):
        if Augmentations is None:
            return [Image]
        else:
            return [Image]
        
    def shuffle(self,a,b):
        import random
        assert len(a) == len(b)
        start_state = random.getstate()
        random.shuffle(a)
        random.setstate(start_state)
        random.shuffle(b)
    # Placeholder
    
    def get_labels(self, image, Labels):
        if Labels is None:
            return np.array([0])
        
        if not isinstance(Labels, List):
            Labels = [Labels]

        for L in Labels:
            L = Label(L)
    
    def _get_from_path(self, paths):
        if not isinstance(paths, List):
            paths = [paths]
        
        while True:
            for path in paths:
                Images = np.load(path)
                for Image in Images:
                        yield Image

    def _get_from_map(self, FeatureMap):
        while True:
            yield self.get(FeatureMap)
                

    