

from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Particles import Particle
from DeepTrack.Noise import Noise
from DeepTrack.Backend.Distributions import draw
from DeepTrack.Backend.Image import Label, Feature, Image
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
    
    # Generates a single random image.
    def get(self, Features):
        if isinstance(Features, List):
            Images = [F.__resolve__(self.Optics) for F in reversed(Features)]
            for F in Features:
                F.__clear__()
        else:
            Images = Features.__resolve__(self.Optics)
            Features.__clear__()
        
        return Images


    def get_epoch(self):
        return self.epoch


    def I2I_generator(self,
                    Features,
                    Labels,
                    batch_size=1,
                    callbacks=None,
                    augmentation=None,
                    shuffle_batch=True):

        if isinstance(Features, str):
            assert os.path.exists(Features), "Path does not exist"

            if os.path.isdir(Features):
                Features = [os.path.join(Features,file) for file in os.listdir(Features) if os.path.isfile(os.path.join(Features,file))]
            else:
                Features = [Features]
            
            get_one = self._get_from_path(Features)
        else:
            get_one = self._get_from_map([Features, Labels])
        
        while True:
            batch = []
            labels = []
            for _ in range(batch_size):
                
                Image_pair = next(get_one)
                Image = [Image_pair[0]]
                Labels = [Image_pair[1]]
                
                aug_images = self.augment(Image, augmentation)
                aug_labels = self.augment(Labels, augmentation)

                for i in range(len(aug_images)):
                    batch.append(aug_images[i])
                    labels.append(aug_labels[i])

            if shuffle_batch:
                self.shuffle(batch,labels)

            for i0 in range(0, len(batch), batch_size):
                sub_batch =  batch[i0:i0+batch_size]
                sub_labels = labels[i0:i0+batch_size]
                
                if callbacks is not None:
                    if not isinstance(callbacks, List):
                        callbacks = [callbacks]
                    for c in callbacks:
                        c(self, [sub_batch,sub_labels])
                sub_batch = np.array(sub_batch)
                sub_labels = np.array(sub_labels)
                if sub_batch.ndim == 3: # Needs to add a channel
                    sub_batch = np.expand_dims(sub_batch, axis=-1)
                if sub_labels.ndim == 3: # Needs to add a channel
                    sub_labels = np.expand_dims(sub_labels, axis=-1)
                yield (np.array(sub_batch), np.array(sub_labels))
        
    
    def I2L_generator(self,
                    Features,
                    Labels,
                    batch_size=1,
                    callbacks=None,
                    augmentation=None,
                    shuffle_batch=True):

        if isinstance(Features, str):
            assert os.path.exists(Features), "Path does not exist"

            if os.path.isdir(Features):
                Features = [os.path.join(Features,file) for file in os.listdir(Features) if os.path.isfile(os.path.join(Features,file))]
            else:
                Features = [Features]
            
            get_one = self._get_from_path(Features)
        else:
            get_one = self._get_from_map(Features)

        

        while True:
            batch = []
            labels = []
            for _ in range(batch_size):
                
                Image = [next(get_one)]

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
                sub_batch = np.array(sub_batch)
                sub_labels = np.array(sub_labels)
                if sub_batch.ndim == 3 and sub_labels.ndim == 2: # Needs to add a channel
                    sub_batch = np.expand_dims(sub_batch, axis=-1)
                yield (np.array(sub_batch), np.array(sub_labels))
                self.epoch += 1


    def generate(self,
                    Features,
                    Labels,
                    batch_size=1,
                    callbacks=None,
                    augmentation=None,
                    shuffle_batch=True):
        
        # If string, handle as path
        if isinstance(Labels, Feature):
            G = self.I2I_generator(
                Features,
                Labels,
                batch_size,
                callbacks,
                augmentation,
                shuffle_batch
            )
        else:
            G = self.I2L_generator(
                Features,
                Labels,
                batch_size,
                callbacks,
                augmentation,
                shuffle_batch
            )
        while True:
            yield next(G) 
                

                

    # Placeholder
    def augment(self, Images, Augmentations):
        if Augmentations is None:
            return Images
        else:
            if not isinstance(Augmentations,List):
                Augmentations = [Augmentations]
            for a in Augmentations:
                Images = a(Images)
            return Images

        
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

        l = []
        for L in Labels:
            l.append(Label(L)(image.properties))
        return l
        

    
    def _get_from_path(self, paths):
        if not isinstance(paths, List):
            paths = [paths]
        
        while True:
            for path in paths:
                Images = np.load(path)
                for I in Images:
                        yield I

    def _get_from_map(self, Features):
        while True:
            yield self.get(Features)
                

    