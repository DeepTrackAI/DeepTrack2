

from DeepTrack.Optics import BaseOpticalDevice2D
from DeepTrack.Particles import Particle
from DeepTrack.Noise import Noise
from DeepTrack.Backend.Distributions import draw
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

    def generate(self,
                    Features,
                    Labels,
                    batch_size=1,
                    callbacks=None,
                    augmentation=None,
                    shuffle_batch=True):

        # If string, handle as path
        if isinstance(Features, str):
            assert os.path.exists(Features), "Path does not exist"

            if os.path.isdir(Features):
                Features = [os.path.join(Features,file) for file in os.listdir(Features) if os.path.isfile(f)]
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
                    batch.append(np.array(augmented_image))
                    labels.append(np.array(Label))

            idx = np.arange(len(batch))
            if shuffle_batch:
                np.random.shuffle(idx)
            batch = np.array(batch)
            labels = np.array(labels)
            for i0 in range(0, len(batch), batch_size):
                i_take = idx[i0:i0+batch_size]
                sub_batch = batch[i_take]
                sub_labels =labels[i_take]
                yield (sub_batch, sub_labels)

    # Placeholder
    def augment(self, Image, Augmentations):
        if Augmentations is None:
            return [Image]
        else:
            return [Image]
        

    # Placeholder
    def get_labels(self, image, Labels):
        return np.array([0])
            
    
    def _get_from_path(self, paths):
        if not isinstance(path, List):
            paths = [paths]
        
        while True:
            for path in paths:
                Images = np.load(path) 
                if isinstance(Images, List):
                    for Image in Images:
                        yield Image
                else:
                    yield Images

    def _get_from_map(self, FeatureMap):
        while True:
            yield self.get(FeatureMap)
                

    