from abc import ABC, abstractmethod
from DeepTrack.Distributions import Distribution, sample
import numpy as np
import copy
'''
Make a subclass of ndarray
'''
class Image:
    
    def __init__(self, features:list, copy_mode="deep"):
        if copy_mode == "deep":
            self.features = copy.deepcopy(features)
        elif copy_mode == "shallow":
            self.features = copy.copy(features)
        else:
            self.features = features

        self.has_updated_since_last_resolve = list((False) * len(features))

    def resolve(self, image, order=None):
        if order is None:
            order = np.arange(len(self.features))

        assert len(order) == len(self.features), "Order argument needs to be the same length as features"
        
        for feature in self.features:
            image = feature.resolve(image)
        
        return image
    
    def update(self):
        for feature, has_updated in zip(self.features, self.has_updated_since_last_resolve):
            if not has_updated:
                feature.update()
    
    def clear(self):
        for feature in self.features:
            feature.clear()
    
    def append(self, properties):
        self.features.append(properties)

    def add(self, feature):
        self.append(feature)

    def __add__(self, other):
        self.append(other)
    
    def __getitem__(self, key):
        return self.features[key]
    
    def __setitem__(self, key, value):
        self.features[key] = value
