from abc import ABC, abstractmethod
from DeepTrack.Backend.Distributions import draw
import numpy as np
'''
Make a subclass of ndarray
'''
class Image(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, properties=[]):

        obj = super(Image, subtype).__new__(subtype, shape, dtype,
                                                buffer, offset, strides,
                                                order)
        obj.properties = properties

        return obj

    def append(self, properties):
        self.properties.append(properties)
    
    def __array_finalize__(self, obj):

        if obj is None: return
        
        self.properties = getattr(obj, "properties", [])



class FeatureMap(ABC):
    def __init__(self):
        self.Tree = []

    def __add__(self, other):
        if isinstance(other, tuple):
            self.Tree.append(other)
        else:
            self.Tree.append((other,1))
        return self

    def __mul__(self, other):
        T = FeatureMap()
        T.Tree = [(self, other)]
        return T

    def __call__(self, image, Optics):
        return self.resolve(Optics, image=image)

    def resolve(self, Optics, image=None):
        if image is None:
            image = Image(Optics.shape)
            image[:] = 0
        
        for branch in self.Tree:
            if np.random.rand() <= branch[1]:
                image = branch[0](image, Optics)
        return image
    


class Feature(ABC):

    def __add__(self,other):
        T = FeatureMap()
        T = T + self + other
        return T

    def __mul__(self, other):
        return (self, other)
    
    def __call__(self, Image, Optics):

        Image, props = self.get(Image, Optics)

        Image.append(props)

        return Image

    @abstractmethod
    def get(self, Image, Optics):
        pass

    
    

    
    
    __radd__ = __add__
    __rmul__ = __mul__