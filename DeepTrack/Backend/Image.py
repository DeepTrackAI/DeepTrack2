from abc import ABC, abstractmethod
from DeepTrack.Backend.Distributions import draw
import numpy as np

class Tree:
    def __init__(self):
        self.Tree = []

    def __add__(self, other):
        if isinstance(other, tuple):
            self.Tree.append(other)
        else:
            self.Tree.append((other,1))
        return self

    def resolve(self, Optics):
        image = np.zeros(Optics.shape)
        
        properties = []
        for branch in self.Tree:
            if np.random.rand() <= branch[1]:
                image, props = branch[0](image, Optics)
                properties.append(props)

        return image, properties
    


class Output(ABC):

    def __add__(self,other):
        T = Tree()
        T = T + self + other
        return T

    

    def __mul__(self, other):
        return (self, other)
    def __call__(self, Image, Optics):
        return self.get(Image, Optics)

    @abstractmethod
    def get(self, Image, Optics):
        pass
    
    __radd__ = __add__
    __rmul__ = __mul__