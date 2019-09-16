from abc import ABC, abstractmethod
from DeepTrack.Backend.Distributions import draw
import numpy as np

class Tree:
    def __init__(self):
        self.Tree = []

    def __add__(self, other):
        self.Tree.append(other)
        return self

    def resolve(self, Optics):
        image = np.zeros(Optics.shape)
        
        properties = []
        for branch in self.Tree:
            image, props = branch(image, Optics)
            properties.append(props)
        return image, props
    


class Output(ABC):
    def __add__(self,other):
        T = Tree()
        T = T + self + other
        return T

    def __call__(self, Image, Optics):
        return self.get(Image, Optics)

    @abstractmethod
    def get(self, Image, Optics):
        pass
