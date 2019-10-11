from DeepTrack.Backend.Image import Feature
from DeepTrack.Backend.Distributions import Distribution
import os
import re
import numpy as np

class Load(Feature):
    __name__ = "Load"
    def __init__(self,
                    path):
        self.path = path
        self.__properties__ = {"path": path}

        # Initiates the iterator
        self.iter = next(self)
    
    def get(self, shape, image, **kwargs):
        return self.res
    
    def __update__(self,history):
        if self not in history:
            history.append(self)
            self.res = next(self.iter)
            super().__update__(history)
    
    def __next__(self):
        while True:
            file = np.random.choice(self.get_files())
            image = np.load(file)
            np.random.shuffle(image)
            for i in range(len(image)):
                yield image[i]

        


    def setParent(self, F):
        raise Exception("The Load class cannot have a parent. For literal addition, use the Add class")

    def get_files(self):
        if os.path.isdir(self.path):
             return [os.path.join(self.path,file) for file in os.listdir(self.path) if os.path.isfile(os.path.join(self.path,file))]
        else:
            dirname = os.path.dirname(self.path)
            files =  os.listdir(dirname)
            pattern = os.path.basename(self.path)
            return [os.path.join(self.path,file) for file in files if os.path.isfile(os.path.join(self.path,file)) and re.match(pattern,file)]
        
