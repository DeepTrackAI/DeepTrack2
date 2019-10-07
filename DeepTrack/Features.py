from DeepTrack.Backend.Image import Feature
import os
import re
import numpy as np

class Load(Feature):
    def __init__(self,
                    path):
        self.path = path
        self.iter = next(self)

    def get(self, shape, image, **kwargs):
        return next(self.iter)
        
    
    def __next__(self):
        while True:
            file = np.random.choice(self.get_files())
            image = np.load(file)
            np.random.shuffle(image)
            for i in range(len(image)):
                yield image[i], {"type": "Load", "path": file, "index": i}

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

class Add(Feature):
    def __init__(self, F1, F2):
        self.F1 = F1
        self.F2 = F2
    
    def get(self, shape, image, **kwargs):
        I1 = self.F1.__shape__(shape, **kwargs)
        I2 = self.F2.__shape__(shape, **kwargs)
        return I1 + I2, {"type": "Add"}
        
