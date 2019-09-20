from abc import ABC, abstractmethod
import os
import numpy as np

"Interface for Callbacks. Implementations need to override the __call__ method, taking the Generator class and a list of Images"

class Callback(ABC):
    
    @abstractmethod
    def __call__(self, G, Images):
        pass

class Storage(Callback):

    def __init__(self,
                    filepath,
                    overwrite=True,
                    max_files=np.inf,
                    ):
        self.filepath = filepath
        self.max_files = max_files
        self.overwrite = overwrite

    def __call__(self, G, Images):
        full_path = self.filepath

        base_dir = os.path.dirname(full_path)

        files = os.listdir(base_dir)
        for file in reversed(files):
            file = os.path.join(base_dir,file)
            if not os.path.isfile(file):
                del file
        
        if len(files) > self.max_files:
            return
        idd = 0
        while os.path.isfile(full_path) and not self.overwrite:
            full_path = self.modify(full_path, num=idd)
            idd += 1

        np.save(full_path, Images)


    def modify(self, path, num=0):
        return path + "_" + str(num)


        