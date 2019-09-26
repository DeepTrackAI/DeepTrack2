from abc import ABC, abstractmethod
import os
import numpy as np
import pickle
import re

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
        
        if self.too_many_files(base_dir):
            return
        
        idd = 0
        new_path = full_path[:]
        while os.path.isfile(new_path) and not self.overwrite:
            new_path = self.modify(full_path, num=idd)
            idd += 1
        full_path = new_path
        # Step 2
        with open(full_path, 'wb') as f:
        # Step 3
            pickle.dump(Images, f)

    def too_many_files(self,dir):
        files = os.listdir(dir)
        for file in reversed(files):
            file = os.path.join(dir,file)
            if not os.path.isfile(file):
                del file
        return len(files) > self.max_files
        

    def modify(self, path, num=0):
        return path + "_" + str(num)

        
            
        