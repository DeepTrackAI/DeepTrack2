'''
Includes

NormalizeMinMax

NormalizeStandard

ROI

Flip (lr, ud)

Transpose (main-diag, inverted-diag)

Assumes first dimension is batch dimension.
    
'''

from abc import ABC, abstractmethod
import numpy as np
class Augmentation(ABC):
    
    def update_props(self):
        pass

    @abstractmethod
    def __call__(self, Image):
        pass

class NormalizeMinMax(Augmentation):
    def __call__(self, Images):
        for image in Images:
            image = image - np.amin(image)
            image = image / np.amax(image)
            image[np.isnan(image)] = 0
        return Images

class NormalizeStandard(Augmentation):
    def __call__(self, Images):
        for image in Images:
            image = image - np.mean(image)
            image = image / np.std(image)
            image[np.isnan(image)] = 0

class FlipLR(Augmentation):
    def __call__(self, Images):
        for i in range(len(Images)):
            image = Images[i]
            new_image = np.fliplr(image)
            self.update_props(new_image)
            Images.append()

    def update_props(self, Image):
        for p in Image.properties:
            if hasattr(p, "x"):
                p["x"] = Image.shape[0] - p["x"]

class FlipUD(Augmentation):
    def __call__(self, Images):
        for i in range(len(Images)):
            image = Images[i]
            new_image = np.flipup(image)
            self.update_props(new_image)

    def update_props(self, Image):
        for p in Image.properties:
            if hasattr(p, "y"):
                p["y"] = Image.shape[0] - p["y"]

class Transpose(Augmentation):
    def __call__(self, Images):
        for i in range(len(Images)):
            image = Images[i]
            new_image = np.transpose(image)
            self.update_props(new_image)

    def update_props(self, Image):
        for p in Image.properties:
            if hasattr(p, "x") and hasattr(p, "y"):
                x = p["x"]
                y = p["y"]
                p["x"] = y
                p["y"] = x