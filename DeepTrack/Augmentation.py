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
from DeepTrack.Backend.Image import Image
import copy
class Augmentation(ABC):
    __append__ = False
    def update_props(self, image):
        return image

    def __call__(self, Images):
        for i in range(len(Images)):
            I = Image(Images[i])
            I.properties = copy.deepcopy(Images[i].properties)
            I = self.augment(I)
            I = self.update_props(I)
            if type(self).__append__:
                Images.append(I)
            else:
                Images[i] = I
        return Images

            

    @abstractmethod
    def augment(self, image):
        pass



class NormalizeMinMax(Augmentation):
    def augment(self, image):
        image = image - np.amin(image)
        image = image / np.amax(image)
        image[np.isnan(image)] = 0
        return image

class NormalizeStandard(Augmentation):
    def augment(self, image):
        image = image - np.mean(image)
        image = image / np.std(image)
        image[np.isnan(image)] = 0
        return image

class FlipLR(Augmentation):
    __append__ = True
    def augment(self, image):
        image = np.fliplr(image)
        return image

    def update_props(self, Image):
        for i in range(len(Image.properties)):
            p = Image.properties[i]
            if "position" in p:
                Image.properties[i]["position"][0] = Image.shape[1] - p["position"][0] - 1
        return Image

class FlipUD(Augmentation):
    __append__ = True
    def augment(self, image):
        image = np.flipud(image)
        return image

    def update_props(self, Image):
        for i in range(len(Image.properties)):
            p = Image.properties[i]
            if "position" in p:
                Image.properties[i]["position"][1] = Image.shape[1] - p["position"][1] - 1
        return Image

class Transpose(Augmentation):
    __append__ = True
    def augment(self, image):
        image = np.transpose(image)
        return image

    def update_props(self, Image):
        for i in range(len(Image.properties)):
            p = Image.properties[i]
            if "position" in p:
                x = p["position"][0]
                y = p["position"][1]
                Image.properties[i]["position"][0] = y
                Image.properties[i]["position"][1] = x
        return Image