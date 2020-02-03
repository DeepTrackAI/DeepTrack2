from deeptrack.features import Feature, MERGE_STRATEGY_APPEND, MERGE_STRATEGY_OVERRIDE
from deeptrack.image import Image
from copy import deepcopy
import numpy as np
from scipy import ndimage



class Augmentation(Feature):
    __list_merge_strategy__ = MERGE_STRATEGY_APPEND
    def __init__(self, *args, keep_both=True, update_properties=self.update_properties **kwargs):

        if not keep_both:
            self.__list_merge_strategy__ = MERGE_STRATEGY_OVERRIDE
        
        super().__init__(*args, keep_both=keep_both, update_properties=update_properties, **kwargs)
        
        def _process_and_get(self, image, update_properties=None, **kwargs):
            new_list = super()._process_and_get(image, **kwargs)
            
            if update_properties is None:
                return new_list

            if not isinstance(new_list, list):
                new_list = [new_list]
            
            for image in new_list:
                image.properties = update_properties(image)

            return new_list

    def update_properties(self, image, **kwargs):
        pass   


class FlipLR(Augmentation):
    def get(self, image, **kwargs):
        return np.fliplr(image)

    def update_properties(self, image, **kwargs):
        for prop in image.properties:
            if "position" in prop:
                position = prop["position"]
                new_position = (position[0], image.shape[1] - position[1], *position[2:])
                prop["position"] = type(position)(new_position)


class FlipUD(Augmentation):
    def get(self, image, **kwargs):
        return np.flipud(image)

    def update_properties(self, image, **kwargs):
        for prop in image.properties:
            if "position" in prop:
                position = prop["position"]
                new_position = (image.shape[0] - position[0], *position[1:])
                prop["position"] = type(position)(new_position)


class FlipDiagonal(Augmentation):
    def get(self, image, axes=(1, 0), **kwargs):
        return np.transpose(image, axes=axes)

    def update_properties(self, image, **kwargs):
        for prop in image.properties:
            if "position" in prop:
                position = prop["position"]
                new_position = (image.shape[0] - position[0], *position[1:])
                prop["position"] = type(position)(new_position)


class Rotate(Augmentation):
    def __init__(self, *args, angle=0, axes=(1, 0), reshape=False, order=3, mode="constant", cval=0.0, prefilter=True, **kwargs):
        super().__init__(*args, angle=angle, axes=axis, reshape=reshape, order=order, mode=mode, cval=cval, prefilter=prefilter, **kwargs)
        
    def get(self, image, angle=None, axes=(1, 0), reshape=None, order=None, mode=None, cval=None, prefilter=None **kwargs):

        new_image = Image(ndimage.rotate(image, angle, axes=axes reshape=reshape, order=order, mode=mode, cval=cval, prefilter=prefilter))
        new_image.properties = image.properties
        
        return new_image
    
    def update_properties(self, image, angle=0, **kwargs):
        for prop in image.properties:
            if "position" in prop:
                position = prop["position"]
                new_position = (np.cos(angle) * position[0] + np.sin(angle) * position[1],
                               -np.sin(angle) * position[0] + np.cos(angle) * position[1],
                               *position[2:])
                prop["position"] = type(position)(new_position)


class PreLoad(Augmentation):
    def __init__(self, feature=None, load_size=1, updates_per_reload=None, **kwargs):
        self.feature = feature 
        self.number_of_updates = 0
        self.load()
        
        def get_index():
            return np.random.randint(len(self.preloaded_results))

        super().__init__(load_size=load_size, updates_per_reload=updates_per_reload, index=get_index, **kwargs)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

        self.number_of_updates += 1
        if self.number_of_updates >= self.properties["updates_per_reload"].current_value:
            self.load()
            self.number_of_updates = 0

    def load(self):
        preloaded_results = []
        for _ in range(self.properties["load_size"].current_value):
            self.feature.update()
            preloaded_results.append(self.feature.resolve())
        self.preloaded_results = preloaded_results

    def get(self, index=0):
        return self.preloaded_results[index]

        

        
    