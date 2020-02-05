from deeptrack.features import Feature, MERGE_STRATEGY_APPEND, MERGE_STRATEGY_OVERRIDE
from deeptrack.image import Image
import numpy as np
from scipy import ndimage



class Augmentation(Feature):
    __distributed__ = False
    def __init__(self, feature, load_size=1, updates_per_reload=np.inf, update_properties=None, **kwargs):
        self.feature = feature 

        def get_index():
            return np.random.randint(load_size)

        def get_preloaded_results(load_size=1, number_of_updates=1):
            if number_of_updates == 0:
                self.preloaded_results = self.load(load_size)
            
            return None
        
        def get_number_of_updates(updates_per_reload=1):
            if not hasattr(self.properties["number_of_updates"], "_current_value"):
                return 0
            return (self.properties["number_of_updates"].current_value + 1) % updates_per_reload

        if not update_properties:
            update_properties = self.update_properties
                
        super().__init__(
            load_size=load_size, 
            updates_per_reload=updates_per_reload, 
            index=get_index, 
            number_of_updates=get_number_of_updates,
            preloaded_results=get_preloaded_results,
            update_properties=lambda: update_properties,
            **kwargs)

        
    def _process_and_get(self, *args, update_properties=None, index=0, **kwargs):

        image = self.preloaded_results[index]

        new_image = self.get(image, **kwargs)
    
        if not isinstance(new_image, list):
            new_image = [new_image]

        if update_properties is None:
            return new_image
        
        for image in new_image:
            update_properties(image, **kwargs)
        return new_image


    def load(self, load_size):
        preloaded_results = []
        for _ in range(load_size):
            self.feature.update()
            preloaded_results.append(self.feature.resolve())
        return preloaded_results

    
    def update_properties(self, image, **kwargs):
        pass   


class PreLoad(Augmentation):
    def get(self, image, **kwargs):
        return image


class FlipLR(PreLoad):

    def __init__(self, feature, **kwargs):
        super().__init__(feature, load_size=1, updates_per_reload=2, **kwargs)

    def get(self, image, number_of_updates=0, **kwargs):
        if number_of_updates == 0:
            return image 
        else:
            return np.fliplr(image)

    def update_properties(self, image, number_of_updates=0, **kwargs):
        if number_of_updates == 0:
            return 

        for prop in image.properties:
            if "position" in prop:
                position = prop["position"]
                new_position = (position[0], image.shape[1] - position[1], *position[2:])
                prop["position"] = new_position


class FlipUD(PreLoad):

    def __init__(self, feature, **kwargs):
        super().__init__(feature, load_size=1, updates_per_reload=2, **kwargs)

    def get(self, image, number_of_updates=0, **kwargs):
        if number_of_updates == 0:
            return image 
        else:
            return np.flipud(image)

    def update_properties(self, image, number_of_updates=0, **kwargs):
        if number_of_updates == 0:
            return 

        for prop in image.properties:
            if "position" in prop:
                position = prop["position"]
                new_position = (image.shape[0] - position[0], *position[1:])
                prop["position"] = new_position


class FlipDiagonal(Augmentation):
    def __init__(self, feature, **kwargs):
        super().__init__(feature, load_size=1, updates_per_reload=2, **kwargs)

    def get(self, image, number_of_updates=0, axes=(1, 0, 2), **kwargs):
        if number_of_updates == 0:
            return image 
        else:
            return np.transpose(image, axes=axes)

    def update_properties(self, image, number_of_updates=0, **kwargs):
        if number_of_updates == 0:
            return 

    def update_properties(self, image, **kwargs):
        for prop in image.properties:
            if "position" in prop:
                position = prop["position"]
                new_position = (position[1], position[0], *position[2:])
                prop["position"] = new_position


class Rotate(Augmentation):
    def __init__(self, *args, angle=0, axes=(1, 0), reshape=False, order=3, mode="constant", cval=0.0, prefilter=True, **kwargs):
        super().__init__(*args, angle=angle, axes=axes, reshape=reshape, order=order, mode=mode, cval=cval, prefilter=prefilter, **kwargs)
        
    def get(self, image, angle=None, axes=(1, 0), reshape=None, order=None, mode=None, cval=None, prefilter=None, **kwargs):

        new_image = Image(ndimage.rotate(image, angle, axes=axes, reshape=reshape, order=order, mode=mode, cval=cval, prefilter=prefilter))
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





    

        

        
    