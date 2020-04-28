''' Features that augment images

Augmentations are features that can resolve more than one image without
calling `.resolve()` on the parent feature. Specifically, they create
`updates_per_reload` images, while calling their parent feature
`load_size` times.

Classes
-------
Augmentation
    Base abstract augmentation class.
PreLoad
    Simple storage with no augmentation.
FlipLR
    Flips images left-right.
FlipUD
    Flips images up-down.
FlipDiagonal
    Flips images diagonally.
'''

from deeptrack.features import Feature
from deeptrack.image import Image
import numpy as np
from typing import Callable



class Augmentation(Feature):
    '''Base abstract augmentation class.

    Augmentations are features that can resolve more than one image without
    calling `.resolve()` on the parent feature. Specifically, they create
    `updates_per_reload` images, while calling their parent feature
    `load_size` times. They achieve this by resolving `load_size` results
    from the parent feature at once, and randomly drawing one of these 
    results as input to the method `.get()`. A new input is chosen
    every time `.update()` is called. Once `.update()` has been called
    `updated_per_reload` times, a new batch of `load_size` results are
    resolved from the parent feature.

    The method `.get()` of implementations of this class may accept the
    property `number_of_updates` as an argument. This number represents
    the number of times the `.update()` method has been called since the
    last time the parent feature was resolved.

    Parameters
    ----------
    feature : Feature
        The parent feature.
    load_size : int
        Number of results to resolve from the parent feature.
    updates_per_reload : int
        Number of times `.update()` is called before resolving new results
        from the parent feature.
    update_properties : Callable or None
        Function called on the output of the method `.get()`. Overrides
        the default behaviour, allowing full control over how to update
        the properties of the output to account for the augmentation.
    '''

    __distributed__ = False
    def __init__(self, 
                 feature: Feature, 
                 load_size: int = 1, 
                 updates_per_reload: int = np.inf, 
                 update_properties: Callable or None = None, 
                 **kwargs):
        self.feature = feature 

        def get_preloaded_results(load_size, number_of_updates):
            # Dummy property that loads results from the parent when
            # number of properties=0
            if number_of_updates == 0:
                self.preloaded_results = self._load(load_size)

            return None
        
        def get_number_of_updates(updates_per_reload=1):
            # Updates the number of updates. The very first update is not counted.
            if not hasattr(self.properties["number_of_updates"], "_current_value"):
                return 0
            return (self.properties["number_of_updates"].current_value + 1) % updates_per_reload

        if not update_properties:
            update_properties = self.update_properties
                
        super().__init__(
            load_size=load_size, 
            updates_per_reload=updates_per_reload, 
            index=lambda load_size: np.random.randint(load_size), 
            number_of_updates=get_number_of_updates,
            preloaded_results=get_preloaded_results,
            update_properties=lambda: update_properties,
            **kwargs)


    def _process_and_get(self, *args, update_properties=None, index=0, **kwargs):
        # Loads a result from storage
        image_list_of_lists = self.preloaded_results[index]
        if not isinstance(image_list_of_lists, list):
            image_list_of_lists = [image_list_of_lists]
        
        new_list_of_lists = []
        # Calls get
        for image_list in image_list_of_lists:
            if isinstance(image_list, list):
                new_list_of_lists.append([
                    [self.get(Image(image), **kwargs).merge_properties_from(image) for image in image_list]
                ])
            else: 
                new_list_of_lists.append(
                    self.get(Image(image_list), **kwargs).merge_properties_from(image_list)
                )
            
        if update_properties:
            for image_list in new_list_of_lists:
                if not isinstance(new_list_of_lists, list):
                    image_list = [image_list]
                for image in image_list:
                    image.properties = [dict(prop) for prop in image.properties]
                    update_properties(image, **kwargs)

        return new_list_of_lists
    

    def _load(self, load_size):
        # Resolves parent and stores result
        preloaded_results = []
        for _ in range(load_size):
            if isinstance(self.feature, list):  
                [_feature.update() for _feature in self.feature]
                list_of_results = [_feature.resolve(_augmentation_index=index) for index, _feature in enumerate(self.feature)]
                preloaded_results.append(list_of_results)
            else:
                self.feature.update()
                result = self.feature.resolve(_augmentation_index=0)
                preloaded_results.append(result)
        return preloaded_results

    def update_properties(*args, **kwargs):
        pass



class PreLoad(Augmentation):
    '''Simple storage with no augmentation.

    '''

    def get(self, image, **kwargs):
        return image


class Crop(Augmentation):
    ''' Crops a regions of an image.

    Parameters
    ----------
    feature : feature or list of features
        Feature(s) to augment.
    corner : tuple of ints or Callable[Image] or "random"
        Top left corner of the cropped region. Can be a tuple of ints,

    '''
    def __init__(self, *args, corner="random", crop_size=(64, 64), **kwargs):
        super().__init__(*args, corner=corner, crop_size=crop_size, **kwargs)
    
    def get(self, image, corner, crop_size, **kwargs):
        if corner == "random":
            slice_start = np.random.randint([0] * len(crop_size), np.array(image.shape[:len(crop_size)]) - crop_size)
            
        elif callable(corner):
            slice_start = corner(image)

        else:
            slice_start = corner
    
        slices = tuple([slice(slice_start_i, slice_start_i + crop_size_i) for slice_start_i, crop_size_i in zip(slice_start, crop_size)])

        cropped_image = image[slices]

        for prop in cropped_image.properties:
            if "position" in prop:
                position = np.array(prop["position"])
                try:
                    position[0:2] -= corner[0:2]
                    prop["position"] = position
                except IndexError:
                    pass
        
        return cropped_image

class FlipLR(Augmentation):
    ''' Flips images left-right.

    Updates all properties called "position" to flip the second index.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_size=1, updates_per_reload=2, **kwargs)


    def get(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2:
            image = np.fliplr(image)
        return image


    def update_properties(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2: 
            for prop in image.properties:
                if "position" in prop:
                    position = prop["position"]
                    new_position = (position[0], image.shape[1] - position[1], *position[2:])
                    prop["position"] = new_position



class FlipUD(Augmentation):
    ''' Flips images up-down.

    Updates all properties called "position" by flipping the first index.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_size=1, updates_per_reload=2, **kwargs)


    def get(self, image, number_of_updates=0, **kwargs):
        if number_of_updates % 2:
            image = np.flipud(image)
        return image


    def update_properties(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2: 
            for prop in image.properties:
                if "position" in prop:
                    position = prop["position"]
                    new_position = (image.shape[0] - position[0], *position[1:])
                    prop["position"] = new_position


class FlipDiagonal(Augmentation):
    ''' Flips images along the main diagonal.

    Updates all properties called "position" by swapping the first and second index.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, load_size=1, updates_per_reload=2, **kwargs)


    def get(self, image, number_of_updates, axes=(1, 0, 2), **kwargs):
        if number_of_updates % 2:
            image = np.transpose(image, axes=axes)
        return image


    def update_properties(self, image, number_of_updates, **kwargs):
        if number_of_updates % 2: 
            for prop in image.properties:
                if "position" in prop:
                    position = prop["position"]
                    new_position = (position[1], position[0], *position[2:])
                    prop["position"] = new_position
