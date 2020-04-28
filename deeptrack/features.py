'''Base class Feature and structural features

Provides classes and tools for creating and interacting with features.

Classes
-------
Feature
    Base abstract class.
StructuralFeature
    Abstract extension of feature for interactions between features.
Branch
    Implementation of `StructuralFeature` that resolves two features 
    sequentially.
Probability
    Implementation of `StructuralFeature` that randomly resolves a feature 
    with a certain probability.
Duplicate
    Implementation of `StructuralFeature` that sequentially resolves an 
    integer number of deep-copies of a feature.

'''

import copy

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from deeptrack.image import Image
from deeptrack.properties import Property, PropertyDict



MERGE_STRATEGY_OVERRIDE = 0
MERGE_STRATEGY_APPEND = 1


class Feature(ABC):
    ''' Base feature class.
    Features define the image generation process. All features operate
    on lists of images. Most features, such as noise, apply some
    tranformation to all images in the list. This transformation can
    be additive, such as adding some Gaussian noise or a background
    illumination, or non-additive, such as introducing Poisson noise
    or performing a low-pass filter. This transformation is defined
    by the method `get(image, **kwargs)`, which all implementations of
    the class `Feature` need to define.

    Whenever a Feature is initiated, all keyword arguments passed to the
    constructor will be wrapped as a Property, and stored in the
    `properties` field as a `PropertyDict`. When a Feature is resolved,
    the current value of each property is sent as input to the get method.

    Parameters
    ----------
    *args : dict, optional
        Dicts passed as nonkeyword arguments will be deconstructed to key-value 
        pairs and included in the field `properties` in the same way as keyword 
        arguments.
    **kwargs
        All Keyword arguments will be wrapped as instances of ``Property`` and 
        included in the field `properties`.


    Attributes
    ----------
    properties : dict
        A dict that contains all keyword arguments passed to the
        constructor wrapped as Distributions. A sampled copy of this
        dict is sent as input to the get function, and is appended
        to the properties field of the output image.
    __list_merge_strategy__ : int
        Controls how the output of `.get(image, **kwargs)` is merged with 
        the input list. It can be `MERGE_STRATEGY_OVERRIDE` (0, default),
        where the input is replaced by the new list, or
        `MERGE_STRATEGY_APPEND` (1), where the new list is appended to the
        end of the input list.
    __distributed__ : bool
        Controls whether `.get(image, **kwargs)` is called on each element
        in the list separately (`__distributed__ = True`), or if it is
        called on the list as a whole (`__distributed__ = False`).
    __property_memorability__
        Controls whether to store the features properties to the `Image`. 
        Values 1 or lower will be included by default.
    '''

    __list_merge_strategy__ = MERGE_STRATEGY_OVERRIDE
    __distributed__ = True
    __property_memorability__ = 1


    def __init__(self, *args: dict, **kwargs):
        super(Feature, self).__init__()
        properties = getattr(self, "properties", {})

        # Create an iterable of kwargs and args
        all_dicts = (kwargs, ) + args

        for property_dict in all_dicts:
            for key, value in property_dict.items():
                if not isinstance(value, Property):
                    value = Property(value)

                properties[key] = value

        # hash_key is an inexpensive way to compare dicts of properties
        # The hash here is 4 31 bit integers, for a total of 124 bits.
        if "hash_key" not in properties:
            properties["hash_key"] = Property(lambda: list(np.random.randint(2**31, size=(4, ))))

        self.properties = PropertyDict(**properties)


    @abstractmethod
    def get(self, image: Image or List[Image], **kwargs) -> Image or List[Image]:
        '''Method for altering an image
        Abstract method that define how the feature transforms the input. The current
        value of all properties will be passed as keyword arguments.

        Parameters
        ---------
        image : Image or List[Image]
            The Image or list of images to transform
        **kwargs
            The current value of all properties in `properties` as well as any global
            arguments.

        Returns
        -------
        Image or List[Image]
            The transformed image or list of images
        '''

    def resolve(self,
                image_list: Image or List[Image] = None,
                **global_kwargs):
        ''' Creates the image.
        Transforms the input image by calling the method `get()` with the
        correct inputs. The properties of the feature can be overruled by
        passing a different value as a keyword argument.

        Parameters
        ----------
        image_list : Image or List[Image], optional
            The Image or list of images to be transformed.
        **global_kwargs
            Set of arguments that are applied globally. That is, every
            feature in the set of features required to resolve an image
            will receive these keyword arguments. 

        Returns
        -------
        Image or List[Image]
            The resolved image
        '''
        
        # Remove hash_key from globals.
        if "hash_key" in global_kwargs:
            global_kwargs.pop("hash_key")

        # Ensure that input is a list
        image_list = self._format_input(image_list, **global_kwargs)

        # Get the input arguments to the method .get()
        feature_input = self.properties.current_value_dict(is_resolving=True, **global_kwargs)
        
        # Add global_kwargs to input arguments
        feature_input.update(global_kwargs)
        
        # Call the _process_properties hook, default does nothing.
        # Can be used to ensure properties are formatted correctly
        # or to rescale properties.
        feature_input = self._process_properties(feature_input)

        # _process_and_get calls the get function correctly according
        # to the __distributed__ attribute
        new_list = self._process_and_get(image_list, **feature_input)
 
        # Add feature_input to the image the class attribute __property_memorability__
        # is not larger than the passed property_verbosity keyword
        property_verbosity = global_kwargs.get("property_memorability", 1)
        feature_input["name"] = type(self).__name__
        if self.__property_memorability__ <= property_verbosity:
            for image in new_list:
                    image.append(feature_input)

        # Merge input and new_list
        if self.__list_merge_strategy__ == MERGE_STRATEGY_OVERRIDE:
            image_list = new_list
        elif self.__list_merge_strategy__ == MERGE_STRATEGY_APPEND:
            image_list = image_list + new_list

        # For convencience, list images of length one are unwrapped.
        if len(image_list) == 1:
            return image_list[0]
        else:
            return image_list


    def update(self, **kwargs) -> "Feature":
        '''Updates the state of all properties.

        Parameters
        ----------
        **kwargs
            Arguments that will be passed to the Property
            method `update()`.

        Returns
        -------
        self
        '''
        self.properties.update(**kwargs)
        return self


    def plot(self,
             input_image: Image or List[Image] = None,
             resolve_kwargs: dict = None,
             interval: float = None,
             **kwargs):
        ''' Visualizes the output of the feature
        Resolves the feature and visualizes the result. If the output is an Image,
        show it using `pyplot.imshow`. If the output is a list, create an `Animation`.
        For notebooks, the animation is played inline using `to_jshtml()`. For scripts,
        the animation is played using the matplotlib backend.

        Any parameters in kwargs will be passed to `pyplot.imshow`.

        Parameters
        ----------
        input_image : Image or List[Image], optional
            Passed as argument to `resolve` call
        resolve_kwargs : dict, optional
            Passed as kwarg arguments to `resolve` call
        interval : float
            The time between frames in animation in ms. Default 33.
        kwargs
            keyword arguments passed to the method pyplot.imshow()
        '''
        
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from IPython.display import HTML, display

        if input_image is not None:
            input_image = [Image(input_image)]

        output_image = self.resolve(input_image, **(resolve_kwargs or {}))

        # If a list, assume video
        if isinstance(output_image, Image):
            # Single image
            plt.imshow(output_image[:, :, 0], **kwargs)
            plt.show()

        else:
            # Assume video
            fig = plt.figure()
            images = []
            for image in output_image:
                images.append([plt.imshow(image[:, :, 0], **kwargs)])

            interval = (interval
                        or output_image[0].get_property("interval") 
                        or (1 / 30 * 1000))


            anim = animation.ArtistAnimation(fig, images, interval=interval, blit=True,
                                    repeat_delay=0)

            try: 
                get_ipython # Throws NameError if not in Notebook
                display(HTML(anim.to_jshtml()))

            except NameError as e:
                # Not in an notebook
                plt.show()

            except RuntimeError as e:
                # In notebook, but animation failed
                import ipywidgets as widgets
                Warning("Javascript animation failed. This is a non-performant fallback.")
                def plotter(frame=0):
                    plt.imshow(output_image[frame][:, :, 0], **kwargs)
                    plt.show()

                return widgets.interact(plotter, frame=widgets.IntSlider(value=0, min=0, max=len(images)-1, step=1))


    def _process_and_get(self, image_list, **feature_input) -> List[Image]:
        # Controls how the get function is called
        
        if self.__distributed__:
            # Call get on each image in list, and merge properties from corresponding image
            return [Image(self.get(image, **feature_input)).merge_properties_from(image) for image in image_list]
        else:
            # Call get on entire list.
            new_list = self.get(image_list, **feature_input)
        
            if not isinstance(new_list, list):
                new_list = [Image(new_list)]
            
            return new_list
        

    def _format_input(self, image_list, **kwargs) -> List[Image]:
        # Ensures the input is a list of Image.
        
        if image_list is None:
            return []

        if not isinstance(image_list, list):
            image_list = [image_list]
            
        return [Image(image) for image in image_list]
            

    def _process_properties(self, propertydict) -> dict:
        # Optional hook for subclasses to preprocess input before calling
        # the method .get()
        return propertydict


    def sample(self, **kwargs) -> "Feature":
        '''Returns the feature'''
        
        return self

    def __getattr__(self, key):
        # Allows easier access to properties, while guaranteeing they are updated correctly.
        # Should only every be used from the inside of a property function. 
        # Is not compatible with sequential properties.
        try: 
            return super().__getattr__(key)
        except AttributeError:
            try:
                properties = self.__dict__["properties"]
                if key in properties:
                    properties.update()
                    return properties[key].current_value
                else:
                    raise AttributeError

            except KeyError:
                raise AttributeError
        


    def __add__(self, other: "Feature") -> "Feature":
        # Overrides add operator
        
        return Branch(self, other)

    def __radd__(self, other) -> "Feature":
        # Add when left hand is not a feature
        # If left hand is falesly, return self
        # This allows operations such as sum(list_of_features)
        
        if not other:
            return self
        else:
            return NotImplemented


    def __mul__(self, other: float) -> "Feature":
        # Introduces a probablity of a feature to be resolved.
        
        return Probability(self, other)

    __rmul__ = __mul__


    def __pow__(self, other) -> "Feature":
        # Duplicate the feature to resolve more items
        
        return Duplicate(self, other)




class StructuralFeature(Feature):
    ''' Provides the structure of a feature-set
    Feature with __property_verbosity__ = 2 to avoid adding it to the list
    of properties, and __distributed__ = False to pass the input as-is. 
    '''
    
    __property_verbosity__ = 2
    __distributed__ = False



class Branch(StructuralFeature):
    ''' Resolves to features sequentially
    Parameters
    ----------
    feature_1 : Feature
    feature_2 : Feature
    '''

    def __init__(self, feature_1: Feature, feature_2: Feature, *args, **kwargs):
        super().__init__(*args, feature_1=feature_1, feature_2=feature_2, **kwargs)


    def get(self, image, feature_1, feature_2, **kwargs):
        ''' Resolves `feature_1` and `feature_2` sequentially
        '''
        image = feature_1.resolve(image, **kwargs)
        image = feature_2.resolve(image, **kwargs)
        return image


class Probability(StructuralFeature):
    ''' Resolves a feature with a certain probability

    Parameters
    ----------
    feature : Feature
        Feature to resolve
    probability : float
        Probability to resolve
    '''

    def __init__(self, feature: Feature, probability: float, *args, **kwargs):
        super().__init__(
            *args,
            feature=feature,
            probability=probability,
            random_number=np.random.rand,
            **kwargs)


    def get(self, 
            image, 
            feature: Feature,
            probability: float,
            random_number: float,
            **kwargs):
        ''' Resolves `feature` if `random_number` is less than `probability`
        '''
        if random_number < probability:
            image = feature.resolve(image, **kwargs)

        return image



class Duplicate(StructuralFeature):
    '''Resolves copies of a feature sequentially
    Creates `num_duplicates` copies of the feature and resolves
    them sequentially

    Parameters
    ----------
    feature: Feature
        The feature to duplicate
    num_duplicates: int
        The number of duplicates to create
    '''

    def __init__(self, feature: Feature, num_duplicates: int, *args, **kwargs):

        self.feature = feature
        super().__init__(
            *args,
            num_duplicates=num_duplicates, #py > 3.6 dicts are ordered by insert time.
            features=lambda num_duplicates: [copy.deepcopy(feature) for _ in range(num_duplicates)],
            **kwargs)


    def get(self, image, features: List[Feature], **kwargs):
        ''' Resolves each feature in `features` sequentially
        '''
        for index in range(len(features)):
            image = features[index].resolve(image, **kwargs)

        return image

    def update(self, **kwargs):
        super().update(**kwargs)

        features = self.properties["features"].current_value
        for index in range(len(features)):
            features[index].update(**kwargs)



class ConditionalSetProperty(StructuralFeature):
    ''' Conditionally overrides the properties of child features
    
    Parameters
    ----------
    feature : Feature
        The child feature
    condition : str
        The name of the conditional property
    **kwargs
        Properties to be used if `condition` is True

    '''
    __distributed__ = False
    def __init__(self, feature: Feature, condition="is_label", **kwargs):
        super().__init__(feature=feature, condition=condition, **kwargs)
    

    def get(self, image, feature, condition, **kwargs):
        if kwargs.get(condition, False):
            return feature.resolve(image, **kwargs)
        else:
            for property_key in self.properties.keys():
                kwargs.pop(property_key, None)
            
            return feature.resolve(image, **kwargs)



class ConditionalSetFeature(StructuralFeature):
    ''' Conditionally resolves one of two features
    
    Parameters
    ----------
    on_false : Feature
        Feature to resolve if the conditional property is false
    on_true : Feature
        Feature to resolve if the conditional property is true
    condition : str
        The name of the conditional property

    '''
    __distributed__ = False
    def __init__(self, on_false: Feature = None, on_true: Feature = None,  condition="is_label", **kwargs):
        super().__init__(on_false=on_false, on_true=on_true, condition=condition, **kwargs)
    

    def get(self, image, *, on_false, on_true, condition, **kwargs):
        if kwargs.get(condition, False):
            if on_true:
                return on_true.resolve(image, **kwargs)
            else:
                return image
        else:
            if on_false:
                return on_false.resolve(image, **kwargs)
            else:
                return image



class Label(Feature):
    '''Outputs the properties of this features.

    Parameters
    ----------
    output_shape : tuple of ints
        Reshapes the oiutput to this shape
    
    '''
    __distributed__ = False
    def get(self, image, output_shape=None, hash_key=None, **kwargs):
        result = []
        for key, value in kwargs.items():
            result.append(value)

        if output_shape:
            result = np.reshape(np.array(result), output_shape)

        return np.array(result)