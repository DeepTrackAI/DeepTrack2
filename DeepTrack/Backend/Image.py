from abc import ABC, abstractmethod
from DeepTrack.Backend.Distributions import draw
import numpy as np
import copy
'''
Make a subclass of ndarray
'''
class Image(np.ndarray):
    __array_priority__ = 2
    def __new__(cls, input_array, properties=None):

        obj = np.asarray(input_array).view(cls)
        if properties is None:
            properties = []

        obj.properties = properties
        
        return obj

    def append(self, properties):
        self.properties.append(properties)
    
    def __array_wrap__(self, out_arr, context=None):
        if out_arr is self:  # for in-place operations
            result = out_arr
        else:
            result = Image(out_arr)

        if context is not None:
            func, args, _ = context
            input_args = args[:func.nin]
            
            for arg in input_args:
                
                props = getattr(arg, "properties", [])
                for p in props:
                    result.append(p)
        return result


    def __array_finalize__(self, obj):

        if obj is None: return

        self.properties = getattr(self, "properties", [])

        props = getattr(obj, "properties", [])
        for property in props:
            self.append(property) 

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Image, self).__reduce__()
        # Create our own tuple to pass to __setstate__, appending properties
        new_state = pickled_state[2] + (self.properties,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.properties = state[-1]  # Set the peroperties attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(Image, self).__setstate__(state[0:-1])



# class FeatureMap(ABC):

#     def __call__(self, image, Optics):
#         return self.resolve(Optics, image=image)

#     def resolve(self, Optics, image=None):
#         if image is None:
#             image = Image(np.zeros(Optics.shape))
#             image[:] = 0
        
#         for branch in self.Tree:
#             if np.random.rand() <= branch[1]:
#                 image = branch[0](image, Optics)
#         return image
    
#     __rmul__ = __mul__
#     __radd__ = __add__
    
# # class Fork(FeatureMap):
# #     def __init__(self, root):

class Feature(ABC):

    def __add__(self,other):
        o_copy = copy.copy(other)
        o_copy.parent = self
        return o_copy

    def __radd__(self, other): 
        o_copy = copy.copy(other)
        self.parent = o_copy
        return o_copy

    def __mul__(self, other):
        o_copy = copy.copy(self)
        o_copy.probability = other
        return o_copy

    __rmul__ = __mul__

    '''
    Recursively resolves the feature feature tree backwards, starting at this node. 
    Each recursive step checks the content of __cache to check if the node has already 
    been calculated. This allows for a very efficient evaluation of more complex structures
    with several outputs.

    The function checks its parent property. For None values, the node is seen as input, 
    and creates a new image. For ndarrays and Images, those values are copied over. For
    Features, the image is calculated by recursivelt calling the __resolve__ method on the 
    parent.

    INPUTS:
        Optics: Optical system used to image the particle
    
    OUTPUTS:
        Image: An Image instance.
    '''
    def __resolve__(self, Optics):
        
        cache = getattr(self, "cache", None)
        if not cache is None:
            return cache

        parent = getattr(self, "parent", None)

        # If parent does not exist, initiate with zeros
        if parent is None:
            image = Image(np.zeros(Optics.shape))
        # If parent is ndarray, set as ndarray
        elif isinstance(parent, np.ndarray):
            image = Image(parent)
        # If parent is image, set as Image
        elif isinstance(parent, Image):
            image = parent
        # If parent is Feature, retrieve it
        elif isinstance(parent, Feature):
            image = parent.__resolve__(Optics)
        # else, pray
        else:
            image = parent
        # Get probability of draw
        p = getattr(self, "probability", 1)
        if np.random.rand() <= p:
            image, props = self.get(image, Optics)
            image.append(props)
        
        # Store to cache
        self.cache = copy.copy(image)
        
        return image

    '''
    Rcursively clears the __cache property. Should be on each output node between each call to __resolve__
    to ensure a correct initial state.
    '''
    def __clear__(self):
        self.cache = None
        parent = getattr(self, "parent", None)
        if isinstance(parent, Feature):
            parent.__clear__()

    @abstractmethod
    def get(self, Image, Optics):
        pass
    

class Label:
    def __init__(self, L, on_none=None, on_multiple=None):
        self.attr = ""
        if isinstance(L, Label):
            self = L
            return

        self.attr = L
        if on_none is not None:
            self.on_none = on_none

        if on_multiple is not None:
            self.on_multiple = on_multiple

    def on_none(self, props):
        return 0

    def on_multiple(self, found, props):
        return found[0]

    def __call__(self, properties):
        res = []
        for p in properties:
            a = p.get(self.attr,None)
            if a is not None:
                res.append(a)
        
        if len(res) == 0:
            return self.on_none(properties)

        if len(res) > 1:
            return self.on_multiple(res, properties)

        return res[0]
