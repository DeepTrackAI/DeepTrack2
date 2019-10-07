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


class Feature(ABC):
    def __input_shape__(self, shape):
        return shape

    def __add__(self, other):
        o_copy = copy.copy(other)
        o_copy = o_copy.setParent(self)
        return o_copy

    def __radd__(self, other): 
        o_copy = copy.copy(other)
        self = self.setParent(o_copy)
        return self

    def __mul__(self, other):
        G = Group(copy.copy(self))
        G.probability = other
        return G

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
    def __resolve__(self, shape, **kwargs):

        cache = getattr(self, "cache", None)
        if not cache is None:
            return cache

        parent = getattr(self, "parent", None)
        # If parent does not exist, initiate with zeros
        if parent is None:
            image = Image(np.zeros(self.__input_shape__(shape)))
        # If parent is ndarray, set as ndarray
        elif isinstance(parent, np.ndarray):
            image = Image(parent)
        # If parent is image, set as Image
        elif isinstance(parent, Image):
            image = parent
        # If parent is Feature, retrieve it
        elif isinstance(parent, Feature):
            image = parent.__resolve__(shape, **kwargs)
        # else, pray
        else:
            image = parent
        
        # Get probability of draw
        p = getattr(self, "probability", 1)
        if np.random.rand() <= p:
            image, props = self.get(shape, image, **kwargs)
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

    def getRoot(self):
        if hasattr(self, "parent"):
            return self.parent.getRoot()
        else:
            return self

    def setParent(self, Feature):
        if hasattr(self, "parent"):
            G = Group(self)
            G = G.setParent(Feature)
            return G
        else:            
            self.parent = Feature
            return self

    @abstractmethod
    def get(self, Image, Optics=None):
        pass
    
class Group(Feature):
    def __init__(self, Features):
        self.group = Features

    def __input_shape__(self,shape):
        return self.group.__input_shape__(shape)

    def __clear__(self):
        # print("clearing " + str(type(self)))
        group = getattr(self, "group", None)
        if isinstance(group, Feature) and hasattr(group, "parent"):
            group.__clear__()
            

        self.cache = None
        parent = getattr(self, "parent", None)
        if isinstance(parent, Feature):
            parent.__clear__()
    

    def get(self, shape, Image, **kwargs):
        return self.group.__resolve__(shape, **kwargs), {}

    # TODO: What if already has parent? Possible?
    def setParent(self, Feature):
        self.parent = Feature
        self.group.getRoot().setParent(Feature)
        return self



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
