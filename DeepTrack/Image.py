from abc import ABC, abstractmethod
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