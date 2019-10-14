import numpy as np

'''
Returns a ndarray of the same shape as the input argument, with each number uniformly distributed between 0 and scale

Input arguments:
    scale:          The maximum of the distribution for each output.
'''

def uniform_random(scale):
    def distribution():
        return np.random.rand(len(scale))*np.array(scale)
    return distribution

class Distribution:
    def __init__(self, D):
        self.D = D
    
    def __update__(self, history):
        if self not in history:
            history.append(self)
            self.value = sample(self)
        return self
    

    def __sample__(self):
        try:
            return self.D.__sample__()
        except AttributeError:
            pass
        
        if isinstance(self.D, dict):
            out = {}
            for key, val in self.D.items():
                out[key] = sample(val)
            return out
        
        try:
            return next(self.D)
        except TypeError:
            pass
            
        if callable(self.D):
            return self.D()
        
        # Else, check if input is an array, and extract a single element
        if isinstance(self.D, (list, np.ndarray)):
            return sample(np.random.choice(self.D))

        # Else, assume it's elementary.
        return self.D


    @property
    def value(self):
         self._value

    @value.setter
    def value(self, v):
        self._value = v
    
    @value.getter
    def value(self):
        if not hasattr(self, "_value"):
            sample(self)
        return self._value





'''
    Takes an input and recursively extracts a single elementary type.

    For callable elements, the output of the call is drawn

    For lists or ndarrays, a single element is drawn

    If the input is neither callable, a list, nor an ndarray, return the element.

    To return multiple values, store them as a tuple. 
'''
def sample(E):
    try:
        return E.__sample__()
    except AttributeError:
        return E
    

