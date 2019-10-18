# TODO: overall description

import numpy as np

'''
Returns a ndarray of the same shape as the input argument, with each number uniformly distributed between 0 and scale

Input arguments:
    scale:          The maximum of the distribution for each output.
'''

class Distribution:
# TODO: description of class
    
    @property
    def current_value(self):
         self._current_value

    @current_value.setter
    def current_value(self, updated_current_value):
        self._current_value = updated_current_value
    
    @current_value.getter
    def current_value(self):
        if not hasattr(self, "_current_value"):
            self.__update__([]) # generate new current value
        return self._current_value

    
    def __init__(self, sampling_rule):
        self.sampling_rule = sampling_rule

    
    def __update__(self, history):
        if self not in history:
            history.append(self)
            self.current_value = self.__sample__()
        return self

    
    def __sample__(self):
        # TODO: if else + help functions
        
        try:
            return self.sampling_rule.__sample__()
        except AttributeError:
            pass
        
        if isinstance(self.sampling_rule, dict):
            out = {}
            for key, val in self.sampling_rule.items():
                out[key] = sample(val)
            return out
        
        try:
            return next(self.sampling_rule)
        except TypeError:
            pass
            
        if callable(self.sampling_rule):
            return self.sampling_rule()
        
        # Else, check if input is an array, and extract a single element
        if isinstance(self.sampling_rule, (list, np.ndarray)):
            return np.random.choice(self.sampling_rule)

        # Else, assume it's elementary.
        return self.sampling_rule


def random_uniform(scale):
    def distribution():
        return np.random.rand(len(scale)) * np.array(scale)
    return distribution