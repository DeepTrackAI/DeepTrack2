'''
Contains tools for randomizing values. This randomization is 
performed using the Distribution class, which samples random
values according to a ruleset defined as its input.

The file also contains standard implementations of rulesets,
such as random_uniform.
'''

import numpy as np
from utils import isiterable, hasfunction



class Distribution:

    '''
    The Distribution class wraps an input, which is treated
    internally as a sampling rule. It uses this sampling rule
    to update and store an value sampled from this ruleset. 
    The ruleset can be virtually anything, from classes, to
    numbers, lists and dictionaries.

    Concretely, there are two methods that interract with the
    ruleset: __sample__ and __update__:

    __sample__() evaluates the sampling rule, and outputs a single
    value. 
    __update__() calls __sample__(), and sets the current_value 
    field as its output.

    Example:
        D = Disitribution(1)
        D.current_value => 1
        D.__update__([])
        D.current_value => 1

        D = Distribution([1,2])
        D.current_value => 2 # Either 2 or 1 randomly
        D.current_value => 2 # Same as last call
        D.__update__([])
        D.current_value => 1 # Either 2 or 1 randomly

    Inputs:
        sampling_rule           defines the output space of the 
                                distribution. See __sample__
                                for specifications.                 
    '''

    # Constructor
    def __init__(self, sampling_rule):
        self.sampling_rule = sampling_rule
    

    '''
        current_value is the result of the latest __update__
        call. Allows consistent access to a random parameter.

        The getter function is overridden to update itself 
        once if current_value has not yet been set.
    '''
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

    
    

    '''
        The __update__ function samples the sampling rule
        and sets the current_value property as the output.
        It takes a history parameter as an input, which 
        helps avoiding multiple updates during recursive
        calls. 

        Inputs:
            history:        A list of objects that has been 
                            updated.

        Outputs:
            self:           Returns itself.
                            
    '''
    def __update__(self, history):
        if self not in history:
            history.append(self)
            self.current_value = self.__sample__()
        return self

    
    def __sample__(self):
        # TODO: if else + help functions
        
        sampling_rule = self.sampling_rule


        if hasfunction(sampling_rule, "__sample__"):
            # If the ruleset itself implements a __sample__ function,
            # call it instead.
            return sampling_rule.__sample__()

        elif isinstance(sampling_rule, dict):
            # If the ruleset is a dict, return a new dict with each
            # element being sampled from the original dict.
            out = {}
            for key, val in self.sampling_rule.items():
                if hasfunction(val, '__sample__'):
                    out[key] = val.__sample__()
                else:
                    out[key] = val
            return out
        
        elif isiterable(sampling_rule):
            # If it's iterable, return the next value
            return next(sampling_rule)
            
        elif callable(sampling_rule):
            # If it's a function, call it without parameters
            return sampling_rule()
        
        elif (isinstance(sampling_rule, list) or
              isinstance(sampling_rule, np.ndarray) and sampling_rule.ndim == 1):
            # If it's either a list or a 1-dimensional ndarray,
            # return a random element from the list. 
            return np.random.choice(sampling_rule)
        
        else:
            # Else, assume it's elementary.
            return self.sampling_rule

'''
    Standard callable rulesets.

    Below are a number of standard functions that returns a lambda function with no input argument.
    These are all valid input arguments to the distribution class.
'''

def random_uniform(scale):
    '''
    Returns a ndarray of the same shape as the input argument, with each number uniformly distributed between 0 and scale

    Input arguments:
        scale:          The maximum of the distribution for each output. (array-like)
    '''
    def distribution():
        return np.random.rand(len(scale)) * np.array(scale)
    return distribution