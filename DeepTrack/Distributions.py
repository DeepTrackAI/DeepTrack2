''' Feature Property Randomization.

Contains tools used to randomize the properties of a Feature. 
This randomization is typically performed using the Distribution class,
which samples random values according to a ruleset defined as its 
input. However, any object with a defined current_value field and a 
__update__(history) method are valid substitutions.

The file also contains standard implementations of rulesets,
such as random_uniform.

Classes
--------
Distribution
    Standard way to randomize properties. Updates the current_value field
    according to a sampling rule.

Methods
-------
random_uniform(scale : array-like)
    returns a function that, when called, returns a uniform random ndarray
    of the same shape as scale, with each element between 0 and scale.
'''

import numpy as np
from utils import isiterable, hasfunction



# CLASSES

class Distribution:

    '''Class for randomizing Features.
    
    The Distribution class wraps an input, which is treated
    internally as a sampling rule. It uses this sampling rule
    to update and store an value sampled from this ruleset. 
    The ruleset can be virtually anything, from classes, to
    numbers, lists and dictionaries.

    Attributes
    ----------
    sampling_rule : any
        A property used to define how the distribution is sampled.
    current_value : any
        The current value drawn from the distribution.
    
    Methods
    -------
    __sample__()
        Samples the distribution according to the sampling_rule and
        returns the result.
    __update__(history : list)
        Samples the distribution according to the sampling_rule and
        sets the current value as the result.
    
    Examples
    --------
    The distribution of a number will always return the number
    >>> np.random.seed(0)
    >>> D = Disitribution(1)
    >>> D.current_value
    1
    >>> D.__update__([])
    >>> D.current_value
    
    1

    The Distribution of a list will randomly return one of the 
    elements.
    >>> D = Distribution([1, 2, 3])
    >>> D.current_value => 2 # Either 1, 2 or 3 randomly
    2
    >>> D.current_value => 2 # Same as last call
    2
    >>> D.__update__([])
    >>> D.current_value
    1               
    '''

    # Constructor
    def __init__(self, sampling_rule : any):
        '''Constructor for the Distribution class.

        Parameters
        ----------
        sampling_rule : any        
            defines the output space of the distribution. See 
            __sample__ for how different types are sampled.
        '''
        self.sampling_rule = sampling_rule
    


    @property
    def current_value(self):
        '''
        current_value is the result of the latest __update__
        call. Allows consistent access to a random parameter.

        The getter function is overridden to update itself 
        once if current_value has not yet been set.
        '''
        self._current_value

    @current_value.setter
    def current_value(self, updated_current_value):
        self._current_value = updated_current_value
    
    @current_value.getter
    def current_value(self):
        if not hasattr(self, "_current_value"):
            self.__update__([]) # generate new current value
        return self._current_value

    

    def __update__(self, history:list):
        '''
        The __update__ function samples the sampling rule
        and sets the current_value property as the output.
        It takes a history parameter as an input, which 
        helps avoiding multiple updates during recursive
        calls. It also appends itself to the history.

        Parameters
        ------
        history  
            A list of objects that has been updated.

        '''

        if self not in history:
            history.append(self)
            self.current_value = self.__sample__()



    def __sample__(self):
        '''
            Returns a sampled instance of the sampling_rule field.
            The logic behind the sampling depends on the type of the
            sample_rule. These are checked in the following order of
            priority:

            1. Any object with a callable __sample__ method has this
                method called and returned.
            2. If the rule is a dict, the dict is copied and any value
                with has a callable __sample__ method is replaced with the
                output of that call.
            3. If the rule is an iterable, return the next output.
            4. If the rule is callable, return the output of a call with
                no input parameters.
            5. If the rule is a list or a 1-dimensional ndarray, return 
                a single element drawn from that list/ndarray.
            6. If none of the above apply, return the rule itself.

            Returns
            -------
            any
                A sampled instance of the sampling_rule. 
        '''
        
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



# FUNCTIONS
import types
# TODO: allow for min/max definition
def random_uniform(scale) -> types.FunctionType:
    '''
    Returns a ndarray of the same shape as the input argument, with each number uniformly distributed between 0 and scale

    Parameters:
        scale          
            The maximum of the distribution.
    '''
    def distribution():
        return np.random.rand(len(scale)) * np.array(scale)
    return distribution