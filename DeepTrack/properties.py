'''Tools to manage the properties of features

This module contains:
    
The class `Property`, which represents the values of a property of a feature.
A Property can be represented by:
* A constant (inizialization with, e.g., a number, a tuple)
* A sequence of variables (inizialization with, e.g., a generator)
* A discrete random variable (inizialization with, e.g., a list, a dictionary)
* A continuous random variable (inizialization with, e.g., a function)

The class `Properties`, which is a dictionary with each element a Property. 
The class provides utility functions to update, sample, clear and retrieve 
properties. 

A series of standard functions:
    
random_uniform(scale) -> function
    Uniform random distribution

'''

from DeepTrack.utils import isiterable, hasmethod
import numpy as np


# CLASSES

class Property:
    r'''Represents a property of a feature
    
    The class Property wraps an input, which is treated
    internally as a sampling rule. This sampling rule is used 
    to update the value of the property of the feature. 
    The sampling rule can be, for example:
    * A constant (initialization with, e.g., a number, a tuple)
    * A sequence of variables (initialization with, e.g., a generator)
    * A discrete random variable (initialization with, e.g., a list, a dictionary)
    * A continuous random variable (initialization with, e.g., a function)

    Parameters
    ----------
    sampling_rule : any        
        Defines the sampling rule to update the value of the feature property. 
        See method `sample()` for how different sampling rules are sampled.

    Attributes
    ----------
    sampling_rule : any
        The sampling rule to update the value of the feature property.
    current_value : any
        The current value obtained from the last call to the sampling rule.
    
    Examples
    --------
    When the sampling rule is a number, 
    the current value is always the number itself:

    >>> D = Property(1)
    >>> D.current_value
    1
    >>> D.update()
    >>> D.current_value
    1

    When the sampling rule is a list, 
    the current value is one of the elements of the list:

    >>> np.random.seed(0)
    >>> D = Property([1, 2, 3])
    >>> D.current_value # Either 1, 2 or 3 randomly
    2 #random
    >>> D.current_value # Same as last call
    2 #random
    >>> D.update()
    >>> D.current_value
    1 #random        

    '''

    def __init__(self, sampling_rule:any):
        self.sampling_rule = sampling_rule
    


    @property
    def current_value(self):
        r'''Current value of the property of the feature

        `current_value` is the result of the latest `update` call.
        Note that any randomization only occurs when the method `update` is called
        and, therefore, the current value does not change between calls.

        The method getter calls the method `update` 
        if `current_value` has not yet been set.

        '''

        self._current_value

    @current_value.setter
    def current_value(self, updated_current_value):
        self._current_value = updated_current_value
    
    @current_value.getter
    def current_value(self):
        if not hasattr(self, "_current_value"):
            self.update() # generate new current value
        return self._current_value

    

    def update(self):
        r'''Updates the current value

        The method `update` sets the property `current_value` 
        as the output of the method `sample`.

        '''
        self.current_value = self.sample()



    def sample(self):
        r'''Samples the sampling rule

        Returns a sampled instance of the `sampling_rule` field.
        The logic behind the sampling depends on the type of
        `sampling_rule`. These are checked in the following order of
        priority:

        1. Any object with a callable `sample` method has this
            method called and returned.
        2. If the rule is a ``dict``, the ``dict`` is copied and any value
            with has a callable sample method is replaced with the
            output of that call.
        3. If the rule is a ``list`` or a 1-dimensional ``ndarray``, return 
        4. If the rule is an ``iterable``, return the next output.
        5. If the rule is callable, return the output of a call with
            no input parameters.
            a single element drawn from that ``list``/``ndarray``.
        6. If none of the above apply, return the rule itself.

        Returns
        -------
        any
            A sampled instance of the `sampling_rule`. 
            
        '''

        sampling_rule = self.sampling_rule

        if hasmethod(sampling_rule, "sample"):
            # If the ruleset itself implements a sample method,
            # call it instead.
            return sampling_rule.sample()

        elif isinstance(sampling_rule, dict):
            # If the ruleset is a dict, return a new dict with each
            # element being sampled from the original dict.
            out = {}
            for key, val in self.sampling_rule.items():
                if hasmethod(val, 'sample'):
                    out[key] = val.sample()
                else:
                    out[key] = val
            return out
        
        elif (isinstance(sampling_rule, list) or
              isinstance(sampling_rule, np.ndarray) and sampling_rule.ndim == 1):
            # If it's either a list or a 1-dimensional ndarray,
            # return a random element from the list. 
            return np.random.choice(sampling_rule)
        
        elif isiterable(sampling_rule):
            # If it's iterable, return the next value
            return next(sampling_rule)
            
        elif callable(sampling_rule):
            # If it's a function, call it without parameters
            return sampling_rule()
        
        else:
            # Else, assume it's elementary.
            return self.sampling_rule

# TODO: Maybe PropertyDict is clearer?
class Properties(dict):
    ''' Dictionary with Property elements
    
    A dictionary of properties. It provides utility functions to update, 
    sample, reset and retrieve properties.

    Parameters
    ----------
    *args, **kwargs
        Arguments used to initialize a dict
    
    '''

    def __init__(self, *args, **kwargs):
        self.prop_dict = dict(*args, **kwargs)


    # TODO: call reset
    # TODO: change prop to property / properties
    def clear(self) -> 'Properties':
        ''' Clears/resets properties

        Clears any property with a defined method `clear`  

        Returns
        -------
        Properties
            Returns itself
        '''
        for prop in self.data.values():
            if hasmethod(prop, 'clear'):
                prop.clear()
        return self
    

    def sample(self) -> dict:
        ''' Samples all properties

        Returns
        -------
        dict
            A dictionary with each key-value pair the result of a 
            `sample()` call on the property with the same key.

        '''

        sample_dict = {}
        for key in self.prop_dict.keys():
            sample_dict[key] = self.prop_dict[key].sample()
        return sample_dict


    def update(self) -> 'Properties':
        ''' Updates all properties

        Calls the method `update` on each property in the dictionary.

        Returns
        -------
        Properties
            Returns itself

        '''
        for prop in self.prop_dict.values():
            prop.update()
        return self

    # TODO: first method, also order other methods readably    
    def current_value_dict(self):
        '''
        '''
        current_value_dict = {}
        for key in self.prop_dict.keys():
            current_value_dict[key] = self.prop_dict[key].current_value
        return current_value_dict


# FUNCTIONS

import types
# TODO: allow for min/max definition
def random_uniform(scale) -> types.FunctionType:
    ''' Uniform random distribution

    Parameters
    ----------
    scale          
        The maximum of the distribution
        
    Returns
    -------
    function
        function that returns a ndarray of the same shape 
        as the input argument `scale`, where each number 
        uniformly distributed between 0 and `scale`

    '''

    scale = np.array(scale)
    def distribution():
        return np.random.rand(*scale.shape) * scale
    return distribution