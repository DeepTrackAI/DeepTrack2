'''Tools to manage the properties of a feature

Classes
-------
Property
    The class `Property`, which represents the values of a property of a feature.
    A Property be:
    * A constant (initialization with, e.g., a number, a tuple)
    * A sequence of variables (initialization with, e.g., an iterator)
    * A random variable (initialization with, e.g., a function)
SequentialProperty
    The class `SequentialProperty`, which extends `Property` to sample one value
    for each step in a sequence.
PropertyDict
    The class `PropertyDict`, which is a dictionary with each element a Property.
    The class provides utility functions to update, sample, clear and retrieve
    properties.
'''

import numpy as np
from deeptrack.utils import isiterable, hasmethod, get_kwarg_names, kwarg_has_default
import deeptrack



class Property:
    '''Represents a property of a feature

    The class Property` wraps an input, which is treated
    internally as a sampling rule. This sampling rule is used
    to update the value of the property.
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
    has_updated_since_last_resolve : deprecated, bool
        Whether the property has been updated since the last resolve.
    dummy
        Whether the property is used by the feature

    '''

    def __init__(self, sampling_rule: any):
        self.sampling_rule = sampling_rule
        self.last_update_id = -1
    

    @property
    def current_value(self):
        '''Current value of the property of the feature

        `current_value` is the result of the latest `update()` call.
        Note that any randomization only occurs when the method `update()` is called
        and, therefore, the current value does not change between calls.

        The method getter calls the method `update()` if `current_value`
        has not been set yet.

        '''

        self._current_value

    @current_value.setter
    def current_value(self, updated_current_value):
        self._current_value = updated_current_value

    @current_value.getter
    def current_value(self):
        if not hasattr(self, "_current_value"):
            self.update()
        return self._current_value


    def update(self, **kwargs) -> 'Property':
        '''Updates the current value

        The method `update()` sets the property `current_value`
        as the output of the method `sample()`. Will only update
        once per resolve.

        Any object that implements the method `update()` will have it called.

        Returns
        -------
        Property
            Returns itself.

        '''
        old_key =  deeptrack.features._SESSION_STRUCT["update_key"]
        provided_key = kwargs.get("_update_key", False) or deeptrack.features._SESSION_STRUCT["update_key"]

        if self.last_update_id == provided_key:
            return self
        

        
        self.last_update_id = provided_key

        # if hasmethod(self.sampling_rule, "update") and not isinstance(self.sampling_rule, dict):
        #     self.sampling_rule.update(_update_key=provided_key)
        
        kwargs["_update_key"] = provided_key
        self.current_value = self.sample(self.sampling_rule, **kwargs)

        return self


    def sample(self, sampling_rule, **kwargs):
        '''Samples the sampling rule

        Returns a sampled instance of the `sampling_rule` field.
        The logic behind the sampling depends on the type of
        `sampling_rule`. These are checked in the following order of
        priority:

        1. Any object with a callable `sample()` method has this
            method called and returned.
        2. If the rule is a ``dict``, sample each value and combine the 
            result into a new ``dict`` using the original keys.
        3. If the rule is a ``list``, sample each element of the list and
            combine the result into a ne ``list``.
        4. If the rule is an ``iterable``, return the next output.
        5. If the rule is callable, call it with its accepted arguments.
            Example arguments can be the value of some other property.
        6. If none of the above apply, return the rule itself.
        
        Parameters
        ----------
        sampling_rule : any
            The rule to sample values from.
        **kwargs
            Arguments that will be passed on to functions that accepts them.

        Returns
        -------
        any
            A sampled instance of the `sampling_rule`.

        '''

        if isinstance(sampling_rule, deeptrack.Feature):
            # Don't pass my properties to other feature (avoid name clash)
            sampling_rule.update(_update_key=kwargs["_update_key"])
            return sampling_rule

        if hasmethod(sampling_rule, "sample"):
            # If the ruleset itself implements a sample method,
            # call it instead.
            return sampling_rule.sample(**kwargs)

        elif isinstance(sampling_rule, dict):
            # If the ruleset is a dict, return a new dict with each
            # element being sampled from the original dict.
            out = {}
            for key, val in sampling_rule.items():
                out[key] = self.sample(val, **kwargs)
            return out

        elif isinstance(sampling_rule, list):
            return [self.sample(item, **kwargs) for item in sampling_rule]

        elif isinstance(sampling_rule, (tuple, np.ndarray)):
            # tuple and ndarrays are elementary
            return sampling_rule

        elif isiterable(sampling_rule):
            # If it's iterable, return the next value
            try:
                return next(sampling_rule)
            except StopIteration:
                return self.current_value
  
        elif callable(sampling_rule):
            # If it's a function, extract the arguments it accepts.
            function_input = {}

            # Get the kwarg arguments the function accepts
            for key in get_kwarg_names(sampling_rule):
                # If that name is among passed kwarg arguments
                if key in kwargs:
                    if isinstance(kwargs[key], Property):
                        # If it is a property, update it and pass the current value
                        if not kwargs[key] is self:
                            kwargs[key].update(**kwargs)
                        if hasattr(kwargs[key], "current_value"): 
                            if isinstance(kwargs[key], SequentialProperty) and "sequence_step" in kwargs:
                                kwargs[key] = kwargs[key].current_value[kwargs["sequence_step"]]
                            else:
                                kwargs[key] = kwargs[key].current_value
                        else:
                            kwargs[key] = None

                    function_input[key] = kwargs[key]

                elif not kwarg_has_default(sampling_rule, key):
                    function_input[key] = None

            return sampling_rule(**function_input)
            
        else:
            # Else, assume it's elementary.
            return sampling_rule
    
    def reset(self):
        ''' Clear the internal state of the property

        Only useful in rare instances where properties are used without them being
        attached to any feature.

        Returns
        -------
        self
        '''
        self.last_update_id = -1
        return self

    # def __deepcopy__(self, memo):
    #     from copy import deepcopy, copy

        
    #     # if id(self.sampling_rule) in memo:
    #     #     obj = copy(self)
    #     #     obj.sampling_rule = memo[id(self.sampling_rule)]
    #     #     return obj

    #     if self.dummy and isinstance(self.sampling_rule, deeptrack.Feature) and id(self.sampling_rule) not in memo:
    #         return self

    #     else:
    #         if isinstance(self.sampling_rule, deeptrack.Feature):
    #             b = 1+1
    #             pass
    #         oldc = self.__deepcopy__
    #         self.__deepcopy__ = None
    #         obj = deepcopy(self, memo)
    #         obj.__deepcopy__ = oldc
    #         return obj


import copy


class SequentialProperty(Property):
    ''' Property that has multiple sequential values

        Extends standard `Property` to resolve one value for each step
        in a sequence of images. This is often used when creating movies.

        Parameters
        ----------
        initializer : any
            Sampling rule for the first step of the sequence.
        sampling_rule : any
            Sampling rule for each step after the first.
        
        Attributes
        ----------
        initializer : any
            Sampling rule for the first step of the sequence.
        sampling_rule : any
            Sampling rule for each step after the first.
        has_updated_since_last_resolve : bool
            Whether the property has been updated since the last resolve.

    '''

    def __init__(self, sampling_rule, initializer=None):
        super().__init__(sampling_rule)
        if initializer is None:
            self.initializer = sampling_rule
        else:
            self.initializer = initializer

        self.sampling_rule = sampling_rule

        # Deprecated
        self.has_updated_since_last_resolve = False
        

    def update(self, sequence_length=0, **kwargs):
        '''Updates current_value

        For each step in the sequence, sample `self.sampling_rule`.
        `self.initializer` is used for the first step. These rules
        should output one value per step. Sampling rules
        that are functions can optionally accept the following keyword
        arguments:
        
        * sequence_step : the current position in the sequence.
        * sequence_length : the length of the sequence.
        * previous_value : the value of the property at the previous
          step in the sequence.
        * previous_values : the value of the property at all previous
          steps in the sequence.

        Parameters
        ----------
        sequence_length : int, optional
            length of the sequence
        
        Returns
        -------
        self
            returns self
        '''

        provided_key = kwargs.get("_update_key", False) or deeptrack.features._SESSION_STRUCT["update_key"]

        if self.last_update_id == provided_key:
            return self
        
        self.last_update_id = provided_key

        new_current_value = []

        for step in range(sequence_length):
            # Use initializer for first time step
            ruleset = self.sampling_rule
            
            # Elements inserted here can be passed to property functions
            kwargs.update(
                sequence_step=step,
                sequence_length=sequence_length,
                previous_values=new_current_value)
            if step == 0:
                kwargs.update(previous_value=self.sample(self.initializer, **kwargs))
            else:
                kwargs.update(previous_value=new_current_value[-1])

            next_value = self.sample(ruleset, **kwargs)

            new_current_value.append(next_value)

        self.current_value = new_current_value

        return self

class PropertyDict(dict):
    ''' Dictionary with Property elements

    A dictionary of properties. It provides utility functions to update, 
    sample, reset and retrieve properties.

    Parameters
    ----------
    *args, **kwargs
        Arguments used to initialize a dict

    '''


    def current_value_dict(self, is_resolving=False, **kwargs) -> dict:
        ''' Retrieves the current value of all properties as a dictionary

        Returns
        -------
        dict
            A dictionary with the current value of all properties

        '''
        current_value_dict = {}
        for key, property in self.items():
            
            property_value = property.current_value
            
            
            # If the property is sequential, retrieve the value
            # of the current timestep
            if isinstance(property, SequentialProperty):
                

                sequence_step = kwargs.get("sequence_step", None)
                if not sequence_step is None:
                    property_value = property_value[sequence_step]

            current_value_dict[key] = property_value


        return current_value_dict


    def update(self, **kwargs) -> 'PropertyDict':
        ''' Updates all properties

        Calls the method `update()` on each property in the dictionary.

        Returns
        -------
        Properties
            Returns itself

        '''
        property_arguments = dict(self)
        property_arguments.update(kwargs)
        for key, prop in self.items():

            if isinstance(property_arguments[key], Property):
                prop.update(**property_arguments)
            else:
                prop.current_value = kwargs[key]
                provided_key = kwargs.get("_update_key", False) or deeptrack.features._SESSION_STRUCT["update_key"]
                prop.last_update_id = provided_key

        return self


    def sample(self, **kwargs) -> dict:
        ''' Samples all properties

        Returns
        -------
        dict
            A dictionary with each key-value pair the result of a
            `sample()` call on the property with the same key.

        '''

        sample_dict = {}
        for key, property in self.items():
            sample_dict[key] = property.sample(**kwargs)

        return sample_dict

    def reset(self):
        ''' Clear the internal state of each property in the dict

        Only useful in rare instances where properties are used without them being
        attached to any feature.

        Returns
        -------
        self
        '''
        for prop in self.values(): 
            prop.reset()
        return self