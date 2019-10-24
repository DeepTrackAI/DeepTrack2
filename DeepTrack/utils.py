''' Utility functions

Defines a set of utility functions used throughout the code
to make it more readable.

Functions
---------
hasfunction(object:any, function_name:str) -> bool
    returns True if the object has a field named function_name
    that is callable. Otherwise it returns False.
isiterable(object:any) 
    returns True if the object is iterable, else, return False.
'''


def hasmethod(object:any, method_name:str) -> bool:
    ''' Checks if the input has a callable method named method_name
    Parameters
    ----------
    object
        The object upon which to check.
    functmethod_nameion_name : str
        the name of the method
        
    Returns
    -------
    bool
        True if the object has an attribute method_name, and that
        attribute is callable.
    '''
    return hasattr(object, method_name) and callable(object.method_name) 
    
def isiterable(object:any) -> bool:
    ''' Checks if the input is iterable
    Parameters
    ----------
    object
        The object upon which to check.
    
    Returns
    -------
    bool
        True if the object either has a __iter__ field or a __getitem__
        field.
    '''
    return hasattr(object, "__next__")