import numpy as np

'''
Returns a ndarray of the same shape as the input argument, with each number uniformly distributed between 0 and scale

Input arguments:
    scale:          The maximum of the distribution for each output.
'''
def uniform_random(scale):
    def distribution():
        return tuple(np.random.rand(len(scale))*np.array(scale))
    return distribution


'''
    Takes an input and recursively extracts a single elementary type.

    For callable elements, the output of the call is drawn

    For lists or ndarrays, a single element is drawn

    If the input is neither callable, a list, nor an ndarray, return the element.

    To return multiple values, store them as a tuple. 
'''
def draw(E):
    
    # If the input it callable, treat it as a distribution.
    if callable(E):
        return draw(E())
    
    # Else, check if input is an array, and extract a single element
    if isinstance(E, (list, np.ndarray)):
        return draw(np.random.choice(E))

    # Else, assume it's elementary.
    return E
