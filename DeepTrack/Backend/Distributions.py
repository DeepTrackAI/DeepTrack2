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

class SequenceGenerator:
    def __init__(
        self,
        init,
        next=None,
        correlation_length=1,
        return_all = False
    ):
        self.init = init
        if next is None:
            next = init
        self.next = init
        self.correlation_length=correlation_length
        self.return_all=return_all
        
    def __iter__(self):
        self.X = [draw(self.init)]

    def __next__(self):

        self.X.append(self.next(self.X[-self.correlation_length:]))

        if self.return_all:
            return self.X
        else:
            return self.X[-1]



'''
    Takes an input and recursively extracts a single elementary type.

    For callable elements, the output of the call is drawn

    For lists or ndarrays, a single element is drawn

    If the input is neither callable, a list, nor an ndarray, return the element.

    To return multiple values, store them as a tuple. 
'''
def draw(E):
    
    # If the input is a sequence generator, retrieve next item

    if isinstance(E, SequenceGenerator):
        return draw(next(E))

    # If the input is callable, treat it as a distribution.
    if callable(E):
        return draw(E())
    
    # Else, check if input is an array, and extract a single element
    if isinstance(E, (list, np.ndarray)):
        return draw(np.random.choice(E))

    # Else, assume it's elementary.
    return E
