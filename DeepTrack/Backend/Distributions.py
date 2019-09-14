import numpy as np
def uniform_random(scale):
    def distribution():
        return np.random.rand(len(scale))*np.array(scale)
    return distribution

def draw(E):
    
    # If the inpu it callable, treat it as a distribution
    if callable(E):
        return E()
    
    # Else, check if input is an array
    if isinstance(E, (list, np.ndarray)):
        return np.random.choice(E)

    # Else, assume it's a single number.
    return E
