import numpy as np
def uniform_random(scale):
    def distribution():
        return np.random.rand(len(scale))*np.array(scale)
    return distribution