"""
This utility package extends the random number generator objects for both
Python and Numpy by adding functions to generate several instances as well as
dependency tracking with DeepTrackNode objects.
"""

import numpy as np
import random 

from typing import Any

from deeptrack.sources.base import Source
from deeptrack.backend.core import DeepTrackNode


class NumpyRNG(Source, np.random.RandomState):
    """Class that generates multiple numpy random number generators.

    It is used for creating multiple rng's with different seeds.

    Parameters
    ----------
    n_states : int
        The number of random number generators to create.

    seed : int, optional
        The seed used to initialize the first random generator. If not provided, a 
        random seed will be generated automatically using `np.random.randint()`.

    Attributes
    ----------
    rng : list of numpy.Random
        A list of `numpy.Random` objects, each seeded with a unique value.
        
    Methods
    -------
    _generate_states() : list
        Generates and returns a list of independent `numpy.Random` objects.
        
    reset() : None
        Resets the list of random number generators with new seeds.

    __getattribute__(__name) : Any
        Custom attribute access to allow lazy evaluation
        of random number generator methods.
        
    _create_lazy_callback(__name) : callable
        Creates a lazy callback for accessing methods 
        from the `numpy.Random` objects.

    set_index(index) : self
        Sets the current index and resets the random number generators.
    """

    rng: list

    def __init__(self, n_states, seed=None):
        self._n_states = n_states

        if seed is None:
            seed = np.random.randint(0, 2**31)
        self._seed = seed

        states = self._generate_states()

        super().__init__(rng=states)

    def _generate_states(self):

        n_states = self._n_states
        seed = self._seed

        seed_generator = np.random.RandomState(seed)
        return [np.random.RandomState(seed_generator.randint(0, 2**31)) for _ in range(n_states)]

    def reset(self):
        self._dict["rng"] = self._generate_states()

    
    def __getattribute__(self, __name: str) -> Any:
        if hasattr(np.random.RandomState, __name) and not __name.startswith("_"):
            return self._create_lazy_callback(__name)
        return super().__getattribute__(__name)
    
    def _create_lazy_callback(self, __name: str):
        def lazy_callback(*args, **kwargs):
            node = DeepTrackNode(lambda: getattr(self._dict["rng"][self._current_index()], __name)(*args, **kwargs))
            node.add_dependency(self._current_index)
            self._current_index.add_child(node)
            return node
        return lazy_callback
    

    def set_index(self, index):
        self.reset()
        return super().set_index(index)
    

class PythonRNG(Source, random.Random):
    """Class that generates multiple random.Random number generators.

    It is used for creating multiple rng's with different seeds.

    Parameters
    ----------
    n_states : int
        The number of random number generators to create.

    seed : int, optional
        The seed used to initialize the first random generator. If not provided, a 
        random seed will be generated automatically
        using `random.Random.randint()`.

    Attributes
    ----------
    rng : list of random.Random
        A list of `random.Random` objects, each seeded with a unique value.
        
    Methods
    -------
    _generate_states() : list
        Generates and returns a list of independent `random.Random` objects.
        
    reset() : None
        Resets the list of random number generators with new seeds.

    __getattribute__(__name) : Any
        Custom attribute access to allow lazy evaluation
        of random number generator methods.
        
    _create_lazy_callback(__name) : callable
        Creates a lazy callback for accessing methods 
        from the `random.Random` objects.

    set_index(index) : self
        Sets the current index and resets the random number generators.
    """

    
    rng: list

    def __init__(self, n_states, seed=None):
        self._n_states = n_states

        if seed is None:
            seed = np.random.randint(0, 2**31)
        self._seed = seed

        states = self._generate_states()

        super().__init__(rng=states)

    def _generate_states(self):

        n_states = self._n_states
        seed = self._seed

        seed_generator = random.Random(seed)
        return [random.Random(seed_generator.randint(0, 2**31)) for _ in range(n_states)]

    def reset(self):
        self._dict["rng"] = self._generate_states()

    
    def __getattribute__(self, __name: str) -> Any:
        if hasattr(np.random.RandomState, __name) and not __name.startswith("_"):
            return self._create_lazy_callback(__name)
        return super().__getattribute__(__name)
    
    def _create_lazy_callback(self, __name: str):
        def lazy_callback(*args, **kwargs):
            node = DeepTrackNode(lambda: getattr(self._dict["rng"][self._current_index()], __name)(*args, **kwargs))
            node.add_dependency(self._current_index)
            self._current_index.add_child(node)
            return node
        return lazy_callback
    

    def set_index(self, index):
        self.reset()
        return super().set_index(index)