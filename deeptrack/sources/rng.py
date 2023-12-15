
import numpy as np
import random 

from typing import Any

from deeptrack.sources.base import Source
from deeptrack.backend.core import DeepTrackNode


class NumpyRNGSource(Source, np.random.RandomState):

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
    

class PythonRNGSource(Source, random.Random):

    
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