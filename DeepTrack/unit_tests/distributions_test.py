''' Unit tests for the distributions module.
'''

import unittest
import numpy as np
from DeepTrack.distributions import Distribution, random_uniform

class TestDistributions(unittest.TestCase):

    def test_number(self):
        D = Distribution(1)
        for i in range(20):
            with self.subTest(i=i):
                D.__update__([])
                self.assertEqual(D.current_value, 1)


    def test_tuple(self):
        input_tuple = (0, 1, np.inf, None)
        D = Distribution(input_tuple)
        for i in range(20):
            with self.subTest(i=i):
                D.__update__([])
                self.assertEqual(D.current_value, input_tuple)


    def test_list(self):
        input_list = [0, 1, np.inf, None]
        D = Distribution(input_list)
        for i in range(20):
            with self.subTest(i=i):
                D.__update__([])
                self.assertTrue(D.current_value in input_list)


    def test_random_unifrom(self):
        scale = (1, 2, 3)
        D = Distribution(random_uniform(scale))
        for i in range(20):
            with self.subTest(i=i):
                D.__update__([])
                self.assertTrue(np.all(D.current_value >= 0))
                self.assertTrue(np.all(D.current_value <= np.array(scale)))
