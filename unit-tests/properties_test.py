''' Unit tests for distributions.py

'''

import unittest
import numpy as np
from deeptrack.distributions import Distribution, random_uniform

class TestDistributions(unittest.TestCase):

    def test_number(self):
        input_number = 1
        D = Distribution(input_number)
        for i in range(20):
            with self.subTest(i=i):
                D.__update__([])
                self.assertEqual(D.current_value, input_number)


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
        input_random_uniform_numbers = random_uniform(scale)
        D = Distribution(input_random_uniform_numbers)
        for i in range(20):
            with self.subTest(i=i):
                D.__update__([])
                self.assertTrue(np.all(D.current_value >= 0))
                self.assertTrue(np.all(D.current_value <= np.array(scale)))
