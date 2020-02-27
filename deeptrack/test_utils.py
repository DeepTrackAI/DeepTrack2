import unittest

import utils


class TestUtils(unittest.TestCase):
    
    def test_hasmethod(self):
        self.assertTrue(utils.hasmethod(utils, 'hasmethod'))
        self.assertFalse(utils.hasmethod(utils, 'this_is_definetely_not_a_method_of_utils'))
    

if __name__ == '__main__':
    unittest.main()