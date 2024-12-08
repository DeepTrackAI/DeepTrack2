# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

from deeptrack.backend import core


class TestCore(unittest.TestCase):

    def test_DeepTrackDataObject(self):
        dataobj = core.DeepTrackDataObject()

        # Test storing and validating data.
        dataobj.store(1)
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), True)

        # Test invalidating data.
        dataobj.invalidate()
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), False)

        # Test validating data.
        dataobj.validate()
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), True)


    def test_DeepTrackDataDict(self):
        dataset = core.DeepTrackDataDict()

        # Test initial state.
        self.assertEqual(dataset.keylength, None)
        self.assertFalse(dataset.dict)

        # Create indices and store data.
        dataset.create_index((0,))
        dataset[(0,)].store({"image": [1, 2, 3], "label": 0})

        dataset.create_index((1,))
        dataset[(1,)].store({"image": [4, 5, 6], "label": 1})

        self.assertEqual(dataset.keylength, 1)
        self.assertEqual(len(dataset.dict), 2)
        self.assertIn((0,), dataset.dict)
        self.assertIn((1,), dataset.dict)

        # Test retrieving stored data.
        self.assertEqual(dataset[(0,)].current_value(),
                         {"image": [1, 2, 3], "label": 0})
        self.assertEqual(dataset[(1,)].current_value(),
                         {"image": [4, 5, 6], "label": 1})

        # Test validation and invalidation - all.
        self.assertTrue(dataset[(0,)].is_valid())
        self.assertTrue(dataset[(1,)].is_valid())

        dataset.invalidate()
        self.assertFalse(dataset[(0,)].is_valid())
        self.assertFalse(dataset[(1,)].is_valid())

        dataset.validate()
        self.assertTrue(dataset[(0,)].is_valid())
        self.assertTrue(dataset[(1,)].is_valid())

        # Test validation and invalidation - single node.
        self.assertTrue(dataset[(0,)].is_valid())

        dataset[(0,)].invalidate()
        self.assertFalse(dataset[(0,)].is_valid())
        self.assertTrue(dataset[(1,)].is_valid())

        dataset[(1,)].invalidate()
        self.assertFalse(dataset[(0,)].is_valid())
        self.assertFalse(dataset[(1,)].is_valid())

        dataset[(0,)].validate()
        self.assertTrue(dataset[(0,)].is_valid())
        self.assertFalse(dataset[(1,)].is_valid())

        dataset[(1,)].validate()
        self.assertTrue(dataset[(0,)].is_valid())
        self.assertTrue(dataset[(1,)].is_valid())

        # Test iteration over entries.
        for key, value in dataset.dict.items():
            self.assertIn(key, {(0,), (1,)})
            self.assertIsInstance(value, core.DeepTrackDataObject)


    def test_DeepTrackNode(self):
        pass


if __name__ == "__main__":
    unittest.main()
