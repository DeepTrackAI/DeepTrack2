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


    def test_DeepTrackNode_basics(self):
        node = core.DeepTrackNode(action=lambda: 42)

        # Evaluate the node.
        result = node()  # Value is calculated and stored.
        self.assertEqual(result, 42)

        # Store a value.
        node.store(100)  # Value is stored.
        self.assertEqual(node.current_value(), 100)
        self.assertTrue(node.is_valid())

        # Invalidate the node and check the value.
        node.invalidate()
        self.assertFalse(node.is_valid())

        self.assertEqual(node.current_value(), 100)  # Value is retrieved.
        self.assertFalse(node.is_valid())

        self.assertEqual(node(), 42)  # Value is calculated and stored.
        self.assertTrue(node.is_valid())


    def test_DeepTrackNode_dependencies(self):
        parent = core.DeepTrackNode(action=lambda: 10)
        child = core.DeepTrackNode(action=lambda _ID=None: parent() * 2)
        parent.add_child(child)  # Establish dependency.

        # Check that the just create nodes are invalid as not calculated.
        self.assertFalse(parent.is_valid())
        self.assertFalse(child.is_valid())

        # Calculate child, and therefore parent.
        result = child()
        self.assertEqual(result, 20)
        self.assertTrue(parent.is_valid())
        self.assertTrue(child.is_valid())

        # Invalidate parent and check child validity.
        parent.invalidate()
        self.assertFalse(parent.is_valid())
        self.assertFalse(child.is_valid())

        # Validate parent and ensure child is invalid until recomputation.
        parent.validate()
        self.assertTrue(parent.is_valid())
        self.assertFalse(child.is_valid())  ###TODO this test doesn't pass!

        # Recompute child and check its validity
        child()
        self.assertTrue(parent.is_valid())
        self.assertTrue(child.is_valid())


    def test_DeepTrackNode_overloading(self):
        node1 = core.DeepTrackNode(action=lambda: 5)
        node2 = core.DeepTrackNode(action=lambda: 10)

        sum_node = node1 + node2
        self.assertEqual(sum_node(), 15)

        diff_node = node2 - node1
        self.assertEqual(diff_node(), 5)

        prod_node = node1 * node2
        self.assertEqual(prod_node(), 50)

        div_node = node2 / node1
        self.assertEqual(div_node(), 2)

if __name__ == "__main__":
    unittest.main()
