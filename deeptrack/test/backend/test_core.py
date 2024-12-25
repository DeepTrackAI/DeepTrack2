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
        self.assertFalse(child.is_valid())

        # Recompute child and check its validity
        child()
        self.assertTrue(parent.is_valid())
        self.assertTrue(child.is_valid())

    def test_DeepTrackNode_nested_dependencies(self):
        parent = core.DeepTrackNode(action=lambda: 5)
        middle = core.DeepTrackNode(action=lambda: parent() + 5)
        child = core.DeepTrackNode(action=lambda: middle() * 2)

        parent.add_child(middle)
        middle.add_child(child)

        result = child()
        self.assertEqual(result, 20, "Nested computation failed.")

        # Invalidate the middle and check propagation.
        middle.invalidate()
        self.assertTrue(parent.is_valid())
        self.assertFalse(middle.is_valid())
        self.assertFalse(child.is_valid())


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


    def test_DeepTrackNode_citations(self):
        node = core.DeepTrackNode(action=lambda: 42)
        citations = node.get_citations()
        self.assertIn(core.citation_midtvet2021quantitative, citations)


    def test_DeepTrackNode_single_id(self):
        """Test a single _ID on a simple parent-child relationship."""

        parent = core.DeepTrackNode(action=lambda: 10)
        child = core.DeepTrackNode(action=lambda _ID=None: parent(_ID) * 2)
        parent.add_child(child)

        # Store value for a specific _ID's.
        for id, value in enumerate(range(10)):
            parent.store(id, _ID=(id,))

        # Retrieves the values stored in children and parents.
        for id, value in enumerate(range(10)):
            self.assertEqual(child(_ID=(id,)), value * 2)
            self.assertEqual(parent.previous((id,)), value)

    def test_DeepTrackNode_nested_ids(self):
        """Test nested IDs for parent-child relationships."""

        parent = core.DeepTrackNode(action=lambda: 10)
        child = core.DeepTrackNode(
            action=lambda _ID=None: parent(_ID[:1]) * _ID[1]
        )
        parent.add_child(child)

        # Store values for parent at different IDs.
        parent.store(5, _ID=(0,))
        parent.store(10, _ID=(1,))

        # Compute child values for nested IDs
        child_value_0_0 = child(_ID=(0, 0))  # Uses parent(_ID=(0,)).
        self.assertEqual(child_value_0_0, 0)

        child_value_0_1 = child(_ID=(0, 1))  # Uses parent(_ID=(0,)).
        self.assertEqual(child_value_0_1, 5)

        child_value_1_0 = child(_ID=(1, 0))  # Uses parent(_ID=(1,)).
        self.assertEqual(child_value_1_0, 0)

        child_value_1_1 = child(_ID=(1, 1))  # Uses parent(_ID=(1,)).
        self.assertEqual(child_value_1_1, 10)


    def test_DeepTrackNode_replicated_behavior(self):
        """Test replicated behavior where IDs expand."""

        particle = core.DeepTrackNode(action=lambda _ID=None: _ID[0] + 1)

        # Replicate node logic.
        cluster = core.DeepTrackNode(
            action=lambda _ID=None: particle(_ID=(0,)) + particle(_ID=(1,))
        )

        cluster_value = cluster()
        self.assertEqual(cluster_value, 3)

    def test_DeepTrackNode_parent_id_inheritance(self):

        # Children with IDs matching than parents.
        parent_matching = core.DeepTrackNode(action=lambda: 10)
        child_matching = core.DeepTrackNode(
            action=lambda _ID=None: parent_matching(_ID[:1]) * 2
        )
        parent_matching.add_child(child_matching)

        parent_matching.store(7, _ID=(0,))
        parent_matching.store(5, _ID=(1,))

        self.assertEqual(child_matching(_ID=(0,)), 14)
        self.assertEqual(child_matching(_ID=(1,)), 10)

        # Children with IDs deeper than parents.
        parent_deeper = core.DeepTrackNode(action=lambda: 10)
        child_deeper = core.DeepTrackNode(
            action=lambda _ID=None: parent_deeper(_ID[:1]) * 2
        )
        parent_deeper.add_child(child_deeper)

        parent_deeper.store(7, _ID=(0,))
        parent_deeper.store(5, _ID=(1,))

        self.assertEqual(child_deeper(_ID=(0, 0)), 14)
        self.assertEqual(child_deeper(_ID=(0, 1)), 14)
        self.assertEqual(child_deeper(_ID=(0, 2)), 14)

        self.assertEqual(child_deeper(_ID=(1, 0)), 10)
        self.assertEqual(child_deeper(_ID=(1, 1)), 10)
        self.assertEqual(child_deeper(_ID=(1, 2)), 10)

    def test_DeepTrackNode_invalidation_and_ids(self):
        """Test that invalidating a parent affects specific IDs of children."""

        parent = core.DeepTrackNode(action=lambda: 10)
        child = core.DeepTrackNode(action=lambda _ID=None: parent(_ID[:1]) * 2)
        parent.add_child(child)

        # Store and compute values.
        parent.store(0, _ID=(0,))
        parent.store(1, _ID=(1,))
        child(_ID=(0, 0))
        child(_ID=(0, 1))
        child(_ID=(1, 0))
        child(_ID=(1, 1))

        # Invalidate the parent at _ID=(0,).
        parent.invalidate((0,))
        
        self.assertFalse(parent.is_valid((0,)))
        self.assertFalse(parent.is_valid((1,)))
        self.assertFalse(child.is_valid((0, 0)))
        self.assertFalse(child.is_valid((0, 1)))
        self.assertFalse(child.is_valid((1, 0)))
        self.assertFalse(child.is_valid((1, 1)))


    def test_DeepTrackNode_dependency_graph_with_ids(self):
        """Test a multi-level dependency graph with nested IDs."""

        A = core.DeepTrackNode(action=lambda: 10)
        B = core.DeepTrackNode(action=lambda _ID=None: A(_ID[:-1]) + 5)
        C = core.DeepTrackNode(
            action=lambda _ID=None: B(_ID[:-1]) * (_ID[-1] + 1)
        )
        A.add_child(B)
        B.add_child(C)

        # Store values for A at different IDs.
        A.store(3, _ID=(0,))
        A.store(4, _ID=(1,))

        # Compute values for C at nested IDs.
        C_0_1_2 = C(_ID=(0, 1, 2))  # B((0, 1)) * (2 + 1)
                                    # (A((0,)) + 5) * (2 + 1)
                                    # (3 + 5) * (2 + 1)
                                    # 24
        self.assertEqual(C_0_1_2, 24)


if __name__ == "__main__":
    unittest.main()
