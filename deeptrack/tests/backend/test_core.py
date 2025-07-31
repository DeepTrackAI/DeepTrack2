# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

# Use this only when running the test locally.
# import sys
# sys.path.append(".")  # Adds the module to path.

import unittest

import random

from deeptrack.backend import core


class TestCore(unittest.TestCase):

    def test___all__(self):
        from deeptrack import (
            DeepTrackDataDict,
            DeepTrackDataObject,
            DeepTrackNode,
        )
        from deeptrack.backend import (
            DeepTrackDataDict,
            DeepTrackDataObject,
            DeepTrackNode,
        )


    def test_DeepTrackDataObject(self):
        dataobj = core.DeepTrackDataObject()

        # Test default inititialization
        self.assertEqual(dataobj.current_value(), None)
        self.assertEqual(dataobj.is_valid(), False)

        # Test storing and validating data
        dataobj.store(1)
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), True)

        # Test invalidating data
        dataobj.invalidate()
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), False)

        # Test validating data
        dataobj.validate()
        self.assertEqual(dataobj.current_value(), 1)
        self.assertEqual(dataobj.is_valid(), True)

        # Test updating data
        dataobj.store(2)
        self.assertEqual(dataobj.current_value(), 2)
        self.assertEqual(dataobj.is_valid(), True)


    def test_DeepTrackDataDict(self):
        datadict = core.DeepTrackDataDict()

        # Test initial state
        self.assertEqual(datadict.keylength, None)
        self.assertFalse(datadict.dict)  # Empty dict, {}

        # Create indices and store data
        datadict.create_index((0, 0))
        datadict[(0, 0)].store({"image": [0, 0, 0], "label": (0, 0)})

        datadict.create_index((0, 1))
        datadict[(0, 1)].store({"image": [0, 1, 1], "label": (0, 1)})

        datadict.create_index((1, 0))
        datadict[(1, 0)].store({"image": [1, 0, 2], "label": (1, 0)})

        datadict.create_index((1, 1))
        datadict[(1, 1)].store({"image": [1, 1, 3], "label": (1, 1)})

        self.assertEqual(datadict.keylength, 2)
        self.assertEqual(len(datadict), 4)

        self.assertIn((0, 0), datadict.dict)
        self.assertIn((0, 0), datadict.keys())
        self.assertIn((0, 1), datadict.dict)
        self.assertIn((0, 1), datadict.keys())
        self.assertIn((1, 0), datadict.dict)
        self.assertIn((1, 0), datadict.keys())
        self.assertIn((1, 1), datadict.dict)
        self.assertIn((1, 1), datadict.keys())

        # Test retrieving stored data
        self.assertEqual(
            datadict[(0, 0)].current_value(),
            {"image": [0, 0, 0], "label": (0, 0)},
        )
        self.assertEqual(
            datadict[(0, 1)].current_value(),
            {"image": [0, 1, 1], "label": (0, 1)},
        )
        self.assertEqual(
            datadict[(1, 0)].current_value(),
            {"image": [1, 0, 2], "label": (1, 0)},
        )
        self.assertEqual(
            datadict[(1, 1)].current_value(),
            {"image": [1, 1, 3], "label": (1, 1)},
        )

        # Test validation and invalidation - all
        for key, value in datadict.items():
            self.assertTrue(value.is_valid())

        datadict.invalidate()
        for key, value in datadict.items():
            self.assertFalse(value.is_valid())

        datadict.validate()
        for key, value in datadict.items():
            self.assertTrue(value.is_valid())

        # Test validation and invalidation - single node
        self.assertTrue(datadict[(0, 0)].is_valid())

        datadict[(0, 0)].invalidate()
        for key, value in datadict.items():
            if key == (0, 0):
                self.assertFalse(value.is_valid())
            else:
                self.assertTrue(value.is_valid())

        datadict[(1, 1)].invalidate()
        for key, value in datadict.items():
            if key == (0, 0) or key == (1, 1):
                self.assertFalse(value.is_valid())
            else:
                self.assertTrue(value.is_valid())

        datadict[(0, 0)].validate()
        for key, value in datadict.items():
            if key == (1, 1):
                self.assertFalse(value.is_valid())
            else:
                self.assertTrue(value.is_valid())

        datadict[(1, 1)].validate()
        for key, value in datadict.items():
            self.assertTrue(value.is_valid())

        # Test valid_index
        self.assertFalse(datadict.valid_index(()))

        self.assertFalse(datadict.valid_index((0,)))

        self.assertTrue(datadict.valid_index((0, 0)))
        self.assertTrue(datadict.valid_index((1, 1)))
        self.assertTrue(datadict.valid_index((2, 2)))

        self.assertFalse(datadict.valid_index((0, 1, 2)))

        # Test slicing: __getitem__ with shorter _ID
        sliced = datadict[(0,)]
        self.assertIsInstance(sliced, dict)

        self.assertIn((0, 0), sliced)
        self.assertIsInstance(sliced[(0, 0)], core.DeepTrackDataObject)
        self.assertIn((0, 1), sliced)
        self.assertIsInstance(sliced[(0, 1)], core.DeepTrackDataObject)

        # Test trimming: __getitem__ with longer _ID
        for key, value in datadict.items():
            self.assertEqual(
                datadict[key + (99,)].current_value(),
                datadict[key].current_value(),
            )

        # Test items(), keys(), values()
        for item, key, value in zip(
            datadict.items(), datadict.keys(), datadict.values()
        ):
            self.assertEqual(item[0], key)
            self.assertEqual(item[1], value)

        # Test dict property access
        self.assertIs(datadict.dict[(0, 0)], datadict[(0, 0)])


    def test_DeepTrackNode_basics(self):
        ## Without _ID
        node = core.DeepTrackNode(action=lambda: 42)

        # Evaluate the node
        result = node()  # Value is calculated and stored.
        self.assertEqual(result, 42)

        # Store a value
        node.store(100)  # Value is stored.
        self.assertEqual(node.current_value(), 100)
        self.assertTrue(node.is_valid())

        # Invalidate the node and check the value
        node.invalidate()
        self.assertFalse(node.is_valid())

        self.assertEqual(node.current_value(), 100)  # Value is retrieved
        self.assertFalse(node.is_valid())

        self.assertEqual(node(), 42)  # Value is calculated and stored
        self.assertTrue(node.is_valid())

        ## With _ID
        node = core.DeepTrackNode(action=lambda _ID: _ID[0] * 10 + _ID[1])

        # Store values
        self.assertEqual(node((0, 0)), 0)
        self.assertEqual(node((0, 1)), 1)
        self.assertEqual(node((1, 0)), 10)
        self.assertEqual(node((1, 1)), 11)

        # Check validity
        self.assertFalse(node.is_valid())
        self.assertTrue(node.is_valid((0, 0)))
        self.assertTrue(node.is_valid((0, 1)))
        self.assertTrue(node.is_valid((1, 0)))
        self.assertTrue(node.is_valid((1, 1)))

        # Invalidate
        node.invalidate()
        self.assertFalse(node.is_valid((0, 0)))
        self.assertFalse(node.is_valid((0, 1)))
        self.assertFalse(node.is_valid((1, 0)))
        self.assertFalse(node.is_valid((1, 1)))

    def test_DeepTrackNode_dependencies(self):
        parent = core.DeepTrackNode(
            node_name="parent",
            action=lambda: 10,
        )
        child = core.DeepTrackNode(
            node_name="child",
            action=lambda: parent() * 2,
        )
        grandchild = core.DeepTrackNode(
            node_name="grandchild",
            action=lambda: child() * 3,
        )

        # Establish dependencies
        if random.randint(0, 1):  # Test add_child()
            parent.add_child(child)
        else:  # Test add_dependency()
            child.add_dependency(parent)

        if random.randint(0, 1):  # Test add_child()
            child.add_child(grandchild)
        else:  # Test add_dependency()
            grandchild.add_dependency(child)

        # Check that the just created nodes are invalid as not calculated
        self.assertFalse(parent.is_valid())
        self.assertFalse(child.is_valid())
        self.assertFalse(grandchild.is_valid())

        # Calculate child, and therefore parent.
        self.assertEqual(grandchild(), 60)
        self.assertTrue(parent.is_valid())
        self.assertTrue(child.is_valid())
        self.assertTrue(grandchild.is_valid())

        # Invalidate parent and check child validity.
        parent.invalidate()
        self.assertFalse(parent.is_valid())
        self.assertFalse(child.is_valid())
        self.assertFalse(grandchild.is_valid())

        # Validate parent and ensure child is invalid until recomputation.
        child.validate()
        self.assertFalse(parent.is_valid())
        self.assertTrue(child.is_valid())
        self.assertFalse(grandchild.is_valid())

        # Recompute child and check its validity
        grandchild()
        self.assertFalse(parent.is_valid())  # Not recalculated as child valid
        self.assertTrue(child.is_valid())
        self.assertTrue(grandchild.is_valid())

        # Recompute child and check its validity
        parent.invalidate()
        grandchild()
        self.assertTrue(parent.is_valid())
        self.assertTrue(child.is_valid())
        self.assertTrue(grandchild.is_valid())

        # Check dependencies
        self.assertEqual(len(parent.children), 1)
        for node in parent.children:
            self.assertEqual(node.node_name, "child")
        self.assertEqual(len(child.children), 1)
        for node in child.children:
            self.assertEqual(node.node_name, "grandchild")
        self.assertEqual(len(grandchild.children), 0)

        self.assertEqual(len(parent.dependencies), 0)
        self.assertEqual(len(child.dependencies), 1)
        for node in child.dependencies:
            self.assertEqual(node.node_name, "parent")
        self.assertEqual(len(grandchild.dependencies), 1)
        for node in grandchild.dependencies:
            self.assertEqual(node.node_name, "child")

        self.assertEqual(len(parent._all_children), 3)
        self.assertEqual(len(child._all_children), 2)
        self.assertEqual(len(grandchild._all_children), 1)

        self.assertEqual(len(parent.recurse_children()), 3)
        self.assertEqual(len(child.recurse_children()), 2)
        self.assertEqual(len(grandchild.recurse_children()), 1)

        self.assertEqual(len(parent.recurse_dependencies()), 1)
        self.assertEqual(len(child.recurse_dependencies()), 2)
        self.assertEqual(len(grandchild.recurse_dependencies()), 3)

    def test_DeepTrackNode_op_overloading(self):
        node1 = core.DeepTrackNode(action=lambda: 5)
        node2 = core.DeepTrackNode(action=lambda: 10)

        sum_node = node1 + node2
        self.assertEqual(sum_node(), 15)
        sum_node = node1 + 100
        self.assertEqual(sum_node(), 105)
        sum_node = 100 + node2
        self.assertEqual(sum_node(), 110)

        diff_node = node1 - node2
        self.assertEqual(diff_node(), -5)
        diff_node = node1 - 100
        self.assertEqual(diff_node(), -95)
        diff_node = 100 - node2
        self.assertEqual(diff_node(), 90)

        prod_node = node1 * node2
        self.assertEqual(prod_node(), 50)
        prod_node = node1 * 100
        self.assertEqual(prod_node(), 500)
        prod_node = 100 * node2
        self.assertEqual(prod_node(), 1_000)

        truediv_node = node2 / node1
        self.assertEqual(truediv_node(), 2)
        truediv_node = node2 / 2
        self.assertEqual(truediv_node(), 5)
        truediv_node = 50 / node1
        self.assertEqual(truediv_node(), 10)

        floordiv_node = node1 // node2
        self.assertEqual(floordiv_node(), 0)
        floordiv_node = node1 // 2
        self.assertEqual(floordiv_node(), 2)
        floordiv_node = 12 // node2
        self.assertEqual(floordiv_node(), 1)

        lt_node = node1 < node2
        self.assertTrue(lt_node())
        lt_node = node1 < 2
        self.assertFalse(lt_node())
        lt_node = 12 < node2
        self.assertFalse(lt_node())

        gt_node = node1 > node2
        self.assertFalse(gt_node())
        gt_node = node1 > 2
        self.assertTrue(gt_node())
        gt_node = 12 > node2
        self.assertTrue(gt_node())

        le_node = node1 < node2
        self.assertTrue(le_node())
        le_node = node1 < 2
        self.assertFalse(le_node())
        le_node = 12 < node2
        self.assertFalse(le_node())

        ge_node = node1 > node2
        self.assertFalse(ge_node())
        ge_node = node1 > 2
        self.assertTrue(ge_node())
        ge_node = 12 > node2
        self.assertTrue(ge_node())

    def test_DeepTrackNode_citations(self):
        node = core.DeepTrackNode(action=lambda: 42)
        citations = node.get_citations()
        self.assertIn(core.CITATION_MIDTVEDT2021QUANTITATIVE, citations)

    def test_DeepTrackNode_single_id(self):
        # Test a single _ID on a simple parent-child relationship.

        parent = core.DeepTrackNode(action=lambda: 10)
        child = core.DeepTrackNode(action=lambda _ID=None: parent(_ID) * 2)
        parent.add_child(child)

        # Store value for a specific _ID's.
        for id, value in enumerate(range(10)):
            parent.store(id, _ID=(id,))

        # Retrieves the values stored in children and parents.
        for id, value in enumerate(range(10)):
            self.assertEqual(child(_ID=(id,)), value * 2)
            self.assertEqual(parent.current_value((id,)), value)

    def test_DeepTrackNode_nested_ids(self):
        # Test nested IDs for parent-child relationships.

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
        # Test replicated behavior where IDs expand.

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
        # Test that invalidating a parent affects specific IDs of children.

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
        # Test a multi-level dependency graph with nested IDs.

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


    def test__equivalent(self):
        # Identity check (same object)
        a = [1, 2, 3]
        self.assertTrue(core._equivalent(a, a))

        # Both are empty lists (but not the same object)
        self.assertTrue(core._equivalent([], []))
        a, b = [], []
        self.assertTrue(core._equivalent(a, b))

        # Non-empty lists (not same object, not empty)
        self.assertFalse(core._equivalent([1], [1]))

        # Empty list and None
        self.assertFalse(core._equivalent([], None))

        # Different types
        self.assertFalse(core._equivalent(1, "1"))

        # Non-empty lists (same content, not same object)
        a = [1]
        b = [1]
        self.assertFalse(core._equivalent(a, b))

        # One empty list, one non-list empty container
        self.assertFalse(core._equivalent([], ()))


    def test__create_node_with_operator(self):
        import operator

        # Test with integers (should be wrapped automatically)
        node = core._create_node_with_operator(operator.add, 2, 3)
        self.assertIsInstance(node, core.DeepTrackNode)
        self.assertEqual(node(), 5)

        # Test with DeepTrackNode operands (addition)
        a = core.DeepTrackNode(lambda: 10)
        b = core.DeepTrackNode(lambda: 7)
        node2 = core._create_node_with_operator(operator.sub, a, b)
        self.assertIsInstance(node2, core.DeepTrackNode)
        self.assertEqual(node2(), 3)

        # node2 should be a child of both a and b
        self.assertIn(node2, a.children)
        self.assertIn(node2, b.children)

        # a and b should both be dependencies of node2
        self.assertIn(a, node2.dependencies)
        self.assertIn(b, node2.dependencies)

        # Test with one DeepTrackNode, one plain value (multiplication)
        node3 = core._create_node_with_operator(operator.mul, a, 2)
        self.assertEqual(node3(), 20)
        self.assertIsInstance(node3, core.DeepTrackNode)
        self.assertIn(node3, a.children)

        # Ensure wrapping of right operand
        node4 = core._create_node_with_operator(operator.mul, 3, b)
        self.assertEqual(node4(), 21)
        self.assertIsInstance(node4, core.DeepTrackNode)
        self.assertIn(node4, b.children)


if __name__ == "__main__":
    unittest.main()
