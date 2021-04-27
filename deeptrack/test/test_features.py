import sys

# sys.path.append(".")  # Adds the module to path

import unittest
import operator
import itertools

from numpy.testing._private.utils import assert_almost_equal

from .. import features, Image, properties

import numpy as np
import numpy.testing


def grid_test_features(
    tester,
    feature_a,
    feature_b,
    feature_a_inputs,
    feature_b_inputs,
    expected_result_function,
    merge_operator=operator.rshift,
):

    assert callable(feature_a), "First feature constructor needs to be callable"
    assert callable(feature_b), "Second feature constructor needs to be callable"
    assert (
        len(feature_a_inputs) > 0 and len(feature_b_inputs) > 0
    ), "Feature input-lists cannot be empty"
    assert callable(expected_result_function), "Result function needs to be callable"

    for f_a_input, f_b_input in itertools.product(feature_a_inputs, feature_b_inputs):

        f_a = feature_a(**f_a_input)
        f_b = feature_b(**f_b_input)
        f = merge_operator(f_a, f_b)

        tester.assertIsInstance(f, features.Feature)

        try:
            output = f()
        except Exception as e:
            tester.assertRaises(
                type(e),
                lambda: expected_result_function(f_a.properties(), f_b.properties()),
            )
            continue

        expected_result = expected_result_function(
            f_a.properties(),
            f_b.properties(),
        )
        is_equal = np.array_equal(output, expected_result, equal_nan=True)

        tester.failIf(
            not is_equal,
            "Feature output {} is not equal to expect result {}.\n Using arguments \n\tFeature_1: {}, \n\t Feature_2: {}".format(
                output, expected_result, f_a_input, f_b_input
            ),
        )

        tester.failIf(
            not any(p == f_a.properties() for p in output.properties),
            "Feature_a properties {} not in output Image, with properties {}".format(
                f_a.properties(), output.properties
            ),
        )
        tester.failIf(
            not any(p == f_a.properties() for p in output.properties),
            "Feature_a properties {} not in output Image, with properties {}".format(
                f_a.properties(), output.properties
            ),
        )


def test_operator(self, operator):

    value = features.Value(value=2)
    f = operator(value, 3)
    self.assertEqual(f(), operator(2, 3))
    self.assertListEqual(f().get_property("value", get_one=False), [2, 3])

    f = operator(3, value)
    self.assertEqual(f(), operator(3, 2))

    f = operator(value, lambda: 3)
    self.assertEqual(f(), operator(2, 3))
    self.assertListEqual(f().get_property("value", get_one=False), [2, 3])

    grid_test_features(
        self,
        features.Value,
        features.Value,
        [
            {"value": 1},
            {"value": 0.5},
            {"value": np.nan},
            {"value": np.inf},
            {"value": np.random.rand(10, 10)},
        ],
        [
            {"value": 1},
            {"value": 0.5},
            {"value": np.nan},
            {"value": np.inf},
            {"value": np.random.rand(10, 10)},
        ],
        lambda a, b: operator(a["value"], b["value"]),
        operator,
    )


class TestFeatures(unittest.TestCase):
    def test_create_Feature(self):

        feature = features.DummyFeature()

        self.assertIsInstance(feature, features.Feature)
        self.assertIsInstance(feature.properties, properties.PropertyDict)

    def test_create_Feature_with_properties(self):
        feature = features.DummyFeature(prop_a="a", prop_2=2)

        self.assertIsInstance(feature, features.Feature)
        self.assertIsInstance(feature.properties, properties.PropertyDict)

        self.assertIsInstance(feature.properties["prop_a"](), str)
        self.assertEqual(feature.properties["prop_a"](), "a")

        self.assertIsInstance(feature.properties["prop_2"](), int)
        self.assertEqual(feature.properties["prop_2"](), 2)

    def test_Feature_properties_update(self):

        feature = features.DummyFeature(
            prop_a=lambda: np.random.rand(), prop_b="b", prop_c=iter(range(10))
        )

        start = feature.properties()

        self.assertIsInstance(start["prop_a"], float)
        self.assertIsInstance(start["prop_b"], str)
        self.assertIsInstance(start["prop_c"], int)

        without_update = feature.properties()
        self.assertDictEqual(start, without_update)

        feature.update()
        with_update = feature.properties()
        self.assertNotEqual(start, with_update)

    def test_Property_set_value_invalidates_feature(self):
        class ConcreteFeature(features.Feature):
            __distributed__ = False

            def get(self, input, **kwargs):
                return input

        feature = ConcreteFeature(prop=1)

        self.assertFalse(feature.is_valid())

        feature()
        self.assertTrue(feature.is_valid())

        feature.prop.set_value(1)
        self.assertTrue(feature.is_valid())

        feature.prop.set_value(2)
        self.assertFalse(feature.is_valid())

    def test_Feature_memoized(self):

        list_of_inputs = []

        class ConcreteFeature(features.Feature):
            __distributed__ = False

            def get(self, input, **kwargs):
                list_of_inputs.append(input)
                return input

        feature = ConcreteFeature(prop_a=1)

        feature()
        self.assertEqual(len(list_of_inputs), 1)
        feature()
        self.assertEqual(len(list_of_inputs), 1)
        feature.update()
        feature()
        self.assertEqual(len(list_of_inputs), 2)
        # Called with identical input

        feature.prop_a.set_value(1)
        feature()
        self.assertEqual(len(list_of_inputs), 2)

        feature.prop_a.set_value(2)
        feature()
        self.assertEqual(len(list_of_inputs), 3)

        feature([])
        self.assertEqual(len(list_of_inputs), 3)

        feature([1])
        self.assertEqual(len(list_of_inputs), 4)

    def test_Value(self):

        value = features.Value(value=1)
        self.assertEqual(value(), 1)
        self.assertEqual(value.value(), 1)

        value = features.Value(value=lambda: 1)
        self.assertEqual(value(), 1)
        self.assertEqual(value.value(), 1)

    def test_Feature_dependence(self):
        A = features.Value(lambda: np.random.rand())
        B = features.Value(value=A.value)
        C = features.Value(value=B.value + 1)
        D = features.Value(value=C.value + B.value)
        E = features.Value(value=D + C.value)

        self.assertEqual(A(), B())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        A.update()
        print(A.value.is_valid(), B.value.is_valid())
        self.assertEqual(A(), B())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        B.update()
        self.assertEqual(A(), B())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        C.update()
        self.assertEqual(A(), B())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        D.update()
        self.assertEqual(A(), B())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

        E.update()
        self.assertEqual(A(), B())
        self.assertEqual(C(), B() + 1)
        self.assertEqual(D(), C() + B())
        self.assertEqual(E(), D() + C())

    def test_Add(self):
        test_operator(self, operator.add)

    def test_Subtract(self):
        test_operator(self, operator.sub)

    def test_Multiply(self):
        test_operator(self, operator.add)

    def test_TrueDivide(self):
        test_operator(self, operator.truediv)

    def test_TrueDivide(self):
        test_operator(self, operator.floordiv)

    def test_Power(self):
        test_operator(self, operator.pow)

    def test_GreaterThan(self):
        test_operator(self, operator.gt)

    def test_GreaterThanOrEqual(self):
        test_operator(self, operator.ge)

    def test_LessThan(self):
        test_operator(self, operator.lt)

    def test_LessThanOrEqual(self):
        test_operator(self, operator.le)

    def test_Feature_2(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        feature = FeatureAddValue(value_to_add=1)
        feature.update()
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertEqual(output_image, 1)
        self.assertListEqual(
            output_image.get_property("value_to_add", get_one=False), [1]
        )
        output_image = feature.resolve(output_image)
        self.assertEqual(output_image, 2)
        self.assertListEqual(
            output_image.get_property("value_to_add", get_one=False), [1, 1]
        )

    def test_Feature_with_dummy_property(self):
        class FeatureConcreteClass(features.Feature):
            __distributed__ = False

            def get(self, *args, **kwargs):
                image = np.ones((2, 3))
                return image

        feature = FeatureConcreteClass(dummy_property="foo")
        feature.update()
        output_image = feature.resolve()
        self.assertListEqual(
            output_image.get_property("dummy_property", get_one=False), ["foo"]
        )

    def test_Feature_plus_1(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        feature1 = FeatureAddValue(value_to_add=1)
        feature2 = FeatureAddValue(value_to_add=2)
        feature = feature1 >> feature2
        feature.update()
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertEqual(output_image, 3)
        self.assertListEqual(
            output_image.get_property("value_to_add", get_one=False), [1, 2]
        )
        self.assertEqual(output_image.get_property("value_to_add", get_one=True), 1)

    def test_Feature_plus_2(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        class FeatureMultiplyByValue(features.Feature):
            def get(self, image, value_to_multiply=0, **kwargs):
                image = image * value_to_multiply
                return image

        feature1 = FeatureAddValue(value_to_add=1)
        feature2 = FeatureMultiplyByValue(value_to_multiply=10)
        input_image = np.zeros((1, 1))
        feature12 = feature1 >> feature2
        feature12.update()
        output_image12 = feature12.resolve(input_image)
        self.assertEqual(output_image12, 10)
        feature21 = feature2 >> feature1
        output_image21 = feature21.resolve(input_image)
        self.assertEqual(output_image21, 1)

    def test_Feature_plus_3(self):
        class FeatureAppendImageOfShape(features.Feature):
            __distributed__ = False
            __list_merge_strategy__ = features.MERGE_STRATEGY_APPEND

            def get(self, *args, shape, **kwargs):
                image = np.zeros(shape)
                return image

        feature1 = FeatureAppendImageOfShape(shape=(1, 1))
        feature2 = FeatureAppendImageOfShape(shape=(2, 2))
        feature12 = feature1 >> feature2
        feature12.update()
        output_image = feature12.resolve()
        self.assertIsInstance(output_image, list)
        self.assertIsInstance(output_image[0], Image)
        self.assertIsInstance(output_image[1], Image)
        self.assertEqual(output_image[0].shape, (1, 1))
        self.assertEqual(output_image[1].shape, (2, 2))

    def test_Feature_repeat(self):
        feature = features.Value(value=0) >> (features.Add(1) ^ iter(range(10)))

        for n in range(10):
            feature.update()
            output_image = feature()
            self.assertEqual(output_image, n)

    def test_Feature_repeat_random(self):
        feature = features.Value(value=0) >> (
            features.Add(value=lambda: np.random.randint(100)) ^ 100
        )

        feature.update()
        output_image = feature()
        values = output_image.get_property("value", get_one=False)[1:]

        num_dups = values.count(values[0])
        self.assertNotEqual(num_dups, len(values))
        self.assertEqual(output_image, sum(values))

    def test_Feature_repeat_nested(self):

        value = features.Value(0)
        add = features.Add(5)
        sub = features.Subtract(1)

        feature = value >> (((add ^ 2) >> (sub ^ 5)) ^ 3)

        self.assertEqual(feature(), 15)

    def test_Feature_repeat_nested_random_times(self):

        value = features.Value(0)
        add = features.Add(5)
        sub = features.Subtract(1)

        feature = value >> (
            ((add ^ 2) >> (sub ^ 5)) ^ (lambda: np.random.randint(2, 5))
        )

        for _ in range(5):
            feature.update()
            self.assertEqual(feature(), feature.feature_2.N() * 5)

    def test_Feature_repeat_nested_random_addition(self):

        value = features.Value(0)
        add = features.Add(lambda: np.random.rand())
        sub = features.Subtract(1)

        feature = value >> (((add ^ 2) >> (sub ^ 3)) ^ 4)

        feature.update()

        for _ in range(4):

            feature.update()

            added_values = list(
                map(
                    lambda f: f["value"],
                    filter(lambda f: f["name"] == "Add", feature().properties),
                )
            )
            self.assertEqual(len(added_values), 8)
            self.assertAlmostEqual(sum(added_values) - 3 * 4, feature())

        # print("OUT", added_values)
        # self.assertEqual(len(added_values), 6)
        # self.assertEqual(sum(added_values) - 15, feature())

    # def test_Feature_property_memorability(self):
    #     class FeatureWithForgettableProperties(features.Feature):
    #         __property_memorability__ = 2

    #         def get(self, image, forgettable_property, **kwargs):
    #             return image

    #     feature = FeatureWithForgettableProperties(forgettable_property=1)
    #     feature.update()
    #     input_image = np.zeros((1, 1))
    #     output_image = feature.resolve(input_image)
    #     self.assertIsNone(
    #         output_image.get_property("forgettable_property", default=None)
    #     )
    #     output_image = feature.resolve(input_image, property_memorability=2)
    #     self.assertEqual(
    #         output_image.get_property("forgettable_property", default=None), 1
    #     )

    # def test_Feature_with_overruled_property(self):
    #     class FeatureAddValue(features.Feature):
    #         def get(self, image, value_to_add=0, **kwargs):
    #             image = image + value_to_add
    #             return image

    #     feature = FeatureAddValue(value_to_add=1)
    #     feature.update()
    #     input_image = np.zeros((1, 1))
    #     output_image = feature.resolve(input_image, value_to_add=10)
    #     self.assertEqual(output_image, 10)
    #     self.assertListEqual(
    #         output_image.get_property("value_to_add", get_one=False), [10]
    #     )
    #     self.assertEqual(output_image.get_property("value_to_add", get_one=True), 10)

    # def test_nested_Duplicate(self):
    #     for _ in range(5):
    #         A = features.DummyFeature(
    #             a=lambda: np.random.randint(100) * 1000,
    #         )
    #         B = features.DummyFeature(
    #             a=A.a,
    #             b=lambda a: a + np.random.randint(10) * 100,
    #         )
    #         C = features.DummyFeature(
    #             b=B.b,
    #             c=lambda b: b + np.random.randint(10) * 10,
    #         )
    #         D = features.DummyFeature(
    #             c=C.c,
    #             d=lambda c: c + np.random.randint(10) * 1,
    #         )

    #         for _ in range(5):
    #             AB = (A + (B + (C + D) ** 2) ** 2) ** 6
    #             output = AB.update().resolve(0)
    #             al = output.get_property("a", get_one=False)[::3]
    #             bl = output.get_property("b", get_one=False)[::3]
    #             cl = output.get_property("c", get_one=False)[::2]
    #             dl = output.get_property("d", get_one=False)[::1]

    #             self.assertFalse(all(a == al[0] for a in al))
    #             self.assertFalse(all(b == bl[0] for b in bl))
    #             self.assertFalse(all(c == cl[0] for c in cl))
    #             self.assertFalse(all(d == dl[0] for d in dl))
    #             for ai, a in enumerate(al):
    #                 for bi, b in list(enumerate(bl))[ai * 2 : (ai + 1) * 2]:
    #                     self.assertIn(b - a, range(0, 1000))
    #                     for ci, c in list(enumerate(cl))[bi * 2 : (bi + 1) * 2]:
    #                         self.assertIn(c - b, range(0, 100))
    #                         self.assertIn(dl[ci] - c, range(0, 10))

    # def test_LambdaDependence(self):
    #     A = features.DummyFeature(a=1, b=2, c=3)

    #     B = features.DummyFeature(
    #         key="a",
    #         prop=lambda key: A.a if key == "a" else (A.b if key == "b" else A.c),
    #     )

    #     B.update()
    #     self.assertEqual(B.prop.current_value, 1)
    #     B.update(key="a")
    #     self.assertEqual(B.prop.current_value, 1)
    #     B.update(key="b")
    #     self.assertEqual(B.prop.current_value, 2)
    #     B.update(key="c")
    #     self.assertEqual(B.prop.current_value, 3)

    # def test_LambdaDependenceTwice(self):
    #     A = features.DummyFeature(a=1, b=2, c=3)

    #     B = features.DummyFeature(
    #         key="a",
    #         prop=lambda key: A.a if key == "a" else (A.b if key == "b" else A.c),
    #         prop2=lambda prop: prop * 2,
    #     )

    #     B.update()
    #     self.assertEqual(B.prop2.current_value, 2)
    #     B.update(key="a")
    #     self.assertEqual(B.prop2.current_value, 2)
    #     B.update(key="b")
    #     self.assertEqual(B.prop2.current_value, 4)
    #     B.update(key="c")
    #     self.assertEqual(B.prop2.current_value, 6)

    # def test_LambdaDependenceOtherFeature(self):
    #     A = features.DummyFeature(a=1, b=2, c=3)

    #     B = features.DummyFeature(
    #         key="a",
    #         prop=lambda key: A.a if key == "a" else (A.b if key == "b" else A.c),
    #         prop2=lambda prop: prop * 2,
    #     )

    #     C = features.DummyFeature(B_prop=B.prop2, prop=lambda B_prop: B_prop * 2)

    #     C.update()
    #     self.assertEqual(C.prop.current_value, 4)
    #     C.update(key="a")
    #     self.assertEqual(C.prop.current_value, 4)
    #     C.update(key="b")
    #     self.assertEqual(C.prop.current_value, 8)
    #     C.update(key="c")
    #     self.assertEqual(C.prop.current_value, 12)

    # def test_SliceConstant(self):

    #     input = np.arange(9).reshape((3, 3))

    #     A = features.DummyFeature()

    #     A0 = A[0]
    #     A1 = A[1]
    #     A22 = A[2, 2]
    #     A12 = A[1, lambda: -1]

    #     a0 = A0.resolve(input)
    #     a1 = A1.resolve(input)
    #     a22 = A22.resolve(input)
    #     a12 = A12.resolve(input)

    #     self.assertEqual(a0.tolist(), input[0].tolist())
    #     self.assertEqual(a1.tolist(), input[1].tolist())
    #     self.assertEqual(a22, input[2, 2])
    #     self.assertEqual(a12, input[1, -1])

    # def test_SliceColon(self):

    #     input = np.arange(16).reshape((4, 4))

    #     A = features.DummyFeature()

    #     A0 = A[0, :1]
    #     A1 = A[
    #         1,
    #         lambda: 0:
    #         lambda: 4:
    #         lambda: 2
    #     ]
    #     A2 = A[lambda: slice(0, 4, 1), 2]
    #     A3 = A[lambda: 0 :
    #            lambda: 2,
    #            :]

    #     a0 = A0.resolve(input)
    #     a1 = A1.resolve(input)
    #     a2 = A2.resolve(input)
    #     a3 = A3.resolve(input)

    #     self.assertEqual(a0.tolist(), input[0, :1].tolist())
    #     self.assertEqual(a1.tolist(), input[1, 0:4:2].tolist())
    #     self.assertEqual(a2.tolist(), input[:, 2].tolist())
    #     self.assertEqual(a3.tolist(), input[0:2, :].tolist())

    # def test_SliceEllipse(self):

    #     input = np.arange(16).reshape((4, 4))

    #     A = features.DummyFeature()

    #     A0 = A[..., :1]
    #     A1 = A[
    #         ...,
    #         lambda: 0:
    #         lambda: 4:
    #         lambda: 2
    #     ]
    #     A2 = A[lambda: slice(0, 4, 1), ...]
    #     A3 = A[lambda: 0 :
    #            lambda: 2,
    #            lambda: ...]

    #     a0 = A0.resolve(input)
    #     a1 = A1.resolve(input)
    #     a2 = A2.resolve(input)
    #     a3 = A3.resolve(input)

    #     self.assertEqual(a0.tolist(), input[..., :1].tolist())
    #     self.assertEqual(a1.tolist(), input[..., 0:4:2].tolist())
    #     self.assertEqual(a2.tolist(), input[:, ...].tolist())
    #     self.assertEqual(a3.tolist(), input[0:2, ...].tolist())


if __name__ == "__main__":
    unittest.main()
