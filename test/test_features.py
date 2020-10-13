import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.features as features

import numpy as np
from deeptrack.image import Image


class TestFeatures(unittest.TestCase):
    def test_Feature_1(self):
        class FeatureConcreteClass(features.Feature):
            __distributed__ = False

            def get(self, *args, **kwargs):
                image = np.ones((2, 3))
                return image

        feature = FeatureConcreteClass()
        feature.update()
        output_image = feature.resolve()
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image.size, 6)

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
        feature = feature1 + feature2
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
        feature12 = feature1 + feature2
        feature12.update()
        output_image12 = feature12.resolve(input_image)
        self.assertEqual(output_image12, 10)
        feature21 = feature2 + feature1
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
        feature12 = feature1 + feature2
        feature12.update()
        output_image = feature12.resolve()
        self.assertIsInstance(output_image, list)
        self.assertIsInstance(output_image[0], Image)
        self.assertIsInstance(output_image[1], Image)
        self.assertEqual(output_image[0].shape, (1, 1))
        self.assertEqual(output_image[1].shape, (2, 2))

    def test_Feature_times_1(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        input_image = np.zeros((1, 1))
        feature0 = FeatureAddValue(value_to_add=1) * 0
        feature0.update()
        output_image0 = feature0.resolve(input_image)
        self.assertEqual(output_image0, 0)
        feature1 = FeatureAddValue(value_to_add=1) * 1
        feature1.update()
        output_image1 = feature1.resolve(input_image)
        self.assertEqual(output_image1, 1)
        feature05 = FeatureAddValue(value_to_add=1) * 0.5
        for _ in range(100):
            feature05.update()
            output_image05 = feature05.resolve(input_image)
            self.assertTrue(output_image05[0, 0] == 0 or output_image05[0, 0] == 1)

    def test_Feature_exp_1(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        feature = FeatureAddValue(value_to_add=1) ** 10
        feature.update()
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertEqual(output_image, 10)

    def test_Feature_property_memorability(self):
        class FeatureWithForgettableProperties(features.Feature):
            __property_memorability__ = 2

            def get(self, image, forgettable_property, **kwargs):
                return image

        feature = FeatureWithForgettableProperties(forgettable_property=1)
        feature.update()
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertIsNone(
            output_image.get_property("forgettable_property", default=None)
        )
        output_image = feature.resolve(input_image, property_memorability=2)
        self.assertEqual(
            output_image.get_property("forgettable_property", default=None), 1
        )

    def test_Feature_with_overruled_property(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image

        feature = FeatureAddValue(value_to_add=1)
        feature.update()
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image, value_to_add=10)
        self.assertEqual(output_image, 10)
        self.assertListEqual(
            output_image.get_property("value_to_add", get_one=False), [10]
        )
        self.assertEqual(output_image.get_property("value_to_add", get_one=True), 10)

    def test_nested_Duplicate(self):
        for _ in range(5):
            A = features.DummyFeature(
                a=lambda: np.random.randint(100) * 1000,
            )
            B = features.DummyFeature(
                a=A.a,
                b=lambda a: a + np.random.randint(10) * 100,
            )
            C = features.DummyFeature(
                b=B.b,
                c=lambda b: b + np.random.randint(10) * 10,
            )
            D = features.DummyFeature(
                c=C.c,
                d=lambda c: c + np.random.randint(10) * 1,
            )

            for _ in range(5):
                AB = (A + (B + (C + D) ** 2) ** 2) ** 6
                output = AB.update().resolve(0)
                al = output.get_property("a", get_one=False)[::3]
                bl = output.get_property("b", get_one=False)[::3]
                cl = output.get_property("c", get_one=False)[::2]
                dl = output.get_property("d", get_one=False)[::1]

                self.assertFalse(all(a == al[0] for a in al))
                self.assertFalse(all(b == bl[0] for b in bl))
                self.assertFalse(all(c == cl[0] for c in cl))
                self.assertFalse(all(d == dl[0] for d in dl))
                for ai, a in enumerate(al):
                    for bi, b in list(enumerate(bl))[ai * 2 : (ai + 1) * 2]:
                        self.assertIn(b - a, range(0, 1000))
                        for ci, c in list(enumerate(cl))[bi * 2 : (bi + 1) * 2]:
                            self.assertIn(c - b, range(0, 100))
                            self.assertIn(dl[ci] - c, range(0, 10))

    def test_LambdaDependence(self):
        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a if key == "a" else (A.b if key == "b" else A.c),
        )

        B.update()
        self.assertEqual(B.prop.current_value, 1)
        B.update(key="a")
        self.assertEqual(B.prop.current_value, 1)
        B.update(key="b")
        self.assertEqual(B.prop.current_value, 2)
        B.update(key="c")
        self.assertEqual(B.prop.current_value, 3)

    def test_LambdaDependenceTwice(self):
        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a if key == "a" else (A.b if key == "b" else A.c),
            prop2=lambda prop: prop * 2,
        )

        B.update()
        self.assertEqual(B.prop2.current_value, 2)
        B.update(key="a")
        self.assertEqual(B.prop2.current_value, 2)
        B.update(key="b")
        self.assertEqual(B.prop2.current_value, 4)
        B.update(key="c")
        self.assertEqual(B.prop2.current_value, 6)

    def test_LambdaDependenceOtherFeature(self):
        A = features.DummyFeature(a=1, b=2, c=3)

        B = features.DummyFeature(
            key="a",
            prop=lambda key: A.a if key == "a" else (A.b if key == "b" else A.c),
            prop2=lambda prop: prop * 2,
        )

        C = features.DummyFeature(B_prop=B.prop2, prop=lambda B_prop: B_prop * 2)

        C.update()
        self.assertEqual(C.prop.current_value, 4)
        C.update(key="a")
        self.assertEqual(C.prop.current_value, 4)
        C.update(key="b")
        self.assertEqual(C.prop.current_value, 8)
        C.update(key="c")
        self.assertEqual(C.prop.current_value, 12)


if __name__ == "__main__":
    unittest.main()