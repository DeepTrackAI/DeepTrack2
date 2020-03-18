import sys
sys.path.append("..") # Adds the module to path

import unittest

import deeptrack.features as features

import numpy as np
from deeptrack.image import Image



class TestUtils(unittest.TestCase):

    def test_Feature_1(self):
        class FeatureConcreteClass(features.Feature):
            __distributed__ = False
            def get(self, *args, **kwargs):
                image = np.ones((2, 3))
                return image    
        feature = FeatureConcreteClass()
        output_image = feature.resolve()
        self.assertIsInstance(output_image, Image)
        self.assertEqual(output_image.size, 6)
        
    
    def test_Feature_2(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image
        feature = FeatureAddValue(value_to_add=1)
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertEqual(output_image, 1)
        self.assertListEqual(output_image.get_property("value_to_add", get_one=False), [1])
        output_image = feature.resolve(output_image)
        self.assertEqual(output_image, 2)
        self.assertListEqual(output_image.get_property("value_to_add", get_one=False), [1, 1])


    def test_Feature_with_dummy_property(self):
        class FeatureConcreteClass(features.Feature):
            __distributed__ = False
            def get(self, *args, **kwargs):
                image = np.ones((2, 3))
                return image    
        feature = FeatureConcreteClass(dummy_property="foo")
        output_image = feature.resolve()
        self.assertListEqual(output_image.get_property("dummy_property", get_one=False), ["foo"])


    def test_Feature_plus_1(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image
        feature1 = FeatureAddValue(value_to_add=1)
        feature2 = FeatureAddValue(value_to_add=2)
        feature = feature1 + feature2
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertEqual(output_image, 3)
        self.assertListEqual(output_image.get_property("value_to_add", get_one=False), [1, 2])
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
        output_image0 = feature0.resolve(input_image)
        self.assertEqual(output_image0, 0)
        feature1 = FeatureAddValue(value_to_add=1) * 1
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
        feature = FeatureAddValue(value_to_add=1)
        input_image = np.zeros((1, 1))
        output_image = (feature**10).resolve(input_image)
        self.assertEqual(output_image, 10)
    

    def test_Feature_property_memorability(self):
        class FeatureWithForgettableProperties(features.Feature):
            __property_memorability__ = 2
            def get(self, image, forgettable_property, **kwargs):
                return image
        feature = FeatureWithForgettableProperties(forgettable_property=1)
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image)
        self.assertIsNone(output_image.get_property("forgettable_property", default=None))
        output_image = feature.resolve(input_image, property_memorability=2)
        self.assertEqual(output_image.get_property("forgettable_property", default=None), 1)

    
    def test_Feature_with_overruled_property(self):
        class FeatureAddValue(features.Feature):
            def get(self, image, value_to_add=0, **kwargs):
                image = image + value_to_add
                return image
        feature = FeatureAddValue(value_to_add=1)
        input_image = np.zeros((1, 1))
        output_image = feature.resolve(input_image, value_to_add=10)
        self.assertEqual(output_image, 10)
        self.assertListEqual(output_image.get_property("value_to_add", get_one=False), [10])
        self.assertEqual(output_image.get_property("value_to_add", get_one=True), 10)



if __name__ == '__main__':
    unittest.main()