import sys
sys.path.append("..") # Adds the module to path

import unittest

import deeptrack.augmentations as augmentations

from deeptrack.features import Feature
import numpy as np



class TestAugmentations(unittest.TestCase):
    
    class DummyFeature(Feature):
        __distributed__ = False
        def get(self, image, **kwargs):
            output = np.array([
                [[1], [2]], 
                [[0], [0]]
                ])
            return output


    def test_FlipLR(self):
        feature = self.DummyFeature()
        augmented_feature = augmentations.FlipLR(feature)
        augmented_feature.update()
        output_1 = augmented_feature.resolve()
        augmented_feature.update()
        output_2 = augmented_feature.resolve()
        self.assertTrue(np.all(output_1 == np.array([[[1], [2]], [[0], [0]]])))
        self.assertTrue(np.all(output_2 == np.array([[[2], [1]], [[0], [0]]])))        


    def test_FlipUD(self):
        feature = self.DummyFeature()
        augmented_feature = augmentations.FlipUD(feature)
        augmented_feature.update()
        output_1 = augmented_feature.resolve()
        augmented_feature.update()
        output_2 = augmented_feature.resolve()
        self.assertTrue(np.all(output_1 == np.array([[[1], [2]], [[0], [0]]])))
        self.assertTrue(np.all(output_2 == np.array([[[0], [0]], [[1], [2]]])))        


    def test_FlipDiagonal(self):
        feature = self.DummyFeature()
        augmented_feature = augmentations.FlipDiagonal(feature)
        augmented_feature.update()
        output_1 = augmented_feature.resolve()
        augmented_feature.update()
        output_2 = augmented_feature.resolve()
        self.assertTrue(np.all(output_1 == np.array([[[1], [2]], [[0], [0]]])))
        self.assertTrue(np.all(output_2 == np.array([[[1], [0]], [[2], [0]]])))        



if __name__ == '__main__':
    unittest.main()