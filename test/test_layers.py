import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.layers as layers
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models
from tensorflow_addons.layers import InstanceNormalization

import numpy as np


def makeMinimalModel(layer, shape) -> k_models.Model:

    i = k_layers.Input(shape=shape)
    o = layer(i)

    return k_models.Model(i, o)


class TestModels(unittest.TestCase):
    def test_Dense(self):
        block = layers.DenseBlock()
        model = makeMinimalModel(block(1), shape=(2,))
        self.assertIsInstance(model.layers[1], k_layers.Dense)

    def test_Conv(self):
        block = layers.ConvolutionalBlock()
        model = makeMinimalModel(block(1), shape=(4, 4, 1))
        self.assertIsInstance(model.layers[1], k_layers.Conv2D)

    def test_Pool(self):
        block = layers.PoolingBlock()
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[1], k_layers.MaxPool2D)

    def test_Deconv(self):
        block = layers.DeconvolutionalBlock()
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[1], k_layers.Conv2DTranspose)

    def test_StaticUpsample(self):
        block = layers.StaticUpsampleBlock()
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[1], k_layers.UpSampling2D)

    def test_Residual(self):
        block = layers.ResidualBlock()
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertEqual(len(model.layers), 9)

    def test_Identity(self):
        block = layers.Identity()
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[1], k_layers.Layer)

    def test_NoneActivation(self):
        block = layers.Identity(activation=None)
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertEqual(len(model.layers), 2)

    def test_InstanceNorm(self):
        block = layers.Identity(instance_norm=True)
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[2], InstanceNormalization)


if __name__ == "__main__":
    unittest.main()