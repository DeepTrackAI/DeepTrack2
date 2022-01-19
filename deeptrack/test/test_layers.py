import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import layers
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models
import tensorflow_addons as tfa
from ..layers import InstanceNormalization

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

    def test_Norm_as_string(self):
        block = layers.Identity(normalization="InstanceNormalization")
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[2], InstanceNormalization)

    def test_Norm_as_tf_layer(self):
        block = layers.Identity(normalization=k_layers.BatchNormalization)
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[2], k_layers.BatchNormalization)

    def test_Norm_as_callable(self):
        block = layers.Identity(
            normalization=lambda axis, center, scale: tfa.layers.InstanceNormalization(
                axis=axis, center=center, scale=scale
            ),
            norm_kwargs={"axis": -1, "center": False, "scale": False},
        )
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[2], InstanceNormalization)

    # Instance Normalization with no learnable parameters. This is a special case where
    # center and scale are False and do not update during training.
    def test_Norm_key_arguments(self):
        block = layers.Identity(
            normalization=lambda axis, center, scale: tfa.layers.InstanceNormalization(
                axis=axis, center=center, scale=scale
            ),
            norm_kwargs={"axis": -1, "center": False, "scale": False},
        )
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertTrue(model.layers[2].count_params() == 0)


if __name__ == "__main__":
    unittest.main()