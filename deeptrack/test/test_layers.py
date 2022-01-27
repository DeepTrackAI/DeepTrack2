from pyexpat import model
import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import layers
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models
from ..layers import (
    InstanceNormalization,
    MultiHeadSelfAttention,
    MultiHeadGatedSelfAttention,
)

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
        block = layers.Identity(normalization="BatchNormalization")
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[2], k_layers.BatchNormalization)

    def test_Norm_as_tf_layer(self):
        block = layers.Identity(normalization=k_layers.BatchNormalization)
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[2], k_layers.BatchNormalization)

    def test_Norm_as_callable(self):
        block = layers.Identity(
            normalization=lambda axis, momentum, epsilon: k_layers.BatchNormalization(
                axis=axis, momentum=momentum, epsilon=epsilon
            ),
            norm_kwargs={"axis": 1, "momentum": 0.99, "epsilon": 0.001},
        )
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertIsInstance(model.layers[2], k_layers.BatchNormalization)

    def test_Norm_key_arguments(self):
        block = layers.Identity(
            normalization=lambda axis, momentum, epsilon: k_layers.BatchNormalization(
                axis=axis, momentum=momentum, epsilon=epsilon
            ),
            norm_kwargs={"axis": 1, "momentum": 0.95, "epsilon": 0.001},
        )
        model = makeMinimalModel(block(1), shape=(2, 2, 1))
        self.assertTrue(model.layers[2].momentum == 0.95)

    def test_Multi_Head_Attention(self):
        block = layers.MultiHeadSelfAttentionLayer()
        model = makeMinimalModel(block(1), shape=(100, 96))
        self.assertTrue(model.layers[1], MultiHeadSelfAttention)

    def test_Multi_Head_Attention_arguments(self):
        block = layers.MultiHeadSelfAttentionLayer(number_of_heads=6)
        model = makeMinimalModel(block(1), shape=(100, 96))
        self.assertEqual(model.layers[1].number_of_heads, 6)

    def test_Multi_Head_Attention_bias(self):
        block = layers.MultiHeadSelfAttentionLayer(use_bias=False)
        model = makeMinimalModel(block(1), shape=(100, 96))
        self.assertFalse(model.layers[1].key_dense.use_bias)

    def test_Multi_Head_Attention_filters(self):
        block = layers.MultiHeadGatedSelfAttentionLayer()
        model = makeMinimalModel(block(1), shape=(100, 96))
        self.assertEqual(model.layers[1].filters, 96)

    def test_Multi_Head_Gated_Attention(self):
        block = layers.MultiHeadGatedSelfAttentionLayer()
        model = makeMinimalModel(block(1), shape=(100, 96))
        self.assertTrue(model.layers[1], MultiHeadGatedSelfAttention)

    def test_Multi_Head_Gated_Attention_filters(self):
        block = layers.MultiHeadGatedSelfAttentionLayer()
        model = makeMinimalModel(block(1), shape=(100, 96))
        self.assertEqual(model.layers[1].filters, 96)


if __name__ == "__main__":
    unittest.main()