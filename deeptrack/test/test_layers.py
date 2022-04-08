from pyexpat import model
import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import layers
import tensorflow as tf
import tensorflow.keras.layers as k_layers
import tensorflow.keras.models as k_models

import numpy as np


def makeMinimalModel(
    layer, shape=(None, None, 1), input_layer=None, **kwargs
) -> k_models.Model:
    if input_layer is None:
        input_layer = k_layers.Input(shape=shape)

    o = layer(input_layer)

    return k_models.Model(input_layer, o)


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
        self.assertTrue(model.layers[1], layers.MultiHeadSelfAttention)

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
        self.assertTrue(model.layers[1], layers.MultiHeadGatedSelfAttention)

    def test_Multi_Head_Gated_Attention_filters(self):
        block = layers.MultiHeadGatedSelfAttentionLayer()
        model = makeMinimalModel(block(1), shape=(100, 96))
        self.assertEqual(model.layers[1].filters, 96)

    def test_FGNN_layer(self):
        block = layers.FGNNlayer()
        model = makeMinimalModel(
            block(96),
            input_layer=(
                k_layers.Input(shape=(None, 96)),
                k_layers.Input(shape=(None, 10)),
                k_layers.Input(shape=(None, 2), dtype=tf.int32),
                k_layers.Input(shape=(None, 1)),
                k_layers.Input(shape=(None, 2)),
            ),
        )
        self.assertTrue(model.layers[-1], layers.FGNN)

    def test_Class_Token_FGNN_layer(self):
        block = layers.ClassTokenFGNNlayer()
        model = makeMinimalModel(
            block(96),
            input_layer=(
                k_layers.Input(shape=(None, 96)),
                k_layers.Input(shape=(None, 10)),
                k_layers.Input(shape=(None, 2), dtype=tf.int32),
                k_layers.Input(shape=(None, 1)),
                k_layers.Input(shape=(None, 2)),
            ),
        )
        self.assertTrue(model.layers[-1], layers.ClassTokenFGNN)

    def test_Class_Token_FGNN_update_layer(self):
        block = layers.ClassTokenFGNNlayer(
            att_layer_kwargs={"number_of_heads": 6}
        )
        model = makeMinimalModel(
            block(96),
            input_layer=(
                k_layers.Input(shape=(None, 96)),
                k_layers.Input(shape=(None, 10)),
                k_layers.Input(shape=(None, 2), dtype=tf.int32),
                k_layers.Input(shape=(None, 1)),
                k_layers.Input(shape=(None, 2)),
            ),
        )
        self.assertEqual(
            model.layers[-1].update_layer.layers[0].number_of_heads, 6
        )

    def test_Class_Token_FGNN_normalization(self):
        # By setting center=False, scale=False, the number of trainable parameters should be 0
        block = layers.ClassTokenFGNNlayer(
            norm_kwargs={"center": False, "scale": False, "axis": -1}
        )
        model = makeMinimalModel(
            block(96),
            input_layer=(
                k_layers.Input(shape=(None, 96)),
                k_layers.Input(shape=(None, 10)),
                k_layers.Input(shape=(None, 2), dtype=tf.int32),
                k_layers.Input(shape=(None, 1)),
                k_layers.Input(shape=(None, 2)),
            ),
        )
        self.assertEqual(
            model.layers[-1].update_layer.layers[-1].count_params(), 0
        )

    def test_Transformer_Encoder(self):
        block = layers.TransformerEncoderLayer()
        model = makeMinimalModel(block(300), shape=(50, 300))
        self.assertTrue(model.layers[-1], layers.TransformerEncoder)

    def test_Tranformer_Encoder_parameters(self):
        block = layers.TransformerEncoderLayer(number_of_heads=6)
        model = makeMinimalModel(block(300), shape=(50, 300))
        self.assertEqual(model.layers[-1].MultiHeadAttLayer.number_of_heads, 6)

    def test_Transformer_Encoder_bias(self):
        block = layers.TransformerEncoderLayer(use_bias=True)
        model = makeMinimalModel(block(300), shape=(50, 300))
        self.assertTrue(
            model.layers[-1].MultiHeadAttLayer.key_dense.use_bias, True
        )


if __name__ == "__main__":
    unittest.main()
