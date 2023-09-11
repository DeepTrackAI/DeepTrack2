import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import models, layers
import numpy as np

import tensorflow as tf


class TestModels(unittest.TestCase):
    def test_FullyConnected(self):
        model = models.FullyConnected(
            input_shape=(64, 2),
            dense_layers_dimensions=(32, 32),
            number_of_outputs=3,
            output_activation="sigmoid",
            loss="mse",
            flatten_input=True,
        )
        self.assertIsInstance(model, models.KerasModel)

        model.predict(np.zeros((1, 64, 2)))

    def test_Convolutions_with_single_input(self):
        model = models.Convolutional(
            input_shape=(64, 64, 1),
            conv_layers_dimensions=(16, 32, 64),
            dense_layers_dimensions=(32, 32),
            number_of_outputs=3,
            output_activation="sigmoid",
            loss="mse",
        )
        self.assertIsInstance(model, models.KerasModel)

        model.predict(np.zeros((1, 64, 64, 1)))

    def test_Convolutions_with_multiple_inputs(self):
        model = models.Convolutional(
            input_shape=[(64, 64, 1), (64, 64, 2)],
            conv_layers_dimensions=(16, 32, 64),
            dense_layers_dimensions=(32, 32),
            number_of_outputs=3,
            output_activation="sigmoid",
            loss="mse",
        )
        self.assertIsInstance(model, models.KerasModel)

        model.predict([np.zeros((1, 64, 64, 1)), np.zeros((1, 64, 64, 2))])

    def test_Convolutions_with_no_dense_top(self):
        model = models.Convolutional(
            input_shape=(64, 64, 1),
            conv_layers_dimensions=(16, 32, 64),
            dense_layers_dimensions=(32, 32),
            dense_top=False,
            number_of_outputs=3,
            output_activation="sigmoid",
            loss="mse",
        )
        self.assertIsInstance(model, models.KerasModel)

        model.predict(np.zeros((1, 64, 64, 1)))

    def test_UNet(self):
        model = models.UNet(
            input_shape=(64, 64, 1),
            conv_layers_dimensions=(16, 32, 64),
            base_conv_layers_dimensions=(256, 256),
            output_conv_layers_dimensions=(32, 32),
            steps_per_pooling=1,
            number_of_outputs=1,
            output_activation="sigmoid",
            loss="mse",
        )
        self.assertIsInstance(model, models.KerasModel)

        model.predict(np.zeros((1, 64, 64, 1)))

    def test_GAN(self):
        model = models.GAN(
            discriminator=None,
            generator=None,
            latent_dim=128,
        )
        self.assertIsInstance(model.discriminator, tf.keras.Sequential)
        self.assertIsInstance(model.generator, tf.keras.Sequential)

        prediction = model.predict(np.zeros((1, 64, 64, 3)))
        self.assertEqual(prediction.shape, (1, 64, 64, 3))

    def test_VAE(self):
        model = models.VAE(
            encoder=None,
            decoder=None,
            latent_dim=2,
        )
        self.assertIsInstance(model.encoder, tf.keras.Sequential)
        self.assertIsInstance(model.decoder, tf.keras.Sequential)

        pred_enc = model.encoder.predict(np.zeros((1, 28, 28, 1)))
        self.assertEqual(pred_enc.shape, (1, 4))
        
        pred_dec = model.decoder.predict(np.zeros((1,2)))
        self.assertEqual(pred_dec.shape, (1, 28, 28, 1))


    def test_WAE(self):
        model = models.WAE(
            regularizer = 'mmd',
            encoder=None,
            decoder=None,
            discriminator=None,
            latent_dim=2,
            lambda_=10.0,
            sigma_z=1.0,
        )
        self.assertIsInstance(model.encoder, tf.keras.Sequential)
        self.assertIsInstance(model.decoder, tf.keras.Sequential)

        pred_enc = model.encoder.predict(np.zeros((1, 28, 28, 1)))
        self.assertEqual(pred_enc.shape, (1, 2))
        
        pred_dec = model.decoder.predict(pred_enc)
        self.assertEqual(pred_dec.shape, (1, 28, 28, 1))

    def test_RNN(self):
        model = models.rnn(
            input_shape=(None, 64, 64, 1),
            conv_layers_dimensions=(16, 32, 64),
            dense_layers_dimensions=(32,),
            rnn_layers_dimensions=(32,),
            steps_per_pooling=1,
            number_of_outputs=1,
            output_activation="sigmoid",
            loss="mse",
        )
        self.assertIsInstance(model, models.KerasModel)

        model.predict(np.zeros((1, 1, 64, 64, 1)))

    def test_ViT(self):
        model = models.ViT(
            input_shape=(224, 224, 3),
            patch_shape=16,
            num_layers=12,
            hidden_size=768,
            number_of_heads=12,
            fwd_mlp_dim=3072,
            dropout=0.1,
            representation_size=None,
            include_top=True,
            output_size=1000,
            output_activation="linear",
        )
        self.assertIsInstance(model, models.KerasModel)

        model.predict(np.zeros((1, 224, 224, 3)))

    def test_MAGIK(self):
        model = models.MAGIK(
            dense_layer_dimensions=(
                64,
                96,
            ),  # number of features in each dense encoder layer
            base_layer_dimensions=(
                96,
                96,
                96,
            ),  # Latent dimension throughout the message passing layers
            number_of_node_features=7,  # Number of node features in the graphs
            number_of_edge_features=1,  # Number of edge features in the graphs
            number_of_edge_outputs=1,  # Number of predicted features
            edge_output_activation="sigmoid",  # Activation function for the output layer
            output_type="edges",
        )

        self.assertIsInstance(model, models.KerasModel)

        graph = (
            tf.random.uniform((8, 10, 7)),  # Node features
            tf.random.uniform((8, 50, 1)),  # Edge features
            tf.random.uniform((8, 50, 2), minval=0, maxval=10, dtype=tf.int32),  # Edges
            tf.random.uniform((8, 50, 2)),  # Edge dropouts
        )
        model(graph)

    def test_CTMAGIK(self):
        model = models.CTMAGIK(
            dense_layer_dimensions=(
                32,
                64,
                96,
            ),  # number of features in each dense encoder layer
            base_layer_dimensions=(
                96,
                96,
            ),  # Latent dimension throughout the message passing layers
            number_of_node_features=7,  # Number of node features in the graphs
            number_of_edge_features=1,  # Number of edge features in the graphs
            number_of_global_outputs=1,  # Number of predicted features
            global_output_activation="softmax",  # Activation function for the output layer
            output_type="global",
        )

        self.assertIsInstance(model, models.KerasModel)

        graph = (
            tf.random.uniform((8, 10, 7)),  # Node features
            tf.random.uniform((8, 50, 1)),  # Edge features
            tf.random.uniform((8, 50, 2), minval=0, maxval=10, dtype=tf.int32),  # Edges
            tf.random.uniform((8, 50, 2)),  # Edge dropouts
        )
        prediction = model(graph)

        self.assertEqual(prediction.shape, (8, 1))

    def test_MAGIK_with_MaskedFGNN(self):
        model = models.MAGIK(
            dense_layer_dimensions=(
                64,
                96,
            ),  # number of features in each dense encoder layer
            base_layer_dimensions=(
                96,
                96,
                96,
            ),  # Latent dimension throughout the message passing layers
            number_of_node_features=7,  # Number of node features in the graphs
            number_of_edge_features=1,  # Number of edge features in the graphs
            number_of_edge_outputs=1,  # Number of predicted features
            edge_output_activation="sigmoid",  # Activation function for the output layer
            output_type="edges",
            graph_block="MaskedFGNN",
        )

        self.assertIsInstance(model, models.KerasModel)
        self.assertIsInstance(model.layers[15], layers.MaskedFGNN)

        graph = (
            tf.random.uniform((8, 10, 7)),  # Node features
            tf.random.uniform((8, 50, 1)),  # Edge features
            tf.random.uniform((8, 50, 2), minval=0, maxval=10, dtype=tf.int32),  # Edges
            tf.random.uniform((8, 50, 2)),  # Edge dropouts
        )
        model(graph)

    def test_MPGNN(self):
        model = models.MPNGNN(
            dense_layer_dimensions=(
                64,
                96,
            ),  # number of features in each dense encoder layer
            base_layer_dimensions=(
                96,
                96,
                96,
            ),  # Latent dimension throughout the message passing layers
            number_of_node_features=7,  # Number of node features in the graphs
            number_of_edge_features=1,  # Number of edge features in the graphs
            number_of_edge_outputs=1,  # Number of predicted features
            edge_output_activation="sigmoid",  # Activation function for the output layer
            output_type="edges",
        )

        self.assertIsInstance(model, models.KerasModel)

        graph = (
            tf.random.uniform((8, 10, 7)),  # Node features
            tf.random.uniform((8, 50, 1)),  # Edge features
            tf.random.uniform((8, 50, 2), minval=0, maxval=10, dtype=tf.int32),  # Edges
            tf.random.uniform((8, 50, 2)),  # Edge dropouts
        )
        model(graph)

    def test_MPGNN_readout(self):
        model = models.MPNGNN(
            dense_layer_dimensions=(
                32,
                64,
                96,
            ),  # number of features in each dense encoder layer
            base_layer_dimensions=(
                96,
                96,
            ),  # Latent dimension throughout the message passing layers
            number_of_node_features=7,  # Number of node features in the graphs
            number_of_edge_features=1,  # Number of edge features in the graphs
            number_of_global_outputs=1,  # Number of predicted features
            global_output_activation="softmax",  # Activation function for the output layer
            output_type="global",
            readout_block=tf.keras.layers.GlobalAveragePooling1D(),
        )

        self.assertIsInstance(model, models.KerasModel)
        self.assertIsInstance(model.layers[-4], tf.keras.layers.GlobalAveragePooling1D)

        graph = (
            tf.random.uniform((8, 10, 7)),  # Node features
            tf.random.uniform((8, 50, 1)),  # Edge features
            tf.random.uniform((8, 50, 2), minval=0, maxval=10, dtype=tf.int32),  # Edges
            tf.random.uniform((8, 50, 2)),  # Edge dropouts
        )
        prediction = model(graph)

        self.assertEqual(prediction.shape, (8, 1))

    def test_GRU_MPGNN(self):
        model = models.MPNGNN(
            dense_layer_dimensions=(
                64,
                96,
            ),  # number of features in each dense encoder layer
            base_layer_dimensions=(
                96,
                96,
                96,
            ),  # Latent dimension throughout the message passing layers
            number_of_node_features=7,  # Number of node features in the graphs
            number_of_edge_features=1,  # Number of edge features in the graphs
            number_of_edge_outputs=1,  # Number of predicted features
            edge_output_activation="sigmoid",  # Activation function for the output layer
            output_type="edges",
            graph_block="GRUMPN",
        )

        self.assertIsInstance(model, models.KerasModel)

        graph = (
            tf.random.uniform((8, 10, 7)),  # Node features
            tf.random.uniform((8, 50, 1)),  # Edge features
            tf.random.uniform((8, 50, 2), minval=0, maxval=10, dtype=tf.int32),  # Edges
            tf.random.uniform((8, 50, 2)),  # Edge dropouts
        )
        model(graph)


if __name__ == "__main__":
    unittest.main()
