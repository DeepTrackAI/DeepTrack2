import sys

# sys.path.append(".")  # Adds the module to path

import unittest

from .. import models
import numpy as np


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


if __name__ == "__main__":
    unittest.main()