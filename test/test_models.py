import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.models as models
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
            conv_layers_dimensions=(16, 32, 64, 128),
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
            conv_layers_dimensions=(16, 32, 64, 128),
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
            conv_layers_dimensions=(16, 32, 64, 128),
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
            conv_layers_dimensions=(16, 32, 64, 128),
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


if __name__ == "__main__":
    unittest.main()