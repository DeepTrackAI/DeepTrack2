import sys

sys.path.append(".")  # Adds the module to path

import unittest

import deeptrack.losses as losses

import numpy as np
from keras import backend as K


class TestLosses(unittest.TestCase):

    truthly = K.constant(np.ones((2, 2, 1)))
    falsely = K.constant(np.zeros((2, 2, 1)))

    def test_flatten(self):
        from keras.losses import mse

        loss_function = losses.flatten(mse)
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 1, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 1, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 0, 3)

    def test_weighted_crossentropy(self):
        loss_function = losses.flatten(losses.weighted_crossentropy(weight=(10, 1)))
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 8.373037, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 0.8373037, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 0, 3)

    def test_nd_mean_squared_error(self):
        loss_function = losses.nd_mean_squared_error
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 1, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 1, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 0, 3)

    def test_nd_mean_squared_logarithmic_error(self):
        loss_function = losses.nd_mean_squared_logarithmic_error
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 0.48045287, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 0.48045287, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 0, 3)

    def test_nd_poisson(self):
        loss_function = losses.nd_poisson
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0.9999999, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 16.118095, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 1, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 0, 3)

    def test_nd_squared_hinge(self):
        loss_function = losses.nd_squared_hinge
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 1, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 4, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 1, 3)

    def test_nd_binary_crossentropy(self):
        loss_function = losses.nd_binary_crossentropy
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 15.424949, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 15.333239, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 0, 3)

    def test_nd_mean_absolute_percentage_error(self):
        loss_function = losses.nd_mean_absolute_percentage_error
        loss = K.eval(loss_function(self.truthly, self.truthly))
        self.assertAlmostEqual(loss, 0, 3)
        loss = K.eval(loss_function(self.truthly, self.falsely))
        self.assertAlmostEqual(loss, 100, 3)
        loss = K.eval(loss_function(self.falsely, self.truthly))
        self.assertAlmostEqual(loss, 1000000000, 3)
        loss = K.eval(loss_function(self.falsely, self.falsely))
        self.assertAlmostEqual(loss, 0, 3)


if __name__ == "__main__":
    unittest.main()