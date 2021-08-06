from .utils import KerasModel
from .convolutional import Convolutional, UNet
from ..generators import AutoTrackGenerator
from ..losses import (
    rotational_consistency,
    squared_affine_consistency,
    squared_field_affine_consistency,
    size_consistency,
)
from ..layers import ConvolutionalBlock, PoolingBlock, DeconvolutionalBlock
from ..augmentations import Affine

import numpy as np

try:
    import tensorflow_addons as tfa

    TFA_INSTALLED = True
except:
    TFA_INSTALLED = False


class AutoTracker(KerasModel):
    def __init__(
        self,
        model=None,
        input_shape=(64, 64, 1),
        loss="auto",
        symmetries=1,
        mode="tracking",
        **kwargs
    ):
        self.symmetries = symmetries
        self.mode = mode

        if loss == "auto":
            if mode == "tracking":
                loss = squared_affine_consistency
            elif mode == "orientation":
                loss = rotational_consistency
            elif mode == "sizing":
                loss = size_consistency
            else:
                raise ValueError(
                    "Unknown mode provided to the auto tracker. Valid modes are 'tracking', 'orientation' and 'sizing'"
                )

        if model is None:
            model = self.default_model(input_shape=input_shape)

        super().__init__(model, loss=loss, **kwargs)

    def default_model(self, input_shape):

        return Convolutional(
            input_shape=input_shape,
            conv_layers_dimensions=[32, 64, 128],
            dense_layers_dimensions=(32, 32),
            steps_per_pooling=1,
            number_of_outputs=2,
        )

    def data_generator(self, *args, **kwargs):

        transformation_function = None
        if self.mode == "tracking":
            transformation_function = Affine(
                translate=lambda: np.random.randn(2) * 2,
                scale=lambda: np.random.rand() * 0.1 + 0.95,
                rotate=lambda: np.random.rand() * np.pi * 2,
            )
        elif self.mode == "orientation":
            transformation_function = Affine(
                translate=lambda: np.random.randn(2) * 2,
                scale=lambda: np.random.rand() * 0.1 + 0.95,
                rotate=lambda: np.random.rand() * np.pi * 2,
            )
        elif self.mode == "sizing":
            transformation_function = Affine(
                translate=lambda: np.random.randn(2) * 2,
                scale=lambda: np.random.rand() * 1 + 0.7,
                rotate=lambda: np.random.rand() * np.pi * 2,
            )

        return AutoTrackGenerator(
            transformation_function,
            *args,
            symmetries=self.symmetries if self.mode == "orientation" else 1,
            **kwargs
        )
