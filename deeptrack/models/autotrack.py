from .utils import KerasModel
from .convolutional import Convolutional, UNet
from ..generators import AutoTrackGenerator
from ..losses import (
    squared_affine_consistency,
    squared_field_affine_consistency,
    adjacency_consistency,
)
from ..layers import ConvolutionalBlock, PoolingBlock, DeconvolutionalBlock

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
        loss=squared_affine_consistency,
        symmetries=1,
        **kwargs
    ):
        self.symmetries = symmetries

        if model is None:
            model = self.default_model(input_shape=input_shape)

        super().__init__(model, loss=loss, **kwargs)

    def default_model(self, input_shape):
        return Convolutional(
            input_shape=input_shape,
            conv_layers_dimensions=[16, 32, 64],
            dense_layers_dimensions=(32, 32),
            steps_per_pooling=1,
            number_of_outputs=4,
        )

    def data_generator(self, *args, **kwargs):
        return AutoTrackGenerator(*args, symmetries=self.symmetries, **kwargs)

    # def predict(self, x, *args, **kwargs):

    #     a = self.model.predict(x, *args, **kwargs)
    #     b = self.model.predict(x[:, ::-1, ::-1], *args, **kwargs)

    #     return (a - b) / 2


class AutoMultiTracker(KerasModel):

    data_generator = AutoTrackGenerator

    def __init__(
        self,
        model=None,
        input_shape=(None, None, 1),
        loss=squared_field_affine_consistency,
        **kwargs
    ):
        if not TFA_INSTALLED:
            raise RuntimeError(
                "The multipartcle version of the autotracker requires Tensorflow addons. You may need to update Tensorflow to install it."
            )

        if model is None:
            model = self.default_model(input_shape)

        super().__init__(model=model, loss=loss, **kwargs)

    def default_model(self, input_shape):

        valid_conv_block = ConvolutionalBlock(padding="valid")
        valid_pooling_block = PoolingBlock(padding="valid")
        valid_deconv_block = DeconvolutionalBlock(padding="valid")

        return UNet(
            input_shape=input_shape,
            conv_layers_dimensions=[16, 32, 64],
            base_conv_layers_dimensions=(128,),
            steps_per_pooling=1,
            number_of_outputs=3,
            encoder_convolution_block=valid_conv_block,
            decoder_convolution_block=valid_conv_block,
            base_convolution_block=valid_conv_block,
            output_convolution_block=valid_conv_block,
            pooling_block=valid_pooling_block,
            upsampling_block=valid_deconv_block,
        )

    def predict(self, x, *args, **kwargs):

        a = self.model.predict(x, *args, **kwargs)
        b = self.model.predict(x[:, ::-1, ::-1], *args, **kwargs)[:, ::-1, ::-1]

        return (a - b) / 2