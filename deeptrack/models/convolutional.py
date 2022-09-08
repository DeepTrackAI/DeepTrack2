from tensorflow.keras import layers, models

from ..backend.citations import unet_bibtex
from .layers import as_block, TransformerEncoderLayer, DenseBlock, Identity
from .embeddings import ClassToken, LearnablePositionEmbsLayer
from .utils import KerasModel, as_KerasModel, with_citation, GELU


def center_crop(layer, target_layer):
    def inner(x):
        x, y = x
        shape = x.shape
        target_shape = y.shape
        diff_y = (shape[1] - target_shape[1]) // 2
        diff_x = (shape[2] - target_shape[2]) // 2
        return x[
            :,
            diff_y : (diff_y + target_shape[1]),
            diff_x : (diff_x + target_shape[2]),
        ]

    return layers.Lambda(inner, output_shape=lambda x: x[1].shape)(
        [layer, target_layer]
    )


class Convolutional(KerasModel):
    """Creates and compiles a convolutional neural network.

    A convolutional network with a dense top.

    Parameters
    ----------
    input_shape : tuple of ints
        Size of the images to be analyzed.
    conv_layers_dimensions : tuple of ints
        Number of convolutions in each convolutional layer.
    dense_layers_dimensions : tuple of ints
        Number of units in each dense layer.
    dropout : tuple of float
        Adds a dropout between the convolutional layers
    number_of_outputs : int
        Number of units in the output layer.
    output_activation : str or keras activation
        The activation function of the output.
    loss : str or keras loss function
        The loss function of the network.
    layer_function : Callable[int] -> keras layer
        Function that returns a convolutional layer with convolutions
        determined by the input argument. Can be use to futher customize the network.
    Returns
    -------
    keras.models.Model
        Deep learning network
    """

    def __init__(
        self,
        input_shape=(51, 51, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        dense_layers_dimensions=(32, 32),
        steps_per_pooling=1,
        dropout=(),
        dense_top=True,
        number_of_outputs=3,
        output_activation=None,
        output_kernel_size=3,
        loss="mae",
        input_layer=None,
        convolution_block="convolutional",
        pooling_block="pooling",
        dense_block="dense",
        **kwargs,
    ):

        # Update layer functions
        dense_block = as_block(dense_block)
        convolution_block = as_block(convolution_block)
        pooling_block = as_block(pooling_block)

        # INITIALIZE DEEP LEARNING NETWORK

        if isinstance(input_shape, list):
            network_input = [layers.Input(shape) for shape in input_shape]
            inputs = layers.Concatenate(axis=-1)(network_input)
        else:
            network_input = layers.Input(input_shape)
            inputs = network_input

        layer = inputs

        if input_layer:
            layer = input_layer(layer)

        # CONVOLUTIONAL BASIS
        for conv_layer_dimension in conv_layers_dimensions:

            for _ in range(steps_per_pooling):
                layer = convolution_block(conv_layer_dimension)(layer)

            if dropout:
                layer = layers.SpatialDropout2D(dropout[0])(layer)
                dropout = dropout[1:]

            # add pooling layer
            layer = pooling_block(conv_layer_dimension)(layer)

        # DENSE TOP

        if dense_top:
            layer = layers.Flatten()(layer)
            for dense_layer_dimension in dense_layers_dimensions:
                layer = dense_block(dense_layer_dimension)(layer)
            output_layer = layers.Dense(
                number_of_outputs, activation=output_activation
            )(layer)
        else:

            output_layer = layers.Conv2D(
                number_of_outputs,
                kernel_size=output_kernel_size,
                activation=output_activation,
                padding="same",
                name="output",
            )(layer)

        model = models.Model(network_input, output_layer)
        super().__init__(model, loss=loss, **kwargs)


convolutional = Convolutional


class UNet(KerasModel):
    """Creates and compiles a U-Net.

    Parameters
    ----------
    input_shape : tuple of ints
        Size of the images to be analyzed.
    conv_layers_dimensions : tuple of ints
        Number of convolutions in each convolutional layer during down-
        and upsampling.
    base_conv_layers_dimensions : tuple of ints
        Number of convolutions in each convolutional layer at the base
        of the unet, where the image is the most downsampled.
    output_conv_layers_dimensions : tuple of ints
        Number of convolutions in each convolutional layer after the
        upsampling.
    steps_per_pooling : int
        Number of convolutional layers between each pooling and upsampling
        step.
    number_of_outputs : int
        Number of convolutions in output layer.
    output_activation : str or keras activation
        The activation function of the output.
    loss : str or keras loss function
        The loss function of the network.
    layer_function : Callable[int] -> keras layer
        Function that returns a convolutional layer with convolutions
        determined by the input argument. Can be use to futher customize the network.
    Returns
    -------
    keras.models.Model
        Deep learning network.
    """

    citation = unet_bibtex

    def __init__(
        self,
        input_shape=(None, None, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        base_conv_layers_dimensions=(128, 128),
        output_conv_layers_dimensions=(16, 16),
        dropout=(),
        steps_per_pooling=1,
        number_of_outputs=1,
        output_kernel_size=3,
        output_activation=None,
        input_layer=None,
        encoder_convolution_block="convolutional",
        base_convolution_block="convolutional",
        decoder_convolution_block="convolutional",
        output_convolution_block="convolutional",
        pooling_block="pooling",
        upsampling_block="deconvolutional",
        **kwargs,
    ):

        # Update layer functions

        encoder_convolution_block = as_block(encoder_convolution_block)
        base_convolution_block = as_block(base_convolution_block)
        output_convolution_block = as_block(output_convolution_block)
        decoder_convolution_block = as_block(decoder_convolution_block)
        pooling_block = as_block(pooling_block)
        upsampling_block = as_block(upsampling_block)

        unet_input = layers.Input(input_shape)

        concat_layers = []

        layer = unet_input

        if input_layer:
            layer = input_layer(layer)

        # Downsampling path
        for conv_layer_dimension in conv_layers_dimensions:
            for _ in range(steps_per_pooling):
                layer = encoder_convolution_block(conv_layer_dimension)(layer)
            concat_layers.append(layer)

            if dropout:
                layer = layers.SpatialDropout2D(dropout[0])(layer)
                dropout = dropout[1:]

            layer = pooling_block(conv_layer_dimension)(layer)

        # Bottleneck path
        for conv_layer_dimension in base_conv_layers_dimensions:
            layer = base_convolution_block(conv_layer_dimension)(layer)

        # Upsampling path
        for conv_layer_dimension, concat_layer in zip(
            reversed(conv_layers_dimensions), reversed(concat_layers)
        ):

            layer = upsampling_block(conv_layer_dimension)(layer)

            # concat_layer = center_crop(concat_layer, layer) Not currently working

            layer = layers.Concatenate(axis=-1)([layer, concat_layer])

            for _ in range(steps_per_pooling):
                layer = decoder_convolution_block(conv_layer_dimension)(layer)

        # Output step
        for conv_layer_dimension in output_conv_layers_dimensions:
            layer = output_convolution_block(conv_layer_dimension)(layer)

        output_layer = layers.Conv2D(
            number_of_outputs,
            kernel_size=output_kernel_size,
            activation=output_activation,
            padding="same",
        )(layer)

        model = models.Model(unet_input, output_layer)
        super().__init__(model, **kwargs)


unet = UNet


class EncoderDecoder(KerasModel):
    """Creates and compiles an EncoderDecoder.
    Parameters
    ----------
    input_shape : tuple of ints
        Size of the images to be analyzed.
    conv_layers_dimensions : tuple of ints
        Number of convolutions in each convolutional layer during down-
        and upsampling.
    base_conv_layers_dimensions : tuple of ints
        Number of convolutions in each convolutional layer at the base
        of the unet, where the image is the most downsampled.
    output_conv_layers_dimensions : tuple of ints
        Number of convolutions in each convolutional layer after the
        upsampling.
    steps_per_pooling : int
        Number of convolutional layers between each pooling and upsampling
        step.
    number_of_outputs : int
        Number of convolutions in output layer.
    output_activation : str or keras activation
        The activation function of the output.
    loss : str or keras loss function
        The loss function of the network.

    Returns
    -------
    keras.models.Model
        Deep learning network.
    """

    def __init__(
        self,
        input_shape=(None, None, 1),
        conv_layers_dimensions=(16, 32, 64, 128),
        base_conv_layers_dimensions=(128, 128),
        output_conv_layers_dimensions=(16, 16),
        dropout=(),
        steps_per_pooling=1,
        number_of_outputs=1,
        output_kernel_size=3,
        output_activation=None,
        input_layer=None,
        encoder_convolution_block="convolutional",
        base_convolution_block="convolutional",
        decoder_convolution_block="convolutional",
        output_convolution_block="convolutional",
        pooling_block="pooling",
        upsampling_block="deconvolutional",
        **kwargs,
    ):

        # Update layer functions

        encoder_convolution_block = as_block(encoder_convolution_block)
        base_convolution_block = as_block(base_convolution_block)
        output_convolution_block = as_block(output_convolution_block)
        decoder_convolution_block = as_block(decoder_convolution_block)
        pooling_block = as_block(pooling_block)
        upsampling_block = as_block(upsampling_block)

        unet_input = layers.Input(input_shape)

        layer = unet_input

        if input_layer:
            layer = input_layer(layer)

        # Downsampling path
        for conv_layer_dimension in conv_layers_dimensions:
            for _ in range(steps_per_pooling):
                layer = encoder_convolution_block(conv_layer_dimension)(layer)

            if dropout:
                layer = layers.SpatialDropout2D(dropout[0])(layer)
                dropout = dropout[1:]

            layer = pooling_block(conv_layer_dimension)(layer)

        # Bottleneck path
        for conv_layer_dimension in base_conv_layers_dimensions:
            layer = base_convolution_block(conv_layer_dimension)(layer)

        # Upsampling path
        for conv_layer_dimension in reversed(conv_layers_dimensions):

            layer = upsampling_block(conv_layer_dimension)(layer)

            for _ in range(steps_per_pooling):
                layer = decoder_convolution_block(conv_layer_dimension)(layer)

        # Output step
        for conv_layer_dimension in output_conv_layers_dimensions:
            layer = output_convolution_block(conv_layer_dimension)(layer)

        output_layer = layers.Conv2D(
            number_of_outputs,
            kernel_size=output_kernel_size,
            activation=output_activation,
            padding="same",
        )(layer)

        model = models.Model(unet_input, output_layer)
        super().__init__(model, **kwargs)


class ClsTransformerBaseModel(KerasModel):
    """Base class for Transformer models with classification heads.

    Parameters
    ----------
    inputs : list of keras.layers.Input
        Input layers of the network.
    encoder : tf.Tensor
        Encoded representation of the input.
    number_of_transformer_layers : int
        Number of Transformer layers in the model.
    base_fwd_mlp_dimensions : int
        Size of the hidden layers in the forward MLP of the Transformer layers.
    transformer_block : str or keras.layers.Layer
        The Transformer layer to use. By default, uses the TransformerEncoder
        block. See .layers for available Transformer layers.
    cls_layer_dimensions : int, optional
        Size of the ClassToken layer. If None, no ClassToken layer is added.
    node_decoder_layer_dimensions: list of ints
        List of the number of units in each dense layer of the nodes' decoder. The
        number of layers is inferred from the length of this list.
    number_of_cls_outputs: int
        Number of output cls features.
    number_of_nodes_outputs: int
        Number of output nodes features.
    cls_output_activation: str or activation function or layer
        Activation function for the output cls layer. See keras docs for accepted strings.
    node_output_activation: str or activation function or layer
        Activation function for the output node layer. See keras docs for accepted strings.
    transformer_block: str, keras.layers.Layer, or callable
        The transformer layer. See .layers for available transformer blocks.
    dense_block: str, keras.layers.Layer, or callable
        The dense block to use for the nodes' decoder.
    cls_norm_block: str, keras.layers.Layer, or callable
        The normalization block to use for the cls layer.
    use_learnable_positional_embs : bool
        Whether to use learnable positional embeddings.
    output_type: str
        Type of output. Either "cls", "cls_rep", "nodes" or
        "full". If 'key' is not a supported output type, then
        the model output will be the concatenation of the node
        and cls predictions ("full").
    kwargs : dict
        Additional arguments to be passed to the KerasModel constructor.
    """

    def __init__(
        self,
        inputs,
        encoder,
        number_of_transformer_layers=12,
        base_fwd_mlp_dimensions=256,
        cls_layer_dimension=None,
        number_of_cls_outputs=1,
        cls_output_activation="linear",
        transformer_block=TransformerEncoderLayer(
            normalization="LayerNormalization",
            dropout=0.1,
            norm_kwargs={"epsilon": 1e-6},
        ),
        dense_block=DenseBlock(
            activation=GELU,
            normalization="LayerNormalization",
            norm_kwargs={"epsilon": 1e-6},
        ),
        positional_embedding_block=Identity(),
        output_type="cls",
        transformer_input_kwargs={},
        **kwargs,
    ):
        transformer_block = as_block(transformer_block)
        dense_block = as_block(dense_block)
        positional_embedding_block = as_block(positional_embedding_block)

        layer = ClassToken(name="class_token")(encoder)

        layer = positional_embedding_block(
            layer.shape[-1], name="Transformer/posembed_input"
        )(layer)

        # Bottleneck path, Transformer layers
        for n in range(number_of_transformer_layers):
            layer, _ = transformer_block(
                base_fwd_mlp_dimensions, name=f"Transformer/encoderblock_{n}"
            )(layer, **transformer_input_kwargs)

        # Extract global representation
        cls_rep = layers.Lambda(lambda x: x[:, 0], name="RetrieveClassToken")(
            layer
        )

        # Process cls features
        cls_layer = cls_rep
        if cls_layer_dimension is not None:
            cls_layer = dense_block(cls_layer_dimension, name="cls_mlp")(
                cls_layer
            )

        cls_output = layers.Dense(
            number_of_cls_outputs,
            activation=cls_output_activation,
            name="cls_prediction",
        )(cls_layer)

        output_dict = {
            "cls_rep": cls_rep,
            "cls": cls_output,
        }
        try:
            outputs = output_dict[output_type]
        except KeyError:
            outputs = output_dict["cls"]

        model = models.Model(inputs=inputs, outputs=outputs)

        super().__init__(model, **kwargs)


class ViT(ClsTransformerBaseModel):
    """
    Creates and compiles a ViT model.
    input_shape : tuple of ints
        Size of the images to be analyzed.
    patch_shape : int
        Size of the patches to be extracted from the input images.
    hidden_size : int
        Size of the hidden layers in the ViT model.
    number_of_transformer_layers : int
        Number of Transformer layers in the model.
    base_fwd_mlp_dimensions : int
        Size of the hidden layers in the forward MLP of the Transformer layers.
    transformer_block : str or keras.layers.Layer
        The Transformer layer to use. By default, uses the TransformerEncoder
        block. See .layers for available Transformer layers.
    number_of_cls_outputs: int
        Number of output cls features.
    cls_output_activation: str or activation function or layer
        Activation function for the output cls layer. See keras docs for accepted strings.
    use_learnable_positional_embs : bool
        Whether to use learnable positional embeddings.
    output_type: str
        Type of output. Either "cls", "cls_rep", "nodes" or
        "full". If 'key' is not a supported output type, then
        the model output will be the concatenation of the node
        and cls predictions ("full").
    kwargs : dict
        Additional arguments to be passed to the TransformerBaseModel contructor
        for advanced configuration.
    """

    def __init__(
        self,
        input_shape=(28, 28, 1),
        patch_shape=4,
        hidden_size=72,
        number_of_transformer_layers=4,
        base_fwd_mlp_dimensions=256,
        number_of_cls_outputs=10,
        cls_output_activation="linear",
        output_type="cls",
        positional_embedding_block=LearnablePositionEmbsLayer(),
        **kwargs,
    ):

        assert (
            input_shape[0] % patch_shape == 0
        ), "image_size must be a multiple of patch_size"

        vit_input = layers.Input(shape=input_shape)
        encoder_layer = layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_shape,
            strides=patch_shape,
            padding="valid",
            name="embedding",
        )(vit_input)
        encoder_layer = layers.Reshape(
            (encoder_layer.shape[1] * encoder_layer.shape[2], hidden_size)
        )(encoder_layer)

        super().__init__(
            inputs=vit_input,
            encoder=encoder_layer,
            number_of_transformer_layers=number_of_transformer_layers,
            base_fwd_mlp_dimensions=base_fwd_mlp_dimensions,
            number_of_cls_outputs=number_of_cls_outputs,
            cls_output_activation=cls_output_activation,
            output_type=output_type,
            positional_embedding_block=positional_embedding_block,
            **kwargs,
        )


class Transformer(KerasModel):
    """
    Creates and compiles a Transformer model.
    """

    def __init__(
        self,
        number_of_node_features=3,
        dense_layer_dimensions=(64, 96),
        number_of_transformer_layers=12,
        base_fwd_mlp_dimensions=256,
        number_of_node_outputs=1,
        node_output_activation="linear",
        transformer_block=TransformerEncoderLayer(
            normalization="LayerNormalization",
            dropout=0.1,
            norm_kwargs={"epsilon": 1e-6},
        ),
        dense_block=DenseBlock(
            activation=GELU,
            normalization="LayerNormalization",
            norm_kwargs={"epsilon": 1e-6},
        ),
        positional_embedding_block=Identity(),
        **kwargs,
    ):

        dense_block = as_block(dense_block)

        transformer_input, transformer_mask = (
            layers.Input(shape=(None, number_of_node_features)),
            layers.Input(shape=(None, 2), dtype="int32"),
        )

        layer = transformer_input
        # Encoder for input features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            layer = dense_block(
                dense_layer_dimension,
                name="fencoder_" + str(dense_layer_number + 1),
            )(layer)

        layer = positional_embedding_block(
            layer.shape[-1], name="Transformer/posembed_input"
        )(layer)

        # Bottleneck path, Transformer layers
        for n in range(number_of_transformer_layers):
            layer, _ = transformer_block(
                base_fwd_mlp_dimensions, name=f"Transformer/encoderblock_{n}"
            )(layer, edges=transformer_mask)

        # Decoder for node and edge features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)),
            reversed(dense_layer_dimensions),
        ):
            layer = dense_block(
                dense_layer_dimension,
                name="fdecoder" + str(dense_layer_number + 1),
                **kwargs,
            )(layer)

        # Output layers
        output_layer = layers.Dense(
            number_of_node_outputs,
            activation=node_output_activation,
            name="node_prediction",
        )(layer)

        model = models.Model(
            [transformer_input, transformer_mask],
            output_layer,
        )

        super().__init__(model, **kwargs)
