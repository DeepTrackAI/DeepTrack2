import tensorflow as tf
from tensorflow.keras import layers

from ..utils import KerasModel, GELU
from ..layers import as_block, DenseBlock
from ..embeddings import LearnableDistanceEmbedding


class MAGIK(KerasModel):
    """
    Message passing graph neural network.
    Parameters:
    -----------
    dense_layer_dimensions: list of ints
        List of the number of units in each dense layer of the encoder and decoder. The
        number of layers is inferred from the length of this list.
    base_layer_dimensions: list of ints
        List of the latent dimensions of the graph blocks. The number of layers is
        inferred from the length of this list.
    number_of_node_outputs: int
        Number of output node features.
    number_of_edge_outputs: int
        Number of output edge features.
    node_output_activation: str
        Activation function for the output node layer.
    edge_output_activation: str
        Activation function for the output edge layer.
    dense_block: str, keras.layers.Layer, or callable
        The dense block to use for the encoder and decoder.
    graph_block: str, keras.layers.Layer, or callable
        The graph block to use for the graph blocks.
    output_type: str
        Type of output. Either "nodes", "edges", or "graph".
        If 'key' is not a supported output type, then the
        model output will be the concatenation of the node
        and edge predictions.
    kwargs: dict
        Keyword arguments for the dense block.
    Returns:
    --------
    tf.keras.Model
        Keras model for the graph neural network.
    """

    def __init__(
        self,
        dense_layer_dimensions=(32, 64, 96),
        base_layer_dimensions=(96, 96),
        number_of_node_features=3,
        number_of_edge_features=1,
        number_of_node_outputs=1,
        number_of_edge_outputs=1,
        node_output_activation=None,
        edge_output_activation=None,
        encoder_dense_block=DenseBlock(
            activation=GELU, normalization="LayerNormalization"
        ),
        decoder_dense_block=DenseBlock(
            activation=GELU, normalization="LayerNormalization"
        ),
        graph_block="FGNN",
        output_type="graph",
        **kwargs
    ):

        encoder_dense_block = as_block(encoder_dense_block)
        decoder_dense_block = as_block(decoder_dense_block)
        graph_block = as_block(graph_block)

        node_features, edge_features, edges, edge_dropout = (
            tf.keras.Input(shape=(None, number_of_node_features)),
            tf.keras.Input(shape=(None, number_of_edge_features)),
            tf.keras.Input(shape=(None, 2), dtype=tf.int32),
            tf.keras.Input(shape=(None, 2)),
        )

        node_layer = node_features
        edge_layer = edge_features

        # Encoder for node and edge features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            node_layer = encoder_dense_block(
                dense_layer_dimension,
                name="node_ide" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = encoder_dense_block(
                dense_layer_dimension,
                name="edge_ide" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        # Extract distance matrix
        edge_weights = LearnableDistanceEmbedding(name="LearnableDistanceEmb")(
            edge_features[..., 0:1]
        )

        # Bottleneck path, graph blocks
        layer = (node_layer, edge_layer, edges, edge_weights, edge_dropout)
        for base_layer_number, base_layer_dimension in zip(
            range(len(base_layer_dimensions)), base_layer_dimensions
        ):
            layer = graph_block(
                base_layer_dimension,
                name="graph_block_" + str(base_layer_number),
            )(layer)

        # Decoder for node and edge features
        node_layer, edge_layer, *_ = layer
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)),
            reversed(dense_layer_dimensions),
        ):
            node_layer = decoder_dense_block(
                dense_layer_dimension,
                name="node_idd" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = decoder_dense_block(
                dense_layer_dimension,
                name="edge_idd" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        # Output layers
        node_output = layers.Dense(
            number_of_node_outputs,
            activation=node_output_activation,
            name="node_prediction",
        )(node_layer)

        edge_output = layers.Dense(
            number_of_edge_outputs,
            activation=edge_output_activation,
            name="edge_prediction",
        )(edge_layer)

        output_dict = {
            "nodes": node_output,
            "edges": edge_output,
            "graph": [node_output, edge_output],
        }
        try:
            outputs = output_dict[output_type]
        except KeyError:
            outputs = output_dict["graph"]

        model = tf.keras.models.Model(
            [node_features, edge_features, edges, edge_dropout],
            outputs,
        )

        super().__init__(model, **kwargs)


class CTMAGIK(KerasModel):
    """
    Message passing graph neural network.
    Parameters:
    -----------
    dense_layer_dimensions: list of ints
        List of the number of units in each dense layer of the encoder and decoder. The
        number of layers is inferred from the length of this list.
    base_layer_dimensions: list of ints
        List of the latent dimensions of the graph blocks. The number of layers is
        inferred from the length of this list.
    number_of_node_outputs: int
        Number of output node features.
    number_of_edge_outputs: int
        Number of output edge features.
    number_of_global_outputs: int
        Number of output global features.
    node_output_activation: str or activation function or layer
        Activation function for the output node layer. See keras docs for accepted strings.
    edge_output_activation: str or activation function or layer
        Activation function for the output edge layer. See keras docs for accepted strings.
    cls_layer_dimension: int
        Number of units in the decoder layer for global features.
    global_output_activation: str or activation function or layer
        Activation function for the output global layer. See keras docs for accepted strings.
    dense_block: str, keras.layers.Layer, or callable
        The dense block to use for the encoder and decoder.
    graph_block: str, keras.layers.Layer, or callable
        The graph block to use for the graph blocks.
    classtokens_block: str, keras.layers.Layer, or callable
        The embedding block to use for the class tokens.
    output_type: str
        Type of output. Either "nodes", "edges", "global" or
        "graph". If 'key' is not a supported output type, then
        the model output will be the concatenation of the node,
        edge, and global predictions.
    kwargs: dict
        Keyword arguments for the dense block.
    Returns:
    --------
    tf.keras.Model
        Keras model for the graph neural network.
    """

    def __init__(
        self,
        dense_layer_dimensions=(32, 64, 96),
        base_layer_dimensions=(96, 96),
        number_of_node_features=3,
        number_of_edge_features=1,
        number_of_node_outputs=1,
        number_of_edge_outputs=1,
        number_of_global_outputs=1,
        node_output_activation=None,
        edge_output_activation=None,
        cls_layer_dimension=64,
        global_output_activation=None,
        dense_block=DenseBlock(
            activation=GELU, normalization="LayerNormalization"
        ),
        graph_block="CTFGNN",
        classtoken_block="ClassToken",
        output_type="graph",
        **kwargs
    ):

        dense_block = as_block(dense_block)
        graph_block = as_block(graph_block)
        classtoken_block = as_block(classtoken_block)

        node_features, edge_features, edges, edge_dropout = (
            tf.keras.Input(shape=(None, number_of_node_features)),
            tf.keras.Input(shape=(None, number_of_edge_features)),
            tf.keras.Input(shape=(None, 2), dtype=tf.int32),
            tf.keras.Input(shape=(None, 2)),
        )

        node_layer = node_features
        edge_layer = edge_features

        # Encoder for node and edge features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            node_layer = dense_block(
                dense_layer_dimension,
                name="node_ide" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = dense_block(
                dense_layer_dimension,
                name="edge_ide" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        # Extract distance matrix
        edge_weights = LearnableDistanceEmbedding(name="LearnableDistanceEmb")(
            edge_features[..., 0:1]
        )

        # Bottleneck path, graph blocks
        layer = (
            classtoken_block(base_layer_dimensions, name="ClassTokenLayer")(
                node_layer
            ),
            edge_layer,
            edges,
            edge_weights,
            edge_dropout,
        )
        for base_layer_number, base_layer_dimension in zip(
            range(len(base_layer_dimensions)), base_layer_dimensions
        ):
            layer = graph_block(
                base_layer_dimension,
                name="graph_block_" + str(base_layer_number),
            )(layer)

        # Decoder for node, edge, and global features
        node_layer, edge_layer, *_ = layer
        # Split node and global features
        cls_layer, node_layer = (
            tf.keras.layers.Lambda(
                lambda x: x[:, 0], name="RetrieveClassToken"
            )(node_layer),
            node_layer[:, 1:],
        )
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)),
            reversed(dense_layer_dimensions),
        ):
            node_layer = dense_block(
                dense_layer_dimension,
                name="node_idd" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = dense_block(
                dense_layer_dimension,
                name="edge_idd" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        cls_layer = dense_block(cls_layer_dimension, name="cls_mlp", **kwargs)(
            cls_layer
        )

        # Output layers
        node_output = layers.Dense(
            number_of_node_outputs,
            activation=node_output_activation,
            name="node_prediction",
        )(node_layer)

        edge_output = layers.Dense(
            number_of_edge_outputs,
            activation=edge_output_activation,
            name="edge_prediction",
        )(edge_layer)

        global_output = layers.Dense(
            number_of_global_outputs,
            activation=global_output_activation,
            name="global_prediction",
        )(cls_layer)

        output_dict = {
            "nodes": node_output,
            "edges": edge_output,
            "global": global_output,
            "graph": [node_output, edge_output, global_output],
        }
        try:
            outputs = output_dict[output_type]
        except KeyError:
            outputs = output_dict["graph"]

        model = tf.keras.models.Model(
            [node_features, edge_features, edges, edge_dropout],
            outputs,
        )

        super().__init__(model, **kwargs)
