import tensorflow as tf
from tensorflow.keras import layers

from ..layers import (
    MultiHeadGatedSelfAttention,
    MultiHeadSelfAttention,
    register,
)
from ..utils import as_activation, as_normalization, single_layer_call, GELU


class FGNN(tf.keras.layers.Layer):
    """
    Fingerprinting Graph Layer.
    Parameters
    ----------
    filters : int
        Number of filters.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    random_edge_dropout : float, optional
        Random edge dropout.
    use_gates : bool, optional
        Whether to use gated self-attention layers as update layer. Defaults to True.
    att_layer_kwargs : dict, optional
        Keyword arguments for the self-attention layer.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        filters,
        activation=GELU,
        normalization="LayerNormalization",
        random_edge_dropout=False,
        use_gates=True,
        att_layer_kwargs={},
        norm_kwargs={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.random_edge_dropout = random_edge_dropout

        _multi_head_att_layer = (
            MultiHeadGatedSelfAttention
            if use_gates
            else MultiHeadSelfAttention
        )
        # node update layer
        self.update_layer = tf.keras.Sequential(
            [
                _multi_head_att_layer(**att_layer_kwargs),
                as_activation(activation),
                as_normalization(normalization)(**norm_kwargs),
            ]
        )

        # message layer
        self.message_layer = tf.keras.Sequential(
            [
                layers.Dense(self.filters),
                as_activation(activation),
                as_normalization(normalization)(**norm_kwargs),
            ]
        )

    def call(self, inputs):
        nodes, edge_features, edges, edge_weights, edge_dropout = inputs

        number_of_nodes = tf.shape(nodes)[1]
        number_of_edges = tf.shape(edges)[1]
        number_of_node_features = nodes.shape[-1]

        batch_size = tf.shape(nodes)[0]

        # Get neighbors node features, shape = (batch, nOfedges, 2, nOffeatures)
        message_inputs = tf.gather(nodes, edges, batch_dims=1)

        # Concatenate nodes features with edge features,
        # shape = (batch, nOfedges, 2*nOffeatures + nOffedgefeatures)
        messages = tf.reshape(
            message_inputs,
            (
                batch_size,
                number_of_edges,
                2 * number_of_node_features,
            ),
        )
        reshaped = tf.concat(
            [
                messages,
                edge_features,
            ],
            -1,
        )

        # Compute messages/update edges, shape = (batch, nOfedges, filters)
        messages = self.message_layer(reshaped)

        # Compute weighted messages
        # shape = (batch, nOfedges, filters)
        weighted_messages = messages * edge_weights

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge, dropout = x

            merged_edges = tf.math.unsorted_segment_sum(
                message * dropout[:, 1:2],
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (weighted_messages, edges, edge_dropout),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        Combined = [nodes, aggregated]
        updated_nodes = self.update_layer(Combined)

        return (
            updated_nodes,
            weighted_messages,
            edges,
            edge_weights,
            edge_dropout,
        )


@register("FGNN")
def FGNNlayer(
    activation=GELU,
    normalization="LayerNormalization",
    random_edge_dropout=False,
    use_gates=True,
    att_layer_kwargs={},
    norm_kwargs={},
    **kwargs,
):
    """Fingerprinting Graph Layer.
    Parameters
    ----------
    filters : int
        Number of filters.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    random_edge_dropout : float, optional
        Random edge dropout.
    use_gates : bool, optional
        Whether to use gated self-attention layers as update layer. Defaults to True.
    att_layer_kwargs : dict, optional
        Keyword arguments for the self-attention layer.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = FGNN(
            filters,
            activation,
            normalization,
            random_edge_dropout,
            use_gates,
            att_layer_kwargs,
            norm_kwargs,
            **kwargs_inner,
        )
        return lambda x: single_layer_call(x, layer, None, None, {})

    return Layer


class ClassTokenFGNN(FGNN):
    """
    Fingerprinting Graph Layer with Class Token.
    Parameters
    ----------
    filters : int
        Number of filters.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    random_edge_dropout : float, optional
        Random edge dropout.
    use_gates : bool, optional
        Whether to use gated self-attention layers as update layer. Defaults to True.
    att_layer_kwargs : dict, optional
        Keyword arguments for the self-attention layer.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def build(self, input_shape):
        super().build(input_shape)
        self.combine_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1)),
                layers.Dense(self.filters),
            ]
        )

    def call(self, inputs):
        nodes, edge_features, edges, edge_weights, edge_dropout = inputs

        # Split nodes and class-token embeddings
        class_token, nodes = nodes[:, 0:1, :], nodes[:, 1:, :]

        number_of_nodes = tf.shape(nodes)[1]
        number_of_edges = tf.shape(edges)[1]
        number_of_node_features = nodes.shape[-1]

        batch_size = tf.shape(nodes)[0]

        # Get neighbors node features, shape = (batch, nOfedges, 2, nOffeatures)
        message_inputs = tf.gather(nodes, edges, batch_dims=1)

        # Concatenate nodes features with edge features,
        # shape = (batch, nOfedges, 2*nOffeatures + nOffedgefeatures)
        messages = tf.reshape(
            message_inputs,
            (
                batch_size,
                number_of_edges,
                2 * number_of_node_features,
            ),
        )
        reshaped = tf.concat(
            [
                messages,
                edge_features,
            ],
            -1,
        )

        # Compute messages/update edges, shape = (batch, nOfedges, filters)
        messages = self.message_layer(reshaped)

        # Compute weighted messages
        # shape = (batch, nOfedges, filters)
        weighted_messages = messages * edge_weights

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge, dropout = x

            merged_edges = tf.math.unsorted_segment_sum(
                message * dropout[:, 1:2],
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (weighted_messages, edges, edge_dropout),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        Combined = tf.concat(
            [class_token, self.combine_layer([nodes, aggregated])], axis=1
        )
        updated_nodes = self.update_layer(Combined)

        return (
            updated_nodes,
            weighted_messages,
            edges,
            edge_weights,
            edge_dropout,
        )


@register("CTFGNN")
def ClassTokenFGNNlayer(
    activation=GELU,
    normalization="LayerNormalization",
    random_edge_dropout=False,
    use_gates=True,
    att_layer_kwargs={},
    norm_kwargs={},
    **kwargs,
):
    """Fingerprinting Graph Layer with Class Token.
    Parameters
    ----------
    number_of_heads : int
        Number of attention heads.
    message_layer : str or callable
        Message layer.
    update_layer : str or callable
        Update layer.
    random_edge_dropout : float, optional
        Random edge dropout.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = ClassTokenFGNN(
            filters,
            activation,
            normalization,
            random_edge_dropout,
            use_gates,
            att_layer_kwargs,
            norm_kwargs,
            **kwargs_inner,
        )
        return lambda x: single_layer_call(x, layer, None, None, {})

    return Layer
