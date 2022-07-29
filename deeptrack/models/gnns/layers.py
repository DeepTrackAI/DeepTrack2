import tensorflow as tf
from tensorflow.keras import layers

from ..layers import (
    MultiHeadGatedSelfAttention,
    MultiHeadSelfAttention,
    register,
)
from ..utils import as_activation, as_normalization, single_layer_call, GELU


class MPN(tf.keras.layers.Layer):
    """
    Message-passing Graph Layer.
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
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
        combine_layer=tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1)),
        norm_kwargs={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.random_edge_dropout = random_edge_dropout
        self.combine_layer = combine_layer

        # message layer
        self.message_layer = tf.keras.Sequential(
            [
                layers.Dense(self.filters),
                as_activation(activation),
                as_normalization(normalization)(**norm_kwargs),
            ]
        )

        # node update layers
        self.update_layer = layers.Dense(self.filters)
        self.update_norm = tf.keras.Sequential(
            [
                as_activation(activation),
                as_normalization(normalization)(**norm_kwargs),
            ]
        )

    def nodes_handler(self, nodes):
        return nodes, None

    def update_node_features(self, nodes, aggregated, learnable_embs, edges):
        Combined = self.combine_layer([nodes, aggregated])
        updated_nodes = self.update_norm(self.update_layer(Combined))
        return updated_nodes

    def call(self, inputs):
        nodes, edge_features, edges, edge_weights, edge_dropout = inputs

        # Handles node features according to the implementation
        nodes, learnable_embs = self.nodes_handler(nodes)

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
        updated_nodes = self.update_node_features(
            nodes, aggregated, learnable_embs, edges
        )

        return (
            updated_nodes,
            weighted_messages,
            edges,
            edge_weights,
            edge_dropout,
        )


@register("MPN")
def MPNLayer(
    filters,
    activation=GELU,
    normalization="LayerNormalization",
    norm_kwargs={},
    **kwargs,
):
    """
    Message-passing Graph Layer.
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = MPN(
            filters,
            activation=activation,
            normalization=normalization,
            norm_kwargs=norm_kwargs,
            **kwargs,
        )
        return lambda x: single_layer_call(x, layer, None, None, {})

    return Layer


class GRUMPN(MPN):
    """
    GRU Graph Layer.
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        filters,
        **kwargs,
    ):
        super().__init__(filters, **kwargs)

        # node update layer
        self.update_layer = layers.GRU(filters, time_major=True)

    def update_node_features(self, nodes, aggregated, learnable_embs, edges):
        Combined = tf.reshape(
            tf.stack([nodes, aggregated], axis=0), (2, -1, nodes.shape[-1])
        )
        updated_nodes = self.update_layer(Combined)
        return tf.reshape(updated_nodes, shape=tf.shape(nodes))


@register("GRUMPN")
def GRUMPNLayer(
    filters,
    activation=GELU,
    normalization="LayerNormalization",
    norm_kwargs={},
    **kwargs,
):
    """
    GRU Graph Layer.
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = GRUMPN(
            filters,
            activation=activation,
            normalization=normalization,
            norm_kwargs=norm_kwargs,
            **kwargs,
        )
        return lambda x: single_layer_call(x, layer, None, None, {})

    return Layer


class FGNN(MPN):
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
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
        use_gates=True,
        att_layer_kwargs={},
        combine_layer=layers.Layer(),
        norm_kwargs={},
        **kwargs,
    ):
        super().__init__(
            filters,
            activation=activation,
            normalization=normalization,
            combine_layer=combine_layer,
            norm_kwargs=norm_kwargs,
            **kwargs,
        )

        multi_head_att_layer = (
            MultiHeadGatedSelfAttention
            if use_gates
            else MultiHeadSelfAttention
        )

        # node update layer
        self.update_layer = multi_head_att_layer(**att_layer_kwargs)
        self.update_norm = tf.keras.Sequential(
            [
                as_activation(activation),
                as_normalization(normalization)(**norm_kwargs),
            ]
        )


@register("FGNN")
def FGNNlayer(
    activation=GELU,
    normalization="LayerNormalization",
    use_gates=True,
    att_layer_kwargs={},
    norm_kwargs={},
    **kwargs,
):
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = FGNN(
            filters,
            activation=activation,
            normalization=normalization,
            use_gates=use_gates,
            att_layer_kwargs=att_layer_kwargs,
            norm_kwargs=norm_kwargs,
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
    dense_to_combine : bool, optional
        Whether to use a dense layer to combine node features and aggregated messages. Defaults to True.
    att_layer_kwargs : dict, optional
        Keyword arguments for the self-attention layer.
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        filters,
        combine_layer=tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1)),
        dense_to_combine=True,
        **kwargs,
    ):

        super().__init__(
            filters,
            combine_layer=combine_layer,
            **kwargs,
        )

        self.dense_to_combine = (
            layers.Dense(self.filters) if dense_to_combine else layers.Layer()
        )

        self.process_class_token = layers.Dense(self.filters)

    def nodes_handler(self, nodes):
        return nodes[:, 1:, :], nodes[:, 0:1, :]

    def update_node_features(self, nodes, aggregated, learnable_embs, edges):
        Combined = tf.concat(
            [
                self.process_class_token(learnable_embs),
                self.dense_to_combine(self.combine_layer([nodes, aggregated])),
            ],
            axis=1,
        )
        updated_nodes = self.update_norm(self.update_layer(Combined))
        return updated_nodes


@register("CTFGNN")
def ClassTokenFGNNlayer(
    activation=GELU,
    normalization="LayerNormalization",
    use_gates=True,
    att_layer_kwargs={},
    norm_kwargs={},
    **kwargs,
):
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
    dense_to_combine : bool, optional
        Whether to use a dense layer to combine node features and aggregated messages. Defaults to True.
    att_layer_kwargs : dict, optional
        Keyword arguments for the self-attention layer.
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = ClassTokenFGNN(
            filters,
            activation=activation,
            normalization=normalization,
            use_gates=use_gates,
            att_layer_kwargs=att_layer_kwargs,
            norm_kwargs=norm_kwargs,
            **kwargs_inner,
        )
        return lambda x: single_layer_call(x, layer, None, None, {})

    return Layer


class MaskedFGNN(FGNN):
    """
    Fingerprinting Graph Layer with Masked Attention.
    Parameters
    ----------
    filters : int
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def update_node_features(self, nodes, aggregated, learnable_embs, edges):
        Combined = self.combine_layer([nodes, aggregated])
        updated_nodes = self.update_norm(
            self.update_layer(Combined, edges=edges)
        )
        return updated_nodes


@register("MaskedFGNN")
def MaskedFGNNlayer(
    activation=GELU,
    normalization="LayerNormalization",
    use_gates=True,
    att_layer_kwargs={},
    norm_kwargs={},
    **kwargs,
):
    """
    Fingerprinting Graph Layer with Masked Attention.
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
    combine_layer : layer, optional
        Layer to combine node features and aggregated messages.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = MaskedFGNN(
            filters,
            activation=activation,
            normalization=normalization,
            use_gates=use_gates,
            att_layer_kwargs=att_layer_kwargs,
            norm_kwargs=norm_kwargs,
            **kwargs_inner,
        )
        return lambda x: single_layer_call(x, layer, None, None, {})

    return Layer


class GraphTransformer(tf.keras.layers.Layer):
    """Graph Transformer.
    Parameters
    ----------
    fwd_mlp_dim : int
        Dimension of the forward MLP.
    number_of_heads : int
        Number of attention heads.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    use_bias: bool, optional
        Whether to use bias in the dense layers of the attention layers. Defaults to False.
    dropout : float
        Dropout rate.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        fwd_mlp_dim,
        number_of_heads=12,
        activation=GELU,
        normalization="LayerNormalization",
        use_bias=True,
        clip_scores_by_value=(-5.0, 5.0),
        dropout=0.0,
        norm_kwargs={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.number_of_heads = number_of_heads
        self.use_bias = use_bias

        self.fwd_mlp_dim = fwd_mlp_dim
        self.dropout = dropout

        self.activation = activation
        self.normalization = normalization

        self.clip_scores_by_value = clip_scores_by_value

        self.MaskedMultiHeadAttLayer = MultiHeadSelfAttention(
            number_of_heads=self.number_of_heads,
            use_bias=self.use_bias,
            name="MaskedMultiHeadAttLayer",
            clip_scores_by_value=self.clip_scores_by_value,
        )
        self.norm_0, self.norm_1 = (
            as_normalization(normalization)(**norm_kwargs),
            as_normalization(normalization)(**norm_kwargs),
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def build(self, input_shape):
        input_shape, *_ = input_shape

        self.feed_forward_layer = tf.keras.Sequential(
            [
                layers.Dense(
                    self.fwd_mlp_dim,
                    name=f"{self.name}/Dense_0",
                ),
                as_activation(self.activation),
                layers.Dropout(self.dropout),
                layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                layers.Dropout(self.dropout),
            ],
            name="feed_forward",
        )

    def call(self, inputs, training):
        nodes, edges = inputs

        x = self.MaskedMultiHeadAttLayer(nodes, edges=edges)
        x = self.dropout_layer(x, training=training)
        x = self.norm_0(nodes + x)

        y = self.feed_forward_layer(x)
        return self.norm_1(x + y), edges


@register("GraphTransformerLayer")
def GraphTransformerLayer(
    number_of_heads=6,
    activation=GELU,
    normalization="LayerNormalization",
    use_bias=True,
    clip_scores_by_value=(-5.0, 5.0),
    dropout=0.0,
    norm_kwargs={},
    **kwargs,
):
    """Graph Transformer Layer.
    Parameters
    ----------
    number_of_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    use_gates : bool, optional
        Whether to use gated self-attention layers as update layer. Defaults to False.
    use_bias: bool, optional
        Whether to use bias in the dense layers of the attention layers. Defaults to True.
    norm_kwargs : dict
        Arguments for the normalization function.
    kwargs : dict
        Additional arguments.
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = GraphTransformer(
            filters,
            number_of_heads,
            activation,
            normalization,
            use_bias,
            clip_scores_by_value,
            dropout,
            norm_kwargs,
            **kwargs,
        )
        return lambda x: single_layer_call(x, layer, None, None, {})

    return Layer
