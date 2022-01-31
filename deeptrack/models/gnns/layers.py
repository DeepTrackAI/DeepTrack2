import tensorflow as tf
from tensorflow.keras import layers

from ..layers import *

GraphDenseBlock = DenseBlock(activation=GELU, normalization="LayerNormalization")


class FGNN(tf.keras.layers.Layer):
    """
    Fingerprinting Graph Layer.
    Parameters
    ----------
    filters : int
        Number of filters.
    message_layer : str or callable
        Message layer.
    update_layer : str or callable
        Update layer.
    random_edge_dropout : float, optional
        Random edge dropout.
    kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        filters,
        message_layer=GraphDenseBlock,
        update_layer="MultiHeadGatedSelfAttention",
        random_edge_dropout=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.random_edge_dropout = random_edge_dropout

        self.message_layer = as_block(message_layer)(filters)
        self.update_layer = as_block(update_layer)(None)

    def build(self, input_shape):
        self.sigma = tf.Variable(
            initial_value=tf.constant_initializer(value=0.005)(
                shape=(1,), dtype="float32"
            ),
            name="sigma",
            trainable=True,
            constraint=lambda value: tf.clip_by_value(value, 0.002, 1),
        )

        self.beta = tf.Variable(
            initial_value=tf.constant_initializer(value=4)(shape=(1,), dtype="float32"),
            name="beta",
            trainable=True,
            constraint=lambda value: tf.clip_by_value(value, 1, 10),
        )

    def call(self, inputs):
        nodes, edge_features, distance, edges = inputs

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

        if self.random_edge_dropout:
            messages = tf.nn.dropout(
                messages,
                self.random_edge_dropout,
                noise_shape=(1, number_of_edges, 1),
            )

        # Compute edge weights and apply them to the messages
        # shape = (batch, nOfedges, filters)
        edge_weights = tf.math.exp(
            -1
            * tf.math.pow(
                tf.math.square(distance) / (2 * tf.math.square(self.sigma)),
                self.beta,
            )
        )
        weighted_messages = messages * edge_weights

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge = x

            merged_edges = tf.math.unsorted_segment_sum(
                message,
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (weighted_messages, edges),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        Combined = [nodes, aggregated]
        updated_nodes = self.update_layer(Combined)

        return (updated_nodes, weighted_messages, distance, edges)


@register("FGnn")
def FGNNlayer(
    message_layer=GraphDenseBlock,
    update_layer="MultiHeadGatedSelfAttention",
    random_edge_dropout=False,
    activation=None,
    normalization=None,
    norm_kwargs={},
    **kwargs,
):
    """Fingerprinting Graph Layer.
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
        layer = FGNN(
            filters,
            message_layer,
            update_layer,
            random_edge_dropout,
            **kwargs_inner,
        )
        return lambda x: single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer


class ClassTokenFGNN(FGNN):
    """
    Fingerprinting Graph Layer with Class Token.
    Parameters
    ----------
    filters : int
        Number of filters.
    message_layer : str or callable
        Message layer.
    update_layer : str or callable
        Update layer.
    random_edge_dropout : float, optional
        Random edge dropout.
    kwargs : dict
        Additional arguments.
    """

    def build(self, input_shape):
        self.combine_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1)),
                layers.Dense(self.filters),
            ]
        )
        super().build(input_shape)

    def call(self, inputs):
        nodes, edge_features, distance, edges = inputs

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

        if self.random_edge_dropout:
            messages = tf.nn.dropout(
                messages,
                self.random_edge_dropout,
                noise_shape=(1, number_of_edges, 1),
            )

        # Compute edge weights and apply them to the messages
        # shape = (batch, nOfedges, filters)
        edge_weights = tf.math.exp(
            -1
            * tf.math.pow(
                tf.math.square(distance) / (2 * tf.math.square(self.sigma)),
                self.beta,
            )
        )
        weighted_messages = messages * edge_weights

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge = x

            merged_edges = tf.math.unsorted_segment_sum(
                message,
                edge[:, 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (weighted_messages, edges),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        Combined = tf.concat(
            [class_token, self.combine_layer([nodes, aggregated])], axis=1
        )
        updated_nodes = self.update_layer(Combined)

        return (updated_nodes, weighted_messages, distance, edges)


def ClassTokenFGNNlayer(
    message_layer=GraphDenseBlock,
    update_layer="MultiHeadGatedSelfAttention",
    random_edge_dropout=False,
    activation=None,
    normalization=None,
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
            message_layer,
            update_layer,
            random_edge_dropout,
            **kwargs_inner,
        )
        return lambda x: single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer