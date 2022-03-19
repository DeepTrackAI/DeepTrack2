import tensorflow as tf

from .utils import single_layer_call
from .layers import register


class ClassToken(tf.keras.layers.Layer):
    """ClassToken Layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(
                shape=(1, 1, self.hidden_size), dtype="float32"
            ),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)


@register("ClassToken")
def ClassTokenLayer(
    activation=None, normalization=None, norm_kwargs={}, **kwargs
):
    """ClassToken Layer that append a class token to the input.

    Can optionally perform normalization or some activation function.

    Accepts arguments of keras.layers.Layer.

    Parameters
    ----------

    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    **kwargs
        Other keras.layers.Layer arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = ClassToken(**kwargs_inner)
        return lambda x: single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer


class LearnablePositionEmbs(tf.keras.layers.Layer):
    """Adds or concatenates positional embeddings to the inputs.
    Parameters
    ----------
    initializer : str or tf.keras.initializers.Initializer
        Initializer function for the embeddings. See tf.keras.initializers.Initializer for accepted functions.
    concat : bool
        Whether to concatenate the positional embeddings to the inputs. If False,
        adds the positional embeddings to the inputs.
    kwargs: dict
        Other arguments for the keras.layers.Layer
    """

    def __init__(
        self,
        initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
        concat=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.concat = concat

        assert initializer is callable or isinstance(
            initializer, tf.keras.initializers.Initializer
        ), "initial_value must be callable or a tf.keras.initializers.Initializer"
        self.initializer = initializer

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pos_embedding = tf.Variable(
            name="pos_embedding",
            initial_value=self.initializer(shape=(1, *(input_shape[-2:]))),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        if self.concat:
            return tf.concat(
                [inputs, tf.cast(self.pos_embedding, dtype=inputs.dtype)],
                axis=-1,
            )
        else:
            return inputs + tf.cast(self.pos_embedding, dtype=inputs.dtype)


@register("LearnablePositionEmbs")
def LearnablePositionEmbsLayer(
    initializer=tf.keras.initializers.RandomNormal(stddev=0.06),
    concat=False,
    activation=None,
    normalization=None,
    norm_kwargs={},
    **kwargs,
):
    """Adds or concatenates positional embeddings to the inputs.

    Can optionally perform normalization or some activation function.

    Accepts arguments of keras.layers.Layer.

    Parameters
    ----------

    initializer : str or tf.keras.initializers.Initializer
        Initializer function for the embeddings. See tf.keras.initializers.Initializer for accepted functions.
    concat : bool
        Whether to concatenate the positional embeddings to the inputs. If False,
        adds the positional embeddings to the inputs.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    **kwargs
        Other arguments for the keras.layers.Layer
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = LearnablePositionEmbs(
            initializer=initializer, concat=concat, **kwargs_inner
        )
        return lambda x: single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer


class LearnableDistanceEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.sigma = tf.Variable(
            initial_value=tf.constant_initializer(value=0.12)(
                shape=(1,), dtype="float32"
            ),
            name="sigma",
            trainable=True,
            constraint=lambda value: tf.clip_by_value(value, 0.002, 1),
        )

        self.beta = tf.Variable(
            initial_value=tf.constant_initializer(value=4)(
                shape=(1,), dtype="float32"
            ),
            name="beta",
            trainable=True,
            constraint=lambda value: tf.clip_by_value(value, 1, 10),
        )

    def call(self, inputs):
        return tf.math.exp(
            -1
            * tf.math.pow(
                tf.math.square(inputs) / (2 * tf.math.square(self.sigma)),
                self.beta,
            )
        )
