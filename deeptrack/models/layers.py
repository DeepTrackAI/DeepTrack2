""" Standardized layers implemented in keras.
"""


from warnings import WarningMessage
from tensorflow.keras import layers

try:
    import tensorflow_addons as tfa

    InstanceNormalization = tfa.layers.InstanceNormalization
except Exception:
    import warnings

    InstanceNormalization = layers.Layer
    warnings.warn(
        "DeepTrack not installed with tensorflow addons. Instance normalization will not work. Consider upgrading to tensorflow >= 2.0.",
        ImportWarning,
    )


BLOCKS = {}


def register(*names):
    "Register a block to a name for use in models."

    def decorator(block):
        for name in names:
            if name in BLOCKS:
                warnings.warn(
                    f"Overriding registered block {name} with a new block.",
                    WarningMessage,
                )

            BLOCKS[name] = block()
        return block

    return decorator


def as_block(x):
    """Converts input to layer block"""
    if isinstance(x, str):
        if x in BLOCKS:
            return BLOCKS[x]
        else:
            raise ValueError(
                "Invalid blockname {0}, valid names are: ".format(x)
                + ", ".join(BLOCKS.keys())
            )
    if isinstance(x, layers.Layer) or not callable(x):
        raise TypeError("Layer block should be a function that returns a keras Layer.")
    else:
        return x


def _as_activation(x):
    if x is None:
        return layers.Layer()
    elif isinstance(x, str):
        return layers.Activation(x)
    elif isinstance(x, layers.Layer):
        return x
    else:
        return layers.Layer(x)


def _get_norm_by_name(x):
    if hasattr(layers, x):
        return getattr(layers, x)
    elif hasattr(tfa.layers, x):
        return getattr(tfa.layers, x)
    else:
        raise ValueError(f"Unknown normalization {x}.")


def _as_normalization(x):
    if x is None:
        return layers.Layer()
    elif isinstance(x, str):
        return _get_norm_by_name(x)
    elif isinstance(x, layers.Layer) or callable(x):
        return x
    else:
        return layers.Layer(x)


def _single_layer_call(x, layer, activation, normalization, norm_kwargs):
    assert isinstance(norm_kwargs, dict), "norm_kwargs must be a dict. Got {0}".format(
        type(norm_kwargs)
    )
    y = layer(x)

    if normalization:
        y = _as_normalization(normalization)(**norm_kwargs)(y)

    if activation:
        y = _as_activation(activation)(y)

    return y


@register("convolutional", "conv")
def ConvolutionalBlock(
    kernel_size=3,
    activation="relu",
    padding="same",
    strides=1,
    normalization=False,
    norm_kwargs={},
    **kwargs,
):
    """A single 2d convolutional layer.

    Accepts arguments of keras.layers.Conv2D.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    padding : str
        How to pad the input tensor. See keras docs for accepted strings.
    strides : int
        Step length of kernel
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    **kwargs
        Other keras.layers.Conv2D arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            **kwargs_inner,
        )
        return lambda x: _single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer


@register("dense")
def DenseBlock(activation="relu", normalization=False, norm_kwargs={}, **kwargs):
    """A single dense layer.

    Accepts arguments of keras.layers.Dense.

    Parameters
    ----------
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    **kwargs
        Other keras.layers.Dense arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = layers.Dense(filters, **kwargs_inner)
        return lambda x: _single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer


@register("pool", "pooling")
def PoolingBlock(
    pool_size=(2, 2),
    activation=None,
    padding="same",
    strides=2,
    normalization=False,
    norm_kwargs={},
    **kwargs,
):
    """A single max pooling layer.

    Accepts arguments of keras.layers.MaxPool2D.

    Parameters
    ----------
    pool_size : int
        Size of the pooling kernel
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    padding : str
        How to pad the input tensor. See keras docs for accepted strings.
    strides : int
        Step length of kernel
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    **kwargs
        Other keras.layers.MaxPool2D arguments
    """

    def Layer(filters=None, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = layers.MaxPool2D(
            pool_size=pool_size, padding=padding, strides=strides, **kwargs_inner
        )
        return lambda x: _single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer


@register("deconvolutional", "deconv")
def DeconvolutionalBlock(
    kernel_size=(2, 2),
    activation=None,
    padding="valid",
    strides=2,
    normalization=False,
    norm_kwargs={},
    **kwargs,
):
    """A single 2d deconvolutional layer.

    Accepts arguments of keras.layers.Conv2DTranspose.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    padding : str
        How to pad the input tensor. See keras docs for accepted strings.
    strides : int
        Step length of kernel
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    **kwargs
        Other keras.layers.Conv2DTranspose arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = layers.Conv2DTranspose(
            filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            **kwargs_inner,
        )
        return lambda x: _single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer


@register("upsample")
def StaticUpsampleBlock(
    size=(2, 2),
    activation=None,
    interpolation="bilinear",
    normalization=False,
    kernel_size=(1, 1),
    strides=1,
    padding="same",
    with_conv=True,
    norm_kwargs={},
    **kwargs,
):
    """A single no-trainable 2d deconvolutional layer.

    Accepts arguments of keras.layers.UpSampling2D.

    Parameters
    ----------
    size : int
        Size of the kernel
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    interpolation
        Interpolation type. Either "bilinear" or "nearest".
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    **kwargs
        Other keras.layers.Conv2DTranspose arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = layers.UpSampling2D(size=size, interpolation=interpolation)

        conv = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            **kwargs_inner,
        )

        def call(x):
            y = layer(x)
            if with_conv:
                return _single_layer_call(
                    y, conv, activation, normalization, norm_kwargs
                )
            else:
                return layer(x)

        return call

    return Layer


@register("residual")
def ResidualBlock(
    kernel_size=(3, 3),
    activation="relu",
    strides=1,
    normalization="InstanceNormalization",
    norm_kwargs={},
    **kwargs,
):
    """A 2d residual layer with two convolutional steps.

    Accepts arguments of keras.layers.Conv2D.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    strides : int
        Step length of kernel
    normalization : str or normalization function or layer
        Normalization function of the layer. See keras and tfa docs for accepted strings.
    norm_kwargs : dict
        Arguments for the normalization function.
    **kwargs
        Other keras.layers.Conv2D arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        identity = layers.Conv2D(filters, kernel_size=(1, 1))

        conv = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
        )

        conv2 = layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides, padding="same"
        )

        def call(x):
            y = _single_layer_call(x, conv, activation, normalization, norm_kwargs)
            y = _single_layer_call(y, conv2, None, normalization, norm_kwargs)
            y = layers.Add()([identity(x), y])
            if activation:
                y = _as_activation(activation)(y)
            return y

        return call

    return Layer


@register("none", "identity", "None")
def Identity(activation=None, normalization=False, norm_kwargs={}, **kwargs):
    """Identity layer that returns the input tensor.

    Can optionally perform instance normalization or some activation function.

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
        layer = layers.Layer(**kwargs_inner)
        return lambda x: _single_layer_call(
            x, layer, activation, normalization, norm_kwargs
        )

    return Layer
