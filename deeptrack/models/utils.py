from functools import wraps, reduce

import numpy as np
from tensorflow.keras import layers, models

from .. import features
from ..generators import ContinuousGenerator


try:
    import tensorflow_addons as tfa

    InstanceNormalization = tfa.layers.InstanceNormalization
    GELU = layers.Lambda(lambda x: tfa.activations.gelu(x, approximate=False))
except Exception:
    import warnings

    InstanceNormalization, GELU = (layers.Layer(),) * 2
    warnings.warn(
        "DeepTrack not installed with tensorflow addons. Instance normalization and GELU activation will not work. Consider upgrading to tensorflow >= 2.0.",
        ImportWarning,
    )

import pkg_resources

installed_pkg = [pkg.key for pkg in pkg_resources.working_set]

__all__ = [
    "compile",
    "load_model",
    "Model",
    "KerasModel",
    "LoadModel",
    "single_layer_call",
    "as_activation",
    "as_normalization",
    "GELU",
    "InstanceNormalization",
]


def compile(model: models.Model, *, loss="mae", optimizer="adam", metrics=[], **kwargs):
    """Compiles a model.

    Parameters
    ----------
    model : keras.models.Model
        The keras model to interface.
    loss : str or keras loss
        The loss function of the model.
    optimizer : str or keras optimizer
        The optimizer of the model.
    metrics : list, optional
    """

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def LoadModel(path, compile_from_file=False, custom_objects={}, **kwargs):
    """Loads a keras model from disk.

    Parameters
    ----------
    path : str
        Path to the keras model to load.
    compile_from_file : bool
        Whether to compile the model using the loss and optimizer in the saved model. If false,
        it will be compiled from the arguments in kwargs (loss, optimizer and metrics).
    custom_objects : dict
        Dict of objects to use when loading the model. Needed to load a model with a custom loss,
        optimizer or metric.
    """
    model = models.load_model(
        path, compile=compile_from_file, custom_objects=custom_objects
    )
    return KerasModel(model, compile=not compile_from_file, **kwargs)


load_model = LoadModel


def as_activation(x):
    """Converts a string or a callable to a keras activation function."""
    if x is None:
        return layers.Layer()
    elif isinstance(x, str):
        return layers.Activation(x)
    elif isinstance(x, layers.Layer):
        return x
    else:
        return layers.Layer(x)


def _get_norm_by_name(x):
    """Returns a normalization layer by name."""
    if hasattr(layers, x):
        return getattr(layers, x)
    elif "tensorflow-addons" in installed_pkg and hasattr(tfa.layers, x):
        return getattr(tfa.layers, x)
    else:
        raise ValueError(f"Unknown normalization {x}.")


def as_normalization(x):
    """Converts a string or a callable to a keras normalization function."""
    if x is None:
        return layers.Layer()
    elif isinstance(x, str):
        return _get_norm_by_name(x)
    elif isinstance(x, layers.Layer) or callable(x):
        return x
    else:
        return layers.Layer(x)


def single_layer_call(
    x, layer, activation, normalization, norm_kwargs, activation_first=True
):
    """Calls a layer with activation and normalization."""
    assert isinstance(norm_kwargs, dict), "norm_kwargs must be a dict. Got {0}".format(
        type(norm_kwargs)
    )

    n = (
        lambda x: as_normalization(normalization)(**norm_kwargs)(x)
        if normalization
        else x
    )
    a = lambda x: as_activation(activation)(x) if activation else x
    fs = [layer, a, n] if activation_first else [layer, n, a]

    return reduce(lambda x, f: f(x), fs, x)


def with_citation(citation):
    """Decorator to add a citation to a model."""

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            assert isinstance(
                res, features.Feature
            ), "Wrapped model is not a deeptrack object. Did you forget @as_KerasModel?"
            res.citation = citation

            return res

        return inner

    return wrapper


def as_KerasModel(func):
    """Decorator to convert a model to a KerasModel."""

    @wraps(func)
    def inner(*args, **kwargs):
        model = func(*args, **kwargs)
        return KerasModel(model, **kwargs)

    return inner


def register_config(config_name, cfg):
    """Registers a config (set of arguments / hyperparameters) to be used by the KerasModel."""

    def wrapper(func):
        @wraps(func)
        def inner(*args, config=None, **kwargs):
            if config_name == config:
                kwargs = {**cfg, **kwargs}
            return func(*args, config=config, **kwargs)

        return inner

    return wrapper


class Model(features.Feature):
    """Base class wrapper for models."""

    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def __getattr__(self, key):
        # Allows access to the model methods and properties
        try:
            return getattr(super(), key)
        except AttributeError:
            return getattr(self.model, key)


class KerasModel(Model):
    """Wrapper for keras models.

    Allows keras models to be used as features. I.e the model can be used as a feature in a pipeline.
    Also wraps the fit method to automatically create a generator for training.
    Compiles the model using the loss and optimizer defined in the constructor.
    """

    data_generator = ContinuousGenerator

    def __init__(
        self,
        model,
        loss="mae",
        optimizer="adam",
        metrics=[],
        compile=True,
        add_batch_dimension_on_resolve=True,
        **kwargs,
    ):

        if compile:
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        super().__init__(
            model,
            add_batch_dimension_on_resolve=add_batch_dimension_on_resolve,
            metrics=metrics,
            **kwargs,
        )

    @wraps(models.Model.fit)
    def fit(self, x, *args, batch_size=32, generator_kwargs={}, **kwargs):
        if isinstance(x, features.Feature):
            generator = self.data_generator(
                x,
                batch_size=batch_size,
                **{
                    **{
                        "min_data_size": batch_size * 20,
                        "max_data_size": batch_size * 50,
                    },
                    **generator_kwargs,
                },
            )
            with generator:
                h = self.model.fit(generator, *args, **kwargs)
                return h
            # Code is not actually unreachable if fit crashes.
            return None

        return self.model.fit(x, *args, **kwargs)

    def export(
        self,
        path,
        minimum_size,
        preprocessing=None,
        dij_config=None,
    ):
        """Export model unto the BioImage Model Zoo format for use with Fiji and ImageJ.

        Uses pyDeepImageJ by E. Gómez-de-Mariscal, C. García-López-de-Haro, L. Donati, M. Unser,
        A. Muñoz-Barrutia and D. Sage for exporting.

        DeepImageJ, used for loading the models into ImageJ, is only compatible with
        tensorflow==2.2.1. Models using newer features may not load correctly.

        Pre-processing of the data should be defined when creating the model using the preprocess
        keyword. Post-processing should be left to other imageJ functionality. If this is not
        sufficient, see `https://github.com/deepimagej/pydeepimagej` for what to pass to the
        preprocessing and postprocessing arguments.

        Parameters
        ----------
        path : str
           Path to store exported files.
        minimum_size : int
           For models where the input size is not fixed (e.g. (None, None 1)), the input
           is required to be a multiple of this value.
        preprocessing : Feature or Layer
           Additional preprocessing. Will be saved as a part of the network, and as
           such need to be compatible with tensorflow tensor operations. Assumed to have the
           same input shape as the first layer of the network.
        dij_config : BioImageModelZooConfig, optional
            Configuration used for deployment. See `https://github.com/deepimagej/pydeepimagej` for
            list of options. If None, a basic config is created for you.
        """
        from pydeepimagej.yaml import BioImageModelZooConfig

        # TODO: Does not yet fully work as intended. Debugging proved to be hard.
        inp = layers.Input(shape=self.model.layers[0].input_shape)
        model = self.model

        if preprocessing:
            processed_inp = preprocessing(inp)
            model = model(processed_inp)
            model = models.Model(inp, model)

        dij_config = BioImageModelZooConfig(model, minimum_size)
        dij_config.Name = "DeepTrack 2.1 model"

        dij_config.add_weights_formats(model, "Tensorflow", authors=dij_config.Authors)
        dij_config.export_model(path)

    def get(self, image, add_batch_dimension_on_resolve, **kwargs):
        if add_batch_dimension_on_resolve:
            image = np.expand_dims(image, axis=0)

        return self.model.predict(image)

    def add_preprocessing(self, other, input_shape="same"):

        # TODO: Add tests for this.
        if input_shape == "same":
            input_shape = self.model.input_shape

        layer = layers.Lambda(other)

        i = layers.Input(input_shape=input_shape)
        p = layer(i)
        o = self.model(p)

        self.model = models.Model(i, o)

        return self

    def __rrshift__(self, other):
        return self.add_preprocessing(other)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
