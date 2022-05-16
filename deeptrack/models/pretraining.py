"""Includes methods for pretraining neural networks. 
"""

import random
import numpy
import tensorflow as tf
import tqdm
import numpy as np
from .. import features, generators


class EMA:
    """Exponentially weighted moving average.

    Parameters
    ----------
    beta : float
        The exponential decay factor.
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new

        return tuple(
            [old[i] * self.beta + new[i] * (1 - self.beta) for i in range(len(old))]
        )


def _update_moving_average(ema_updater, student_model, teacher_model):
    # Updates the weights of the teacher model with the moving average of the student model.
    for student_layer, teacher_layer in zip(student_model.layers, teacher_model.layers):
        new_weights, old_weights = (
            student_layer.get_weights(),
            teacher_layer.get_weights(),
        )
        new_weights = ema_updater.update_average(old_weights, new_weights)
        teacher_layer.set_weights(new_weights)


class PixelwiseContrastiveTraining:
    """Pixelwise pretraining.

    Unsupervised pretraining of neural networks by contrastive learning. The latent space is trained to
    contain high pixelwise contrast, while retaining low contrast for augmentations of the same input.

    The weights of the teacher netowrk are a exponential decay of the weights of the student network.

    Parameters
    ----------
    model : keras model
        The model to be pretrained. The output of the model is the latent space.
    augmenter : Feature
        The augmenter to be used for training. Should not contain global pixel mapping such as flipping, cropping, rotating, scaling etc.
    moving_average_decay : float, optional
        The decay rate of the teacher's moving average. Defaults to 0.99.
    min_roll : int, optional
        The minimum amount to roll the prediction. Defaults to 8.
    max_roll : int, optional
        The maximum amount to roll the prediction. Defaults to `prediction.shape - min_roll`.
    loss_weights : tuple, optional
        The weights to be used for the loss functions. Defaults to (0.5, 0.5).
    optimizer : str or keras optimizer, optional
        The optimizer to be used for training. Defaults to Adam.
    """

    def __init__(
        self,
        model,
        augmenter,
        moving_average_decay=0.99,
        min_roll=8,
        max_roll=None,
        loss_weights=(0.5, 0.5),
        optimizer="adam",
    ):
        self.model = model
        self.teacher_model = tf.keras.models.clone_model(model)

        self.augmenter = augmenter
        self.moving_average_decay = moving_average_decay
        self.min_roll = min_roll
        self.max_roll = max_roll
        self.optimizer = optimizer
        self.loss_weights = loss_weights

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.total_loss,
        )

    def fit(
        self, x, epochs=1, batch_size=32, steps_per_epoch=1000, verbose=1, **kwargs
    ):
        """Fit the model. The training data can be a numpy array, DeepTrack pipeline.

        Parameters
        ----------
        x : numpy array or DeepTrack Feature
            The training data.
        epochs : int, optional
            The number of epochs to train for. Defaults to 1.
        batch_size : int, optional
            The batch size to use. Defaults to 32.
        steps_per_epoch : int, optional
            The number of steps (batches) of training per epoch. Defaults to 1000 if the training data is a feature, and the number of samples in the training data if it is a numpy array.
        verbose : int, optional
            The verbosity level. Defaults to 1.
        """

        if isinstance(x, numpy.ndarray):
            x = features.Value(lambda: random.choice(x))
            steps_per_epoch = len(x)

        ema = EMA(self.moving_average_decay)

        for epoch in range(epochs):

            pbar = tqdm.tqdm(
                range(steps_per_epoch),
                disable=not verbose,
                desc="Epoch {}/{}".format(epoch, epochs),
            )

            for step in pbar:

                _x = x.update()()
                # Get the batch
                batch = numpy.array(
                    [self.augmenter.update()(_x._value) for _ in range(batch_size)]
                )

                # Get target from teacher
                target = self.teacher_model.predict(_x._value[np.newaxis])

                # Get prediction from student
                loss = self.model.train_on_batch(
                    batch, np.repeat(target, batch_size, axis=0)
                )

                if verbose > 0:
                    # Print the loss using tqdm
                    pbar.set_postfix(s=f"Loss: {loss:.4f}")

                # Update the moving average
                _update_moving_average(ema, self.model, self.teacher_model)

    def sim_loss(self, p, t):
        x = tf.norm(p, dim=-1, p=2)
        y = tf.norm(t, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def contrast_loss(self, p, t):

        shape = tf.shape(p)

        min_roll = self.min_roll
        if self.max_roll is None:
            max_roll = shape[1] - min_roll
        else:
            max_roll = self.max_roll

        to_roll = tf.random.uniform(
            shape=(2,), minval=min_roll, maxval=max_roll, dtype=tf.int32
        )
        latent_roll = tf.random.uniform(
            shape=(1,), minval=0, maxval=shape[-1], dtype=tf.int32
        )
        x = (tf.norm(p, dim=-1, p=2) + 1) / 2
        y = tf.roll(x, to_roll[0], axis=1)
        y = tf.roll(y, to_roll[1], axis=2)

        cross_entropy = (x - y) ** 2
        latent_entropy = tf.math.reduce_mean(tf.math.reduce_std(x, axis=-1))

        return tf.reduce_mean(-cross_entropy) - latent_entropy

    def total_loss(self, p, t):
        return self.loss_weights[0] * self.sim_loss(p, t) + self.loss_weights[
            1
        ] * self.contrast_loss(p, t)
