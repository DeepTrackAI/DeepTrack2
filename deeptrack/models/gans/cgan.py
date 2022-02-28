import tensorflow as tf
from ..utils import as_KerasModel

layers = tf.keras.layers


@as_KerasModel
class CGAN(tf.keras.Model):
    """Creates and compiles a conditional generative adversarial
    neural network (CGAN).
    Parameters
    ----------
    generator: keras model
        The generator network
    discriminator: keras model
        The discriminator network
    discriminator_loss: str or keras loss function
        The loss function of the discriminator network
    discriminator_optimizer: str or keras optimizer
        The optimizer of the discriminator network
    discriminator_metrics: list, optional
        List of metrics to be evaluated by the discriminator
        model during training and testing
    assemble_loss: list of str or keras loss functions
        List of loss functions to be evaluated on each output
        of the assemble model (stacked generator and discriminator),
        such as `assemble_loss = ["mse", "mse", "mae"]` for
        the prediction of the discriminator, the predicted
        perceptual features, and the generated image, respectively
    assemble_optimizer: str or keras optimizer
        The optimizer of the assemble network
    assemble_loss_weights: list or dict, optional
        List or dictionary specifying scalar coefficients (floats)
        to weight the loss contributions of the assemble model outputs
    metrics: list, optional
        List of metrics to be evaluated on the generated images during
        training and testing
    """

    def __init__(
        self,
        generator=None,
        discriminator=None,
        discriminator_loss=None,
        discriminator_optimizer=None,
        discriminator_metrics=None,
        assemble_loss=None,
        assemble_optimizer=None,
        assemble_loss_weights=None,
        metrics=[],
        **kwargs
    ):
        super().__init__()

        # Build and compile the discriminator
        self.discriminator = discriminator
        self.discriminator.compile(
            loss=discriminator_loss,
            optimizer=discriminator_optimizer,
            metrics=discriminator_metrics,
        )

        # Build the generator
        self.generator = generator

        # Input shape
        self.model_input = self.generator.input

        # The generator model_input and generates img
        img = self.generator(self.model_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes the generated images as input and determines validity
        validity = self.discriminator([img, self.model_input])

        # The assembled model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.assemble = tf.keras.models.Model(self.model_input, [validity, img])

        self.num_losses = len(assemble_loss)

        self.assemble.compile(
            loss=assemble_loss,
            optimizer=assemble_optimizer,
            loss_weights=assemble_loss_weights,
            metrics=metrics,
        )
        self._metrics = [tf.metrics.get(m) for m in metrics]

    def train_step(self, data):

        # Compute data and labels
        batch_x, batch_y = data
        gen_imgs = self.generator(batch_x)

        # Train the discriminator
        with tf.GradientTape() as tape:
            # Train in two steps
            disc_pred_real = self.discriminator([batch_y, batch_x])
            disc_pred_fake = self.discriminator([gen_imgs, batch_x])
            shape = tf.shape(disc_pred_real)
            valid, fake = tf.ones(shape), tf.zeros(shape)
            d_loss = (
                self.discriminator.compiled_loss(disc_pred_real, valid)
                + self.discriminator.compiled_loss(disc_pred_fake, fake)
            ) / 2

        # Compute gradient and apply gradient
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Train the assembly

        with tf.GradientTape() as tape:
            assemble_output = self.assemble(batch_x)

            generated_image_copies = [assemble_output[1]] * (self.num_losses - 1)

            batch_y_copies = [batch_y] * (self.num_losses - 1)

            g_loss = self.assemble.compiled_loss(
                [assemble_output[0], *generated_image_copies],
                [valid, *batch_y_copies],
            )

        # Compute gradient and apply gradient
        grads = tape.gradient(g_loss, self.assemble.trainable_weights)
        self.assemble.optimizer.apply_gradients(
            zip(grads, self.assemble.trainable_weights)
        )

        # Update the metrics
        self.compiled_metrics.update_state(assemble_output[1], batch_y)

        # Define output
        loss = {
            "d_loss": d_loss,
            "g_loss": g_loss,
            **{m.name: m.result() for m in self.metrics},
        }

        return loss

    def call(self, *args, **kwargs):
        return self.generator.call(*args, **kwargs)
