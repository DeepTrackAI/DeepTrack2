import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import default
from .utils import as_KerasModel

layers = tf.keras.layers


@as_KerasModel
class PCGAN(tf.keras.Model):
    """Creates and compiles a conditional generative adversarial
       neural network (cgan) with an additional perceptual discriminator. 
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
        content_discriminator="DenseNet121",
        metrics=[],
        **kwargs
    ):
        super(PCGAN).__init__()

        # Build and compile the discriminator
        self.discriminator = discriminator
        self.discriminator.compile(
            loss=discriminator_loss,
            optimizer=discriminator_optimizer,
            metrics=discriminator_metrics,
        )

        # perceptual discriminator (pre-trained)
        self.content_discriminator = tf.keras.Sequential(
            [
                layers.Lambda(
                    lambda img: layers.Concatenate(axis=-1)([img] * 3)
                ),
                getattr(tf.keras.applications, content_discriminator)(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(*generator.output.shape[1:3], 3),
                ),
            ]
        )
        self.content_discriminator.layers[1].trainable = False

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

        # Compute content loss
        content = self.content_discriminator(img)

        # The assembled model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.assemble = tf.keras.models.Model(
            self.model_input, [validity, content, img]
        )

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
            disc_pred_1 = self.discriminator([batch_y, batch_x])
            disc_pred_2 = self.discriminator([gen_imgs, batch_x])
            shape = tf.shape(disc_pred_1)
            valid, fake = tf.ones(shape), tf.zeros(shape)
            d_loss = (
                self.discriminator.compiled_loss(disc_pred_1, valid)
                + self.discriminator.compiled_loss(disc_pred_2, fake)
            ) / 2

        # Compute gradient and apply gradient
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Compute content loss
        content_valid = self.content_discriminator(batch_y)

        # Train the assembly
        with tf.GradientTape() as tape:
            assemble_output = self.assemble(batch_x)

            generated_image_copies = [assemble_output[2]] * (
                self.num_losses - 1
            )

            batch_y_copies = [batch_y] * (self.num_losses - 1)

            g_loss = self.assemble.compiled_loss(
                [
                    assemble_output[0],
                    assemble_output[1],
                    *generated_image_copies,
                ],
                [valid, content_valid, *batch_y_copies],
            )

        # Compute gradient and apply gradient
        grads = tape.gradient(g_loss, self.assemble.trainable_weights)
        self.assemble.optimizer.apply_gradients(
            zip(grads, self.assemble.trainable_weights)
        )

        # Update the metrics
        self.compiled_metrics.update_state(assemble_output[2], batch_y)

        # Define output
        loss = {
            "d_loss": d_loss,
            "g_loss": g_loss,
            **{m.name: m.result() for m in self.metrics},
        }

        return loss

    def call(self, *args, **kwargs):
        return self.generator.call(*args, **kwargs)
