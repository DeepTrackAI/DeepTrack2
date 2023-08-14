import tensorflow as tf
from tensorflow.keras import layers
from ..utils import as_KerasModel

class WAE_GAN(tf.keras.Model):
    """Wasserstein Autoencoder Generative Adversarial Network (WAE-GAN) model.

    Parameters:
    input_shape: tuple, optional
        Shape of the input data.
    encoder: keras model, optional
        The encoder network.
    decoder: keras model, optional
        The decoder network.
    discriminator: keras model, optional
        The discriminator network.
    latent_dim: int, optional
        Dimension of the latent space.
    lambda_: float, optional
        Hyperparameter for regularization.
    sigma_z: float, optional
        Standard deviation for sampling in the latent space.
    """

    def __init__(
        self,
        input_shape=(28, 28, 1),
        encoder=None,
        decoder=None,
        discriminator=None,
        latent_dim=2,
        lambda_=10.0,
        sigma_z=1.0,
        **kwargs
    ):
        super(WAE_GAN, self).__init__(**kwargs)

        # Define input shape, latent dimension, and hyperparameters
        self.input_enc = input_shape
        self.latent_dim = latent_dim
        self.lambda_ = lambda_
        self.sigma_z = sigma_z

        # Initialize encoder, decoder, and discriminator, or use defaults
        if encoder is None:
            self.encoder = self.default_encoder()

        if decoder is None:
            self.decoder = self.default_decoder()

        if discriminator is None:
            self.discriminator = self.default_discriminator()

    def compile(
        self, enc_optimizer=None, dec_optimizer=None, disc_optimizer=None, loss_fn=None
    ):
        super(WAE_GAN, self).compile()

        # Set optimizers and loss function for training if not provided
        if enc_optimizer is None:
            self.enc_optim = tf.keras.optimizers.Adam(learning_rate=1e-3)

        if dec_optimizer is None:
            self.dec_optim = tf.keras.optimizers.Adam(learning_rate=1e-3)

        if disc_optimizer is None:
            self.disc_optim = tf.keras.optimizers.Adam(learning_rate=5e-4)

        if loss_fn is None:
            self.loss_fn = tf.keras.metrics.Mean(name="mse")

    def train_step(self, data):
        data, _ = data

        # Gradient tape for automatic differentiation
        with tf.GradientTape(persistent=True) as tape:
            # Encode input data and sample from latent space
            z_mean, z_log_var = tf.split(self.encoder(data), 2, axis=1)
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            q_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

            # Decode sampled points and compute discriminator output
            x_hat = self.decoder(q_z)
            d_qz = self.discriminator(q_z)

            # Compute reconstruction loss and penalty for regularization
            rloss = self.loss_fn(data, x_hat)
            penalty = tf.keras.losses.binary_crossentropy(tf.ones_like(d_qz), d_qz)
            loss = rloss + tf.reduce_mean(self.lambda_ * penalty)

        # Compute gradients and update encoder and decoder weights
        enc_grads = tape.gradient(loss, self.encoder.trainable_weights)
        dec_grads = tape.gradient(loss, self.decoder.trainable_weights)
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_weights))
        self.dec_optim.apply_gradients(zip(dec_grads, self.decoder.trainable_weights))

        # Sample points from the latent space for the discriminator
        batch_size = tf.shape(data)[0]
        with tf.GradientTape() as tape:
            pz = tf.random.normal(
                shape=(batch_size, self.latent_dim), stddev=tf.sqrt(self.sigma_z)
            )
            d_pz = self.discriminator(pz)
            z_mean, z_log_var = tf.split(self.encoder(data), 2, axis=1)
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            q_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            d_qz = self.discriminator(q_z)

            # Compute losses for real and fake samples and discriminator loss
            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(d_pz), d_pz)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(d_qz), d_qz)
            disc_loss = self.lambda_ * (
                tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
            )

        # Compute gradients and update discriminator weights
        disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_optim.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_weights)
        )

        # Update metrics for visualization
        self.compiled_metrics.update_state(data, x_hat)

        # Return various loss values for monitoring
        return {
            "loss": loss,
            "reconstruction_loss": rloss,
            "discriminator_loss": disc_loss,
        }

    def call(self, inputs):
        # Use encoder to obtain latent representation
        return self.encoder(inputs)

    def default_encoder(self):
        # Define the default encoder architecture
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.input_enc),
                layers.Conv2D(
                    32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(
                    64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dense(16),
                layers.LeakyReLU(alpha=0.2),
                layers.Dense(self.latent_dim + self.latent_dim, name="z_mean_log_var"),
            ],
            name="encoder",
        )

    def default_decoder(self):
        # Define the default decoder architecture
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.latent_dim,)),
                layers.Dense(7 * 7 * 64),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7, 64)),
                layers.Conv2DTranspose(
                    64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(
                    32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                ),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(
                    1,
                    kernel_size=3,
                    activation="sigmoid",
                    padding="same",
                ),
            ],
            name="decoder",
        )

    def default_discriminator(self):
        # Define the default discriminator architecture
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.latent_dim,)),
                layers.Dense(512, activation="relu"),
                layers.Dense(512, activation="relu"),
                layers.Dense(512, activation="relu"),
                layers.Dense(512, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
