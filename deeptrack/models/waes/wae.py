import tensorflow as tf
from tensorflow.keras import layers
from ..utils import as_KerasModel


class WAE(tf.keras.Model):
    """Wasserstein Autoencoder based on either Generative Adversarial Network (WAE-GAN) model or maximum mean discrepancy (WAE-MMD).

    Parameters:
    regularizer: 'mmd' or 'gan', default is mmd
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
        regularizer="mmd",
        encoder=None,
        decoder=None,
        discriminator=None,
        latent_dim=8,
        lambda_=10.0,
        sigma_z=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define latent dimension, and hyperparameters
        self.regularizer = regularizer
        self.latent_dim = latent_dim  # For probabilistic encoder set as 2*latent_dim
        self.lambda_ = lambda_
        self.sigma_z = sigma_z

        # Initialize encoder, decoder, and discriminator, or use defaults
        if encoder is None:
            encoder = self.default_encoder()

        if decoder is None:
            decoder = self.default_decoder()

        if self.regularizer=="gan":
            if discriminator is None:
                discriminator = self.default_discriminator()

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def compile(
        self, enc_optimizer=None, dec_optimizer=None, disc_optimizer=None, loss_fn=None
    ):
        super().compile()

        # Set optimizers and loss function for training if not provided
        if enc_optimizer is None:
            enc_optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-3, beta_1=0.5, beta_2=0.999
            )

        if dec_optimizer is None:
            dec_optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-3, beta_1=0.5, beta_2=0.999
            )

        if self.regularizer=="gan":
            if disc_optimizer is None:
                disc_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=5e-4, beta_1=0.5, beta_2=0.999
                )

        if loss_fn is None:
            loss_fn = tf.keras.losses.MeanSquaredError()

        self.enc_optim = enc_optimizer
        self.dec_optim = dec_optimizer
        self.disc_optim = disc_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def mmd_penalty(self, pz, qz, batch_size):
        # Estimator of the MMD with the IMQ kernel as in
        # https://github.com/tolstikhin/wae/blob/master/wae.py#L233 and in
        # https://github.com/w00zie/wae_mnist/blob/master/train_mmd.py

        # Here the property that the sum of positive definite kernels is
        # still a p.d. kernel is used. Various kernels calculated at different
        # scales are summed together in order to "simultaneously look at various
        # scales" [https://github.com/tolstikhin/wae/issues/2].

        norms_pz = tf.reduce_sum(tf.square(pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(pz, pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2.0 * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(qz, qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2.0 * dotprods_qz

        dotprods = tf.matmul(qz, pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2.0 * dotprods

        cbase = tf.constant(2.0 * self.latent_dim * self.sigma_z)
        stat = tf.constant(0.0)
        nf = tf.cast(batch_size, dtype=tf.float32)

        for scale in tf.constant([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]):
            C = cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1.0 - tf.eye(batch_size))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2.0 / (nf * nf)
            stat += res1 - res2
        return stat

    def train_step(self, data):
        data, _ = data

        batch_size = tf.shape(data)[0]

        # Gradient tape for automatic differentiation
        with tf.GradientTape(persistent=True) as tape:
            # Encode input data
            q_z = self.encoder(data)
            # For probabilistic encoder sample from latent space
            # z_mean, z_log_var = tf.split(self.encoder(data), 2, axis=1)
            # epsilon = tf.random.normal(shape=tf.shape(z_mean))
            # q_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

            # Decode sampled points and compute discriminator output
            x_hat = self.decoder(q_z)

            # Compute reconstruction loss
            rloss = self.loss_fn(data, x_hat)

            # Compute penalty for regularization
            if self.regularizer=="gan":
                d_qz = self.discriminator(q_z)
                penalty = tf.keras.losses.binary_crossentropy(
                    tf.ones_like(d_qz), d_qz
                )
            elif self.regularizer=="mmd":
                p_z = tf.random.normal(
                    shape=(batch_size, self.latent_dim),
                    stddev=tf.sqrt(self.sigma_z),
                )
                penalty = self.mmd_penalty(p_z, q_z, batch_size)
            loss = rloss + tf.reduce_mean(self.lambda_ * penalty)

        # Compute gradients and update encoder and decoder weights
        enc_grads = tape.gradient(loss, self.encoder.trainable_weights)
        dec_grads = tape.gradient(loss, self.decoder.trainable_weights)
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_weights))
        self.dec_optim.apply_gradients(zip(dec_grads, self.decoder.trainable_weights))

        # Sample points from the latent space for the discriminator
        if self.regularizer=="gan":
            with tf.GradientTape() as tape:
                p_z = tf.random.normal(
                    shape=(batch_size, self.latent_dim),
                    stddev=tf.sqrt(self.sigma_z),
                )
                d_pz = self.discriminator(p_z)

                q_z = self.encoder(data)
                # For probabilistic encoder sample from latent space
                # z_mean, z_log_var = tf.split(self.encoder(data), 2, axis=1)
                # epsilon = tf.random.normal(shape=tf.shape(z_mean))
                # q_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
                d_qz = self.discriminator(q_z)

                # Compute losses for real and fake samples and discriminator loss
                real_loss = tf.keras.losses.binary_crossentropy(
                    tf.ones_like(d_pz), d_pz
                )
                fake_loss = tf.keras.losses.binary_crossentropy(
                    tf.zeros_like(d_qz), d_qz
                )
                disc_loss = self.lambda_ * (
                    tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
                )

            # Compute gradients and update discriminator weights
            disc_grads = tape.gradient(
                disc_loss, self.discriminator.trainable_weights
            )
            self.disc_optim.apply_gradients(
                zip(disc_grads, self.discriminator.trainable_weights)
            )

        # Update metrics for visualization
        self.compiled_metrics.update_state(data, x_hat)

        # Return various loss values for monitoring
        if self.regularizer=="gan":
            return {
                "loss": loss,
                "reconstruction_loss": rloss,
                "discriminator_loss": disc_loss,
            }
        elif self.regularizer=="mmd":
            return {
                "loss": loss,
                "reconstruction_loss": rloss,
            }

    def call(self, inputs):
        # Use encoder to obtain latent representation
        return self.encoder(inputs)

    def default_encoder(self):
        # Define the default encoder architecture
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(
                    128,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    256,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    1024,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Flatten(),
                layers.Dense(self.latent_dim),
            ],
            name="encoder",
        )

    def default_decoder(self):
        # Define the default decoder architecture
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.latent_dim,)),
                layers.Dense(7 * 7 * 1024),
                layers.Reshape((7, 7, 1024)),
                layers.Conv2DTranspose(
                    512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2DTranspose(
                    256,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    1,
                    kernel_size=4,
                    padding="same",
                ),
            ],
            name="decoder",
        )

    def default_discriminator(self):
        # Define the default discriminator architecture for WAE_GAN
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.latent_dim,)),
                layers.Dense(512),
                layers.ReLU(),
                layers.Dense(512),
                layers.ReLU(),
                layers.Dense(512),
                layers.ReLU(),
                layers.Dense(512),
                layers.ReLU(),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
