import tensorflow as tf
from tensorflow.keras import layers

from ..utils import as_KerasModel

@as_KerasModel
class WAE_GAN(tf.keras.Model):
    def __init__(self, input_shape = (28, 28, 1), encoder=None, decoder=None, discriminator=None, latent_dim=2, lambda_ = 10.0, sigma_z = 1.0, **kwargs):
        super().__init__(**kwargs)

        # Input shape
        self.input_enc = input_shape
        
        # Dimensionality of the latent space
        self.latent_dim = latent_dim
        
        # Other parameters
        self.lambda_ = lambda_
        self.sigma_z = sigma_z

        #Optimizers
        self.enc_optim = tf.keras.optimizers.Adam()
        self.dec_optim = tf.keras.optimizers.Adam()
        self.disc_optim = tf.keras.optimizers.Adam()

        if encoder is None:
            self.encoder = self.default_encoder()

        if decoder is None:
            self.decoder = self.default_decoder()

        if discriminator is None:
            self.discriminator = self.default_discriminator()

    def train_step(self, data):

        data, _ = data

        with tf.GradientTape(persistent=True) as tape:
            # non-deterministic encoder
            z_mean, z_log_var = tf.split(self.encoder(data), 2, axis=1)
            # Sample a random point in the latent space
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            q_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            x_hat = self.decoder(q_z)
            d_qz = self.discriminator(q_z)
             # Reconstruction loss
            rloss = self.loss(data, x_hat)
            penalty = tf.keras.losses.binary_crossentropy(tf.ones_like(d_qz), d_qz)
            loss = rloss + tf.reduce_mean(self.lambda_*penalty)

        # Compute gradients
        enc_grads = tape.gradient(loss, self.encoder.trainable_weights)
        dec_grads = tape.gradient(loss, self.decoder.trainable_weights)

        # Update weights
        self.enc_optim.apply_gradients(zip(enc_grads, self.encoder.trainable_weights))
        self.dec_optim.apply_gradients(zip(dec_grads, self.decoder.trainable_weights))

        batch_size = tf.shape(data)[0]
        with tf.GradientTape() as tape:
            pz = tf.random.normal(shape=(batch_size, self.latent_dim), stddev=tf.sqrt(self.sigma_z))
            d_pz = self.discriminator(pz)
            z_mean, z_log_var = tf.split(self.encoder(data), 2, axis=1)
            # Sample a random point in the latent space
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            q_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            d_qz = self.discriminator(q_z)

            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(d_pz), d_pz)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(d_qz), d_qz)
            disc_loss = self.lambda_*(tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss))

        disc_grads = tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.disc_optim.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))

        # Update metrics
        self.compiled_metrics.update_state(data, x_hat)

        return {
            "loss": loss,
            "reconstruction_loss": rloss,
            "discriminator_loss": disc_loss,
        }


    def call(self, inputs):
        return self.encoder(inputs)

    def default_encoder(self):
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
                layers.Dense(
                    self.latent_dim + self.latent_dim , name="z_mean_log_var"
                ),
            ],
            name="encoder",
        )

    def default_decoder(self):
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
        return tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.latent_dim,)),
                layers.Dense(512, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(1,activation = 'sigmoid')
            ],
            name="discriminator",
        )