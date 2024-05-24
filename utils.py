import os
import keras
import numpy as np
import tensorflow as tf

from cGAN import conditionalGAN

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 32
latent_dim = 128


def load_dataset():
    """
    Loads the CIFAR10 dataset, preprocesses it, and returns it as a TensorFlow dataset.

    Returns:
        tf.data.Dataset: Dataset containing the preprocessed CIFAR10 images and labels.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    all_images = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    all_images = all_images.astype("float32") / 255.0
    all_labels = keras.utils.to_categorical(all_labels, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset


def build_models():
    """
    Builds the generator and discriminator models for the cGAN.

    Returns:
        keras.Model: Generator model.
        keras.Model: Discriminator model.
    """
    # - - - - - - - Calculate the number of input channels - - - - - - -
    gen_channels = latent_dim + num_classes
    dis_channels = num_channels + num_classes

    # - - - - - - - Generator - - - - - - -
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((gen_channels,)),
            keras.layers.Dense(7 * 7 * gen_channels),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Reshape((7, 7, gen_channels)),
            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2DTranspose(
                batch_size, kernel_size=4, strides=2, padding="same"
            ),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2DTranspose(
                1, kernel_size=7, strides=1, padding="same", activation="sigmoid"
            ),
        ],
        name="generator",
    )

    # - - - - - - - Discriminator - - - - - - -
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((32, 32, dis_channels)),
            keras.layers.Conv2D(batch_size, kernel_size=3, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
            keras.layers.LeakyReLU(negative_slope=0.2),
            keras.layers.GlobalMaxPool2D(),
            keras.layers.Dense(1),
        ],
        name="discriminator",
    )

    return generator, discriminator


def build_conditional_gan(generator, discriminator):
    """
    Builds the conditional GAN (cGAN) model.

    Args:
        generator (keras.Model): Generator model.
        discriminator (keras.Model): Discriminator model.

    Returns:
        conditionalGAN: Compiled cGAN model.
    """
    cond_gan = conditionalGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=latent_dim,
        image_size=image_size,
        num_classes=num_classes,
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    return cond_gan


def train_model(dataset, cond_gan):
    """
    Trains the conditional GAN (cGAN) model.

    Args:
        dataset (tf.data.Dataset): Dataset containing the training images and labels.
        cond_gan (conditionalGAN): Compiled cGAN model.
    """
    cond_gan.fit(dataset, epochs=2)


def save_model_weights(cond_gan, filename):
    """
    Saves the weights of the conditional GAN (cGAN) model.

    Args:
        cond_gan (conditionalGAN): Compiled cGAN model.
        filename (str): Filepath to save the model weights.
    """
    if os.path.exists(filename):
        os.remove(filename)
    cond_gan.save_weights(filename)


def load_model_with_weights(filename):
    """
    Loads the conditional GAN (cGAN) model with saved weights.

    Args:
        filename (str): Filepath to the saved model weights.

    Returns:
        conditionalGAN: cGAN model with loaded weights.
    """
    generator, discriminator = build_models()

    new_cond_gan = build_conditional_gan(generator, discriminator)

    new_cond_gan.load_weights(filename)
    return new_cond_gan
