import tensorflow as tf
import math
import random
import tensorflow_datasets as tfds
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time



resize_and_rescale = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.Rescaling(1./127.5, offset=-1)
])
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.GaussianNoise(0.1)
])


def load_data(dir_name):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dir_name,
        labels = None,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(64, 64),
        shuffle=True
    )
    return dataset

initializer = tf.keras.initializers.HeNormal()


def make_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((1, 1, 100),  input_shape = (100, )),

        tf.keras.layers.Conv2DTranspose(1024, (4, 4), strides = (1,1), use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(512, (5, 5), strides = (2,2), padding = 'same', use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(256, (5, 5), strides = (2,2), padding = 'same', use_bias = False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides = (2,2), padding = 'same', use_bias = False),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2DTranspose(3, (5, 5), strides = (2,2), padding = 'same', use_bias = False, activation = 'tanh'),
    ])
    return model

def make_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same', input_shape = (64, 64, 3,)),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2D(256, (5, 5), strides = (2, 2), padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(512, (5, 5), strides = (2, 2), padding = 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(1024, (5, 5), strides = (2, 2), padding = 'same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model

seed = tf.random.normal((1,100))
generator = make_generator()
hj = generator(seed)
disc = make_discriminator()
out = disc(hj)
disc.summary()
generator.summary()
loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def disc_loss(real_output, fake_output):
    loss_true = loss(tf.ones_like(real_output), real_output)
    loss_false = loss(tf.zeros_like(fake_output), fake_output)
    total_loss = loss_true + loss_false
    return total_loss

def gen_loss(fake_output):
    total_loss = loss(tf.ones_like(fake_output), fake_output)
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

@tf.function
def train_step(images):
    noise = tf.random.normal((BATCH_SIZE, noise_dim))
    images = resize_and_rescale(images)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training = True)

        real_output = disc(images, training = True)
        fake_output = disc(generated_images, training = True)

        g_loss = gen_loss(fake_output)
        d_loss = disc_loss(real_output, fake_output)

    grad_of_gen = gen_tape.gradient(g_loss, generator.trainable_variables)
    grad_of_disc = disc_tape.gradient(d_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients( zip(grad_of_gen, generator.trainable_variables) )
    discriminator_optimizer.apply_gradients( zip(grad_of_disc, disc.trainable_variables) )


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)
        if epoch%1 == 0:
            print("epoch " + str(epoch))
            noise = tf.random.normal((1,100))
            img = generator(noise).numpy()
            img = (img + 1.)*0.5
            plt.imshow(img[0])
            plt.savefig('/home/maciej/Desktop/Czarna dziura/PROJEKTY MACIEJA/AI/car_creator/training_preview/epoch'+str(epoch))
            generator.save_weights('/home/maciej/Desktop/Czarna dziura/PROJEKTY MACIEJA/AI/car_creator/gan_data/generator')
            disc.save_weights('/home/maciej/Desktop/Czarna dziura/PROJEKTY MACIEJA/AI/car_creator/gan_data/discriminator')

#generator.load_weights('/home/maciej/Desktop/Czarna dziura/PROJEKTY MACIEJA/AI/car_creator/gan_data/generator')
#disc.load_weights('/home/maciej/Desktop/Czarna dziura/PROJEKTY MACIEJA/AI/car_creator/gan_data/discriminator')
ds = load_data('/home/maciej/Desktop/Czarna dziura/PROJEKTY MACIEJA/AI/car_creator/car')
train(ds, 5000)
