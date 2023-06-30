import tensorflow as tf
import math
import random
import tensorflow_datasets as tfds
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
import cv2
import random

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


generator = make_generator()
disc = make_discriminator()

generator.load_weights('gan_data/generator')
disc.load_weights('gan_data/discriminator')

photo = []

def generate():
    global photo
    noise = []
    for x in w:
        noise.append(x.get()/100.0)
    noise = np.array(noise)
    noise = np.reshape(noise, (1,100))
    img = generator(noise).numpy()
    img = img[0]
    img = (img + 1.)*0.5*255.0
    print(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LANCZOS4)
    cv2.imwrite('savedimage.png', img)
    photo = tk.PhotoImage(file='savedimage.png')
    canvas.itemconfig(cont,image=photo)

ws = tk.Tk()
ws.title('Generator')
ws.geometry('512x512')

canvas = tk.Canvas(
    ws,
    width = 256,
    height = 256
    )
canvas.grid()
img = tk.PhotoImage(file='savedimage.png')
cont = canvas.create_image(
    10,
    10,
    anchor=tk.NW,
    image=img
    )

w = []
r = 0
for g in range(0,100):
    w.append(tk.Scale(ws, from_=-100, to=100, orient=tk.HORIZONTAL))
    w[-1].set(random.gauss()*100)
    w[-1].grid(row = r, column = 1+(g%8))
    if g%8 == 7:
        r = r + 1


tk.Button(ws, text='Show', command = generate).grid()
ws.mainloop()
