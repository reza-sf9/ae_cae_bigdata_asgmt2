# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:37:25 2021

@author: rsaadatifard
"""

## import dependencies

import os

import sklearn.model_selection
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, AvgPool2D, BatchNormalization, Flatten, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model

## Seeding
np.random.seed(10)
tf.random.set_seed(10)

## loading dataset
(x_train, y_train), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
x_train, x_te = x_train / 255.0, x_te / 255.0


x_tr, x_val, y_tr, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

## convolutional AE

# Encoder
inputs = Input(shape=(28, 28, 1), name='inputs')
x= inputs

# first conv layer
filt_num_1 = 32
filt_size_1 = (3, 3)
x = Conv2D(filt_num_1, filt_size_1)(x)
x = BatchNormalization()(x)
alpha_leaky = 0.1
x = LeakyReLU(alpha=alpha_leaky)(x)
max_pool_size = (2, 2)
x = MaxPool2D(max_pool_size)(x)
vec_size_max_pool = (x.shape[1], x.shape[2], x.shape[3])

# flatten
x = Flatten()(x)
l_flatten = x.shape[1]

# getting latent variable
l_latent = 64
x = Dense(l_latent, name= 'latent')(x)

# make flatten layer again
x = Dense(l_flatten)(x)
x = LeakyReLU(alpha=alpha_leaky)(x)

# Decoder
# deflatten and reshaping to filter conv type
x = Reshape(vec_size_max_pool)(x)

x = Conv2DTranspose(filt_num_1, (2, 2), strides=2, )(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=alpha_leaky)(x)

x = Conv2DTranspose(1, filt_size_1)(x)
x = BatchNormalization()(x)
x = Activation("sigmoid", name="outputs")(x)

outputs = x

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
autoencoder.summary()


## running the model
epoch_num = 50
batch_size_num = 256
x_tr = np.reshape(x_tr, (x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], 1))
y_tr = np.reshape(y_tr, (y_tr.shape[0],  1))


# x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
# x_te = np.reshape(x_te, (x_te.shape[0], x_te.shape[1], x_te.shape[2], 1))
autoencoder.fit(x_tr, x_tr, epochs=epoch_num, batch_size=batch_size_num, shuffle=False) # , validation_data=(x_val, y_val)

test_pred_y = autoencoder.predict(x_te)

n = 10  ## how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ## display original
    ax = plt.subplot(2, n, i + 1)
    ax.set_title("Original Image")
    plt.imshow(x_te[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ## display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    ax.set_title("Predicted Image")
    plt.imshow(test_pred_y[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("results/convolutonal_autoencoder.png")

## convolutaional VAE

## simple AE

## VAE



k=1