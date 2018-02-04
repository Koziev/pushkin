# -*- coding: utf-8 -*-

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import gensim
import codecs
import numpy as np
from numpy import linalg
import pickle
import six

import sys

import numpy as np


class GAN():
    def __init__(self, train_data, i2c):

        self.noise_dims = 100

        self.train_data = train_data
        self.i2c = i2c

        self.img_rows = self.train_data.shape[1]
        self.img_cols = self.train_data.shape[2]
        self.img_shape = (self.img_rows, self.img_cols)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.noise_dims,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (self.noise_dims,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))  #256
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))  # 512
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1000))  # 1000
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='sigmoid'))  # tanh
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        X_train = self.train_data
        half_batch = batch_size // 2

        d_acc_avg = 50.0
        avg_moment = 0.99

        for epoch in range(1,epochs+1):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, self.noise_dims))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            if False:
                noise = np.random.normal(0, 1, (batch_size, self.noise_dims))

                # The generator wants the discriminator to label the generated samples
                # as valid (ones)
                valid_y = np.array([1] * batch_size)

                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid_y)
            else:
                # The generator wants the discriminator to label the generated samples
                # as valid (ones)
                valid_y = np.array([1] * half_batch)

                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            #if epoch % 100 == 0:
            #    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            d_acc = 100*d_loss[1]
            d_acc_avg = avg_moment*d_acc_avg + d_acc*(1.0-avg_moment)
            if epoch % 1000 == 0:
                # Plot the progress
                #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], d_acc, g_loss))
                print('epoch={} D acc={}'.format(epoch, d_acc_avg))


            # If at save interval => save generated image samples
            if epoch % 5000 == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        nb_words = 10
        noise = np.random.normal(0, 1, (nb_words, self.noise_dims))
        gen_imgs = self.generator.predict(noise)

        word_len = gen_imgs.shape[1]

        print('')
        for iword in range(nb_words):
            word_chars = []
            for ichar in range(word_len):
                v = gen_imgs[iword, ichar]
                c = self.i2c[ np.argmax(v) ]
                word_chars.append(c)

            word = u''.join(word_chars)
            print(u'[{}]'.format(word))


if __name__ == '__main__':
    train_data = np.load('./data/words_4gan.npz')
    train_data = train_data['arr_0']

    with open('./data/c2i_4gan.pkl', 'r') as f:
        c2i = pickle.load(f)

    i2c = dict([(i, c) for (c, i) in six.iteritems(c2i)])

    print('nb_chars={0}'.format(train_data.shape[2]))
    print('max_word_len={}'.format(train_data.shape[1]))
    print('nb_words={}'.format(train_data.shape[0]))

    gan = GAN(train_data, i2c)
    gan.train(epochs=1000000, batch_size=32, save_interval=200)


