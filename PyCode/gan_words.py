# -*- coding: utf-8 -*-
"""
Простейшая модель Generative Adversarial Network для текстовых данных.
Модель учится генерировать короткие русскоязычные предложения.

За основу взят код из https://github.com/eriklindernoren/Keras-GAN/gan/gan.py
"""

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import MaxPooling1D

import codecs
import numpy as np
from numpy import linalg
import pickle

import matplotlib.pyplot as plt

import sys

import numpy as np


class GAN():
    def __init__(self, vtexts, word2vec, w2v_dims):

        self.noise_dims = 64

        self.word2id = dict([(w, i) for (i, w) in enumerate(word2vec.keys())])
        self.word_vecs = np.zeros( (len(word2vec), w2v_dims) )
        for word, i in self.word2id.iteritems():
            self.word_vecs[i, :] = word2vec[word]

        self.vtexts = vtexts.copy()

        # проверим нормализацию векторов
        vmin = np.amin(self.vtexts)
        vmax = np.amax(self.vtexts)
        print('vmin={} vmax={}'.format(vmin, vmax))

        self.img_rows = self.vtexts.shape[1] # 28
        self.img_cols = self.vtexts.shape[2] # 28
        #self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols) #, self.channels)

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

        model.add(Dense(self.noise_dims, input_shape=noise_shape))  #256
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(int(self.noise_dims*1.5)))  # 512
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(int(self.noise_dims*2)))  # 512
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(150))  # 1000
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols)  #, self.channels)

        model = Sequential()

        if False:
            model.add(Flatten(input_shape=img_shape))
            model.add(Dense(128))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(64))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(1, activation='sigmoid'))
        else:
            model.add(Conv1D(input_shape=img_shape,
                             filters=64,
                             kernel_size=2,
                             padding='valid',
                             activation='relu',
                             strides=1))
            model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))

            model.add(Flatten())

            model.add(Dense(64))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.3))

            model.add(Dense(32))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.2))

            model.add(Dense(16))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.1))

            model.add(Dense(1, activation='sigmoid'))


        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        X_train = self.vtexts
        half_batch = int(batch_size / 2)

        d_acc_avg = 50.0
        d_acc_fake_avg = 50.0
        d_acc_real_avg = 50.0
        avg_moment = 0.995

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_dims))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            #noise = np.random.normal(0, 1, (batch_size, self.noise_dims))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * half_batch)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            d_acc = 100*d_loss[1]
            d_acc_avg = avg_moment*d_acc_avg + d_acc*(1.0-avg_moment)

            d_acc_real = 100*d_loss_real[1]
            d_acc_real_avg = avg_moment*d_acc_real_avg + d_acc_real*(1.0-avg_moment)

            d_acc_fake = 100*d_loss_fake[1]
            d_acc_fake_avg = avg_moment*d_acc_fake_avg + d_acc_fake*(1.0-avg_moment)

            if epoch % 1000 == 0:
                # Plot the progress
                #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], d_acc, g_loss))
                print('epoch={} D acc={} acc_fake={} acc_real={}'.format(epoch, d_acc_avg, d_acc_fake_avg, d_acc_real_avg))

            # Plot the progress
            #if epoch % 100 == 0:
            #    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch>0 and epoch % 50000 == 0:
                self.save_imgs(epoch)

    def match_word(self, probe_v):
        min_dist = 1e38
        best_word = u''
        for word, iword in self.word2id.iteritems():
            v = self.word_vecs[iword]
            d = np.linalg.norm( v - probe_v )
            if d < min_dist:
                min_dist = d
                best_word = word

        return best_word

    def save_imgs(self, epoch):
        nb_phrases = 10
        noise = np.random.normal(0, 1, (nb_phrases, self.noise_dims))
        gen_imgs = self.generator.predict(noise)

        phrase_len = gen_imgs.shape[1]

        print('')
        with codecs.open('./texts/phrases_{}.txt'.format(epoch), 'w', 'utf-8') as wrt:
            for iphrase in range(nb_phrases):
                phrase_words = []
                for iword in range(phrase_len):
                    v = gen_imgs[iphrase, iword]
                    word = self.match_word(v)
                    phrase_words.append(word)

                phrase = u' '.join(phrase_words)
                print(u'{}'.format(phrase))
                wrt.write(u'{}\n'.format(phrase))


if __name__ == '__main__':
    # Файлы датасета уже подготовлены с помощью prepare_vae_datasets.py
    vtexts = np.load('../data/vtexts.npz')
    vtexts = vtexts['arr_0']

    with open('../data/word2vec.pkl', 'r') as f:
        word2vec = pickle.load(f)

    w2v_dims = vtexts.shape[2]
    print('w2v_dims={0}'.format(w2v_dims))

    nb_words = vtexts.shape[1]
    print('nb_words={}'.format(nb_words))

    gan = GAN(vtexts, word2vec, w2v_dims)
    gan.train(epochs=1000000, batch_size=1024, save_interval=200)
