# -*- coding: utf-8 -*-
"""
Generative Adversarial Network для генерации латентных векторов из автоэнкодерной модели.
За основу взят код из https://github.com/eriklindernoren/Keras-GAN/gan/gan.py
"""

from __future__ import print_function, division

import future.utils
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.layers import MaxPooling1D
from keras.models import model_from_json

import codecs
import numpy as np
from numpy import linalg
import pickle
import json
import scipy.spatial.distance
import itertools

import sys
import os

import numpy as np

from word_decoder import W2V_Decoder

tmp_folder = '../tmp'


# ------------------------------------------------------------


# https://github.com/keras-team/keras/issues/3119
# Gradient-reversal layer для Theano

import theano
import keras.layers


class ReverseGradient(theano.Op):
    """ theano operation to reverse the gradients
    Introduced in http://arxiv.org/pdf/1409.7495.pdf
    """

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes


class GradientReversalLayer(keras.layers.Layer):
    """ Reverse a gradient
    <feedforward> return input x
    <backward> return -lambda * delta
    """
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return self.gr_op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                         "lambda": self.hp_lambda}
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# ------------------------------------------------------------
# https://github.com/michetonu/gradient_reversal_keras_tf
# Keras implementation of a gradient inversion layer for the Tensorflow backend


# ---------------------------------------------------------


class AE_Decoder:
    def __init__(self):
        pass

    def load(self):
        with open(os.path.join('../tmp', 'lstm_ae.config'), 'r') as f:
            model_config = json.load(f)

            self.max_seq_len = model_config['max_seq_len']
            decoder_arch_filepath = model_config['decoder_arch_filepath']
            decoder_weights_path = model_config['decoder_weights_path']
            self.word_dims = model_config['word_dims']

        with open(decoder_arch_filepath, 'r') as f:
            self.decoder_model = model_from_json(f.read())

        self.decoder_model.load_weights(decoder_weights_path)
        self.w2v_decoder = W2V_Decoder('../data/word2vec.pkl')

    def decode(self, latents):
        # Пропускаем подготовленный вектор через декодер.
        y_probe = self.decoder_model.predict(latents)

        # декодируем результаты работы модели
        result_phrases = self.w2v_decoder.decode_output(y_probe)
        for phrase in result_phrases:
            print(u'{}'.format(phrase))


# ------------------------------------------------------


class GAN():
    def __init__(self, latents):

        self.ae_decoder = AE_Decoder()
        self.ae_decoder.load()

        self.latent_dim = latents.shape[1]

        self.noise_dims = self.latent_dim//2

        self.latents = latents

        # проверим нормализацию векторов
        self.latent_min = np.amin(self.latents)
        self.latent_max = np.amax(self.latents)

        optimizer = Adam(lr=0.0002, beta_1=0.5)
        #optimizer = RMSprop(lr=0.00005)

        # Build and compile the generator
        # Вход - случайный вектор
        # Выход - сгенерированное изображение
        g_input, g_layers = self.build_generator()
        self.generator = self.build_model(g_input, g_layers)
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)


        # Build and compile the discriminator
        # Вход - сгенерированные или реальные изображения
        # Выход - вероятность того, что изображение реальное.
        d_input, d_layers = self.build_discriminator()
        self.discriminator = self.build_model(d_input, d_layers)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Полная модель, которая для заданного случайного вектора должна выдать
        # на выходе одни 0, так как на вход дискриминатора будет подан только выход генератора.
        gradient_reversal = GradientReversalLayer(hp_lambda=1.0)

        # For the combined model we will only train the generator
        for l in d_layers:
            l.trainable = False
        #self.discriminator.trainable = False

        self.combined = self.build_model(g_input, itertools.chain(g_layers, [gradient_reversal], d_layers))
        self.combined.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])


    def build_model(self, input, layers):
        """
        Стек слоев связываем в граф. Возвращается экземпляр класса Model
        :param input:
        :param layers:
        :return:
        """
        output = input
        for layer in layers:
            output = layer(output)

        return Model(inputs=input, outputs=output)

    def build_generator(self):

        noise = Input(shape=(self.noise_dims,))

        output_dim = self.latent_dim

        layers = []

        a = LeakyReLU(alpha=0.2)
        # a = Activation('relu')

        layers.append(Dense(self.noise_dims))
        layers.append(a)
        layers.append(BatchNormalization(momentum=0.8))

        layers.append(Dense(output_dim))
        layers.append(a)
        layers.append(BatchNormalization(momentum=0.8))

        layers.append(Dense(output_dim))
        layers.append(a)
        layers.append(BatchNormalization(momentum=0.8))

        layers.append(Dense(output_dim, activation='tanh'))

        return noise, layers

    def build_discriminator(self):
        d_input = Input(shape=(self.latent_dim,))

        layers = []
        layers.append(Dense(units=self.latent_dim, input_dim=self.latent_dim))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Dense(self.latent_dim//2))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Dense(1, activation='sigmoid'))

        return d_input, layers

    def generate_noise(self, nb_samples):
        noise = np.random.normal(0, 1, (nb_samples, self.noise_dims))
        #noise = np.random.uniform(low=-2.0, high=2.0, size=(nb_samples, self.noise_dims))
        return noise

    def train(self, epochs, batch_size, save_interval):
        X_train = self.latents
        half_batch = batch_size // 2

        d_acc_avg = 50.0
        d_acc_fake_avg = 50.0
        d_acc_real_avg = 50.0
        avg_moment = 0.995

        D_train_epochs = 0

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of latents vectors
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = self.generate_noise(half_batch)
            # Generate a half batch of new latent vectors
            gen_imgs = self.generator.predict(noise)

            # ОТЛАДКА: вместо шума - тоже реальные векторы
            #idx = np.random.randint(0, X_train.shape[0], half_batch)
            #gen_imgs = X_train[idx]


            # Train the discriminator
            acc_index = self.discriminator.metrics_names.index('acc')
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # noise = self.generate_noise(batch_size)

            if epoch > D_train_epochs:
                # Train the generator
                # так как у нас есть gradient reversal layer между выходом генератора и входом
                # дискриминатора, то в качестве целевого значения ставим 0 - дискриминатор должен
                # опознать шум как фейковые изображения.
                g_loss = self.combined.train_on_batch(noise, np.zeros((half_batch, 1)))


            d_acc = 100*d_loss[acc_index]
            d_acc_avg = avg_moment*d_acc_avg + d_acc*(1.0-avg_moment)

            d_acc_real = 100*d_loss_real[acc_index]
            d_acc_real_avg = avg_moment*d_acc_real_avg + d_acc_real*(1.0-avg_moment)

            d_acc_fake = 100*d_loss_fake[acc_index]
            d_acc_fake_avg = avg_moment*d_acc_fake_avg + d_acc_fake*(1.0-avg_moment)

            if epoch % 1000 == 0:
                # Plot the progress
                #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], d_acc, g_loss))
                print('epoch={} D acc={} acc_fake={} acc_real={}'.format(epoch, d_acc_avg, d_acc_fake_avg, d_acc_real_avg))

            # Plot the progress
            #if epoch % 100 == 0:
            #    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch>0 and epoch % 10000 == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        nb_phrases = 10

        # входной шум
        noise = self.generate_noise(nb_phrases)

        # созданные на базе шума фейковые латентные векторы
        gen_latents = self.generator.predict(noise)

        gen_min = np.amin(gen_latents)
        gen_max = np.amax(gen_latents)
        print('real min={} max={}'.format(self.latent_min, self.latent_max))
        print('fake min={} max={}'.format(gen_min, gen_max))

        # пропускаем сгенерированные векторы через декодер, получаем цепочку
        # векторов слов, затем ищем соответствия векторов и слов.
        self.ae_decoder.decode(gen_latents)



if __name__ == '__main__':
    # Файлы датасета уже подготовлены с помощью lstm_ae.py --estimate 1
    latents = np.load('../tmp/latents.npz')
    latents = latents['arr_0']

    #with open('../data/word2vec.pkl', 'r') as f:
    #    word2vec = pickle.load(f)

    #w2v_dims = vtexts.shape[2]
    #print('w2v_dims={0}'.format(w2v_dims))

    print('latents.shape={}'.format(latents.shape))

    gan = GAN(latents)
    gan.train(epochs=1000000, batch_size=2048, save_interval=200)
