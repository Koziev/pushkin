# -*- coding: utf-8 -*-

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import pickle
import six
import itertools
import numpy as np


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




# ------------------------------------------------------------

class GAN():
    def __init__(self, train_data, i2c):

        self.noise_dims = 100

        self.train_data = train_data
        self.i2c = i2c

        self.img_rows = self.train_data.shape[1]
        self.img_cols = self.train_data.shape[2]
        self.img_shape = (self.img_rows, self.img_cols)

        optimizer = Adam(0.0002, 0.5)

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

        noise_shape = (self.noise_dims,)

        noise = Input(shape=noise_shape)

        layers = []
        layers.append(Dense(256, input_shape=noise_shape))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(BatchNormalization(momentum=0.8))
        layers.append(Dense(512))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(BatchNormalization(momentum=0.8))
        layers.append(Dense(1000))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(BatchNormalization(momentum=0.8))
        layers.append(Dense(np.prod(self.img_shape), activation='sigmoid'))
        layers.append(Reshape(self.img_shape))

        return noise, layers

    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols)

        d_input = Input(shape=img_shape)

        layers = []
        layers.append(Flatten())
        layers.append(Dense(256))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Dense(128))
        layers.append(LeakyReLU(alpha=0.2))
        layers.append(Dense(1, activation='sigmoid'))

        return d_input, layers

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

            noise = np.random.normal(0, 1, (batch_size, self.noise_dims))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            #valid_y = np.array([1] * batch_size)

            # Train the generator
            #g_loss = self.combined.train_on_batch(noise, valid_y)

            # так как у нас есть gradient reversal layer между выходом генератора и входом
            # дискриминатора, то в качестве целевого значения ставим 0 - дискриминатор должен
            # опознать шум как фейковые изображения.
            g_loss = self.combined.train_on_batch(noise, np.array([0.0] * batch_size))

            # Plot the progress
            #if epoch % 100 == 0:
            #    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            d_acc = 100*d_loss[1]
            d_acc_avg = avg_moment*d_acc_avg + d_acc*(1.0-avg_moment)
            if epoch % 100 == 0:
                # Plot the progress
                #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], d_acc, g_loss))
                print('epoch={} D acc={}'.format(epoch, d_acc_avg))

            # If at save interval => save generated image samples
            if epoch % 1000 == 0:
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


