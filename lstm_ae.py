# -*- coding: utf-8 -*-
"""
Для baseline оценки - автоэнкодер и вариационный автоэнкодер предложений
на базе LSTM+Conv (точная конфигурация нейросети настраивается).
Тренируется на подготовленном заранее датасете - см. скрипты prepare_phrases.py
и prepare_vae_dataset.py
"""

from __future__ import print_function, division

import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import recurrent
from keras.models import Model
from keras.models import model_from_json
from keras.layers.core import RepeatVector, Dense
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras import objectives
from keras.layers.core import Lambda

import tqdm
import sklearn.model_selection
import pickle
import numpy as np
import sys
import json
import os

from future.utils import iteritems


tmp_folder = '../tmp'
model_folder = '../tmp'

BATCH_SIZE = 32

# длина вектора на выходе кодирующей части автоэнкодера, фактически это
# длина вектора, представляющего предложение.
latent_dim = 64

NB_EPOCHS = 100  # макс. кол-во эпох обучения

# Конфигурация нейросетки:
# arch - общая архитектура: 'ae' для простого сжимающего автоэнкодера, 'vae' для
#        вариационного автоэнкодера
# encoder - структура кодирующей части
# decoder - структура декодера
# NET_CONFIG={'arch': 'ae', 'encoder': 'lstm(cnn)', 'decoder': 'lstm,dense'}
NET_CONFIG={'arch': 'vae', 'encoder': 'lstm(cnn)', 'decoder': 'lstm,lstm'}



def create_ae(net_config, max_seq_len, word_dims, latent_dim):
    """
    Создается классический автоэнкодер с архитектурой seq2seq.
    Возвращается 3 модели - полный автоэнкодер, кодер и декодер.
    """
    encoder_input = Input(shape=(max_seq_len, word_dims,), dtype='float32', name='input_words')

    if net_config['encoder'] == 'lstm(cnn)':
        convs = []
        encoder_size = 0

        nb_filters = 64
        rnn_size = nb_filters

        for kernel_size in range(1, 4):
            # сначала идут сверточные слои, образующие детекторы словосочетаний
            # и синтаксических конструкций
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='relu',
                          strides=1)

            lstm = recurrent.LSTM(rnn_size, return_sequences=False, name='encoder_lstm_{}'.format(kernel_size))

            conv_layer = conv(encoder_input)
            conv_layer = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer)
            conv_layer = lstm(conv_layer)
            convs.append(conv_layer)

            encoder_size += rnn_size

        encoder_merged = keras.layers.concatenate(inputs=convs)
        encoder_final = Dense(units=int(latent_dim), activation='sigmoid')(encoder_merged)
    elif net_config['encoder'] == 'lstm':
        encoder_final = recurrent.LSTM(latent_dim, return_sequences=False)(encoder_input)
        encoder_final = Dense(units=int(latent_dim), activation='sigmoid')(encoder_final)

    # декодер
    decoder = RepeatVector(max_seq_len)(encoder_final)

    if net_config['decoder'] == 'lstm,dense':
        # первый вариант окончания декодера - афинный слой после LSTM
        lstm_layer = recurrent.LSTM(latent_dim, return_sequences=True, name='decoder_lstm1')
        decoder = lstm_layer(decoder)
        dense_layer = Dense(w2v_dims, activation='tanh', name='decoder_dense')
        decoder = TimeDistributed(dense_layer, name='output')(decoder)
    elif net_config['decoder'] == 'lstm,lstm':
        # Второй вариант окончания декодера - стек из двух LSTM
        lstm_layer1 = recurrent.LSTM(latent_dim, return_sequences=True, name='decoder_lstm1')
        lstm_layer2 = recurrent.LSTM(w2v_dims, return_sequences=True, name='decoder_lstm2')
        decoder = lstm_layer1(decoder)
        decoder = lstm_layer2(decoder)
    else:
        raise NotImplemented()

    # собираем три модели
    ae_model = Model(inputs=encoder_input, outputs=decoder)
    ae_model.compile(loss='mse', optimizer='nadam')

    encoder_model = Model(inputs=encoder_input, outputs=encoder_final)
    encoder_model.compile(loss='mse', optimizer='nadam')

    decoder_input = Input(shape=(latent_dim,), dtype='float32', name='input_latent')
    decoder = RepeatVector(max_seq_len)(decoder_input)
    if net_config['decoder'] == 'lstm,dense':
        # первый вариант окончания декодера - афинный слой после LSTM
        decoder = lstm_layer(decoder)
        decoder = TimeDistributed(dense_layer, name='output')(decoder)
    elif net_config['decoder'] == 'lstm,lstm':
        # Второй вариант окончания декодера - стек из двух LSTM
        decoder = lstm_layer1(decoder)
        decoder = lstm_layer2(decoder)

    decoder_model = Model(inputs=decoder_input, outputs=decoder)
    decoder_model.compile(loss='mse', optimizer='nadam')

    return ae_model, encoder_model, decoder_model


def create_vae(
               net_config,
               timesteps,
               input_dim,
               latent_dim,
               batch_size,
               epsilon_std=1.):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """

    intermediate_dim = 64  # output shape of LSTM.

    encoder_input = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    #h = recurrent.LSTM(intermediate_dim)(x)
    if net_config['encoder'] == 'lstm(cnn)':
        convs = []
        encoder_size = 0

        nb_filters = 64
        rnn_size = nb_filters

        for kernel_size in range(1, 4):
            # сначала идут сверточные слои, образующие детекторы словосочетаний
            # и синтаксических конструкций
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='relu',
                          strides=1)

            lstm = recurrent.LSTM(rnn_size, return_sequences=False, name='encoder_lstm_{}'.format(kernel_size))

            conv_layer = conv(encoder_input)
            conv_layer = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer)
            conv_layer = lstm(conv_layer)
            convs.append(conv_layer)

            encoder_size += rnn_size

        encoder_merged = keras.layers.concatenate(inputs=convs)
        encoder_final = Dense(units=int(latent_dim), activation='sigmoid')(encoder_merged)
    elif net_config['encoder'] == 'lstm':
        encoder_final = recurrent.LSTM(latent_dim, return_sequences=False)(encoder_input)
        #encoder_final = Dense(units=int(latent_dim), activation='sigmoid')(encoder_final)

    h = encoder_final

    # VAE Z layer
    z_mean = Dense(units=latent_dim, name='z_mean')(h)
    z_log_sigma = Dense(units=latent_dim, name='z_log_sigma')(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,), name='vae_lambda')([z_mean, z_log_sigma])

    # decoder layers
    decoder = RepeatVector(timesteps)(z)

    if net_config['decoder'] == 'lstm,lstm':
        lstm1 = recurrent.LSTM(intermediate_dim, return_sequences=True, name='decoder_lstm1')
        lstm2 = recurrent.LSTM(input_dim, return_sequences=True, name='decoder_lstm2')
        decoder = lstm1(decoder)
        decoder = lstm2(decoder)
    elif net_config['decoder'] == 'lstm,dense':
        lstm = recurrent.LSTM(intermediate_dim, return_sequences=True, name='decoder_lstm')
        dense_layer = Dense(input_dim, activation='tanh', name='decoder_dense')
        decoder = lstm(decoder)
        decoder = TimeDistributed(dense_layer, name='output')(decoder)

    # end-to-end autoencoder
    vae = Model(encoder_input, decoder)

    # encoder, from inputs to latent space
    encoder = Model(encoder_input, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    decoder = RepeatVector(timesteps)(decoder_input)
    if net_config['decoder'] == 'lstm,lstm':
        decoder = lstm1(decoder)
        decoder = lstm2(decoder)
    elif net_config['decoder'] == 'lstm,dense':
        decoder = lstm(decoder)
        decoder = TimeDistributed(dense_layer, name='output')(decoder)

    generator = Model(decoder_input, decoder)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator


def decode_output(y, v2w):
    """
    Декодируем выходной тензор автоэнкодера, получаем читабельные
    предложения в том же порядке, как входные.
    """
    decoded_phrases = []

    for iphrase in range(y.shape[0]):
        phrase_vectors = y[iphrase]
        phrase_words = []
        for iword in range(y.shape[1]):
            word_vector = phrase_vectors[iword]
            l2 = np.linalg.norm(word_vector)
            if l2<0.1:
                break

            min_dist = 1e38
            best_word = u''
            for v, w in v2w:
                d = np.linalg.norm(v - word_vector)
                if d < min_dist:
                    min_dist = d
                    best_word = w

            phrase_words.append(best_word)

        decoded_phrases.append(u' '.join(phrase_words))

    return decoded_phrases


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):

    def __init__(self, ae_model, X_data, v2w, batch_size):
        self.epoch = 0
        self.X_data = X_data
        self.model = ae_model
        self.v2w = v2w
        self.batch_size = batch_size

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1

        print('\n')
        # Выберем подмножество паттернов, которые мы прогоним через модель
        idx = np.random.permutation( range(self.batch_size) )
        #idx = list(range(self.batch_size))
        X_test = self.X_data[idx]
        y_test = self.model.predict(X_test, verbose=0)

        nb_tested = 0
        nb_errors = 0
        for iphrase in range(min(10,X_test.shape[0])):
            input_phrase = decode_output(X_test[iphrase:iphrase+1], self.v2w)[0]
            output_phrase = decode_output(y_test[iphrase:iphrase+1], self.v2w)[0]
            print(colors.ok + '☑ ' + colors.close if input_phrase == output_phrase else colors.fail + '☒ ' + colors.close,
                  end='')

            print(u'{} ==> {}'.format(input_phrase, output_phrase))


# -----------------------------------------------------------

# Словарь с парами слово-вектор для печати читабельных результатов
with open('../data/word2vec.pkl', 'r') as f:
    word2vec = pickle.load(f)

v2w = [(v, w) for w, v in iteritems(word2vec)]

# -----------------------------------------------------------

do_train = False
do_vizualize = False
while True:
    print('t - train the model\ng - generate random texts with trained model')
    run_mode = raw_input('? ').decode(sys.stdout.encoding).strip().lower()
    if run_mode == 't':
        do_train = True
        break
    elif run_mode == 'g':
        do_vizualize = True
        break
    else:
        print('Invalid selection, please repeat')


if do_train:
    # Загружаем подготовленный датасет с векторизованными фразами.
    vtexts = np.load('../data/vtexts.npz')
    vtexts = vtexts['arr_0']

    w2v_dims = vtexts.shape[2]
    print('w2v_dims={0}'.format(w2v_dims))

    max_seq_len = vtexts.shape[1]
    print('max_seq_len={}'.format(max_seq_len))

    # ---------------------------------------------------------------
    # Создание нейросетки

    if NET_CONFIG['arch'] == 'ae':
        ae_model, encoder_model, decoder_model = create_ae(NET_CONFIG, max_seq_len, w2v_dims, latent_dim)
    elif NET_CONFIG['arch'] == 'vae':
        ae_model, encoder_model, decoder_model = create_vae(NET_CONFIG, max_seq_len, w2v_dims, latent_dim, BATCH_SIZE )
    else:
        raise NotImplemented()

    print('ae_model:')
    ae_model.summary()

    print('\nencoder_model:')
    encoder_model.summary()

    print('\ndecoder_model:')
    decoder_model.summary()

    weights_path = os.path.join(model_folder, 'lstm_ae_coder.weights')
    arch_filepath = os.path.join(model_folder, 'lstm_ae_coder.arch')

    # сохраним конфиг модели в json файлике.
    model_config = {
        'max_seq_len': max_seq_len,
        'latent_dim': latent_dim,
        'arch_filepath': arch_filepath,
        'weights_path': weights_path,
        'word_dims': w2v_dims
    }

    with open(os.path.join(tmp_folder, 'lstm_ae.config'), 'w') as f:
        json.dump(model_config, f)

    monitor_metric = 'val_loss'
    model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    viz = VisualizeCallback(ae_model, vtexts, v2w, BATCH_SIZE)

    callbacks = [viz, model_checkpoint, early_stopping]

    val_size = (int(0.2*vtexts.shape[0])//BATCH_SIZE)*BATCH_SIZE
    train_data, val_data = sklearn.model_selection.train_test_split( vtexts, test_size=val_size, random_state=123456)

    # для VAE нужно, чтобы данные были точно выровнены на BATCH_SIZE.
    if (train_data.shape[0]%BATCH_SIZE) != 0:
        train_data = train_data[ : (train_data.shape[0]//BATCH_SIZE)*BATCH_SIZE ]


    if True:
        hist = ae_model.fit(x=train_data,
                            y=train_data,
                            validation_data=(val_data, val_data),
                            batch_size=BATCH_SIZE,
                            epochs=NB_EPOCHS,
                            verbose=1,
                            callbacks=callbacks,
                           )
    else:
        nb_train_batches = train_data.shape[0]//BATCH_SIZE
        nb_val_batches = val_data.shape[0]//BATCH_SIZE

        nb_no_impovements = 0
        best_loss = 1e38

        for epoch in range(10):
            print('Epoch {}'.format(epoch))

            # случайная перестановка паттернов в обучающих записях
            #print('Train on {} batches'.format(nb_train_batches))
            train_idx = np.random.permutation(range(train_data.shape[0]))
            for ibatch in tqdm.tqdm(range(nb_train_batches), total=nb_train_batches, desc='Train on {} batches'.format(nb_train_batches)):
                batch_data = train_data[train_idx][ibatch*BATCH_SIZE:(ibatch+1)*BATCH_SIZE]
                ae_model.train_on_batch(x=batch_data, y=batch_data)

            # Оцениваем точность на валидационных данных
            print('Validate on {} batches'.format(nb_val_batches))
            val_loss = 0
            for ibatch in range(nb_val_batches):
                batch_data = val_data[ibatch*BATCH_SIZE:(ibatch+1)*BATCH_SIZE]
                #batch_loss = ae_model.evaluate(x=batch_data, y=batch_data)
                y_batch = ae_model.predict_on_batch(batch_data)
                l2 = np.linalg.norm(batch_data-y_batch, axis=-1)
                batch_loss = np.sum(l2)
                val_loss += batch_loss

            print('val_loss={}'.format(val_loss))
            if val_loss < best_loss:
                print('val_loss improved from {} to {}, storing weights to {}'.format(best_loss, val_loss, weights_path))
                ae_model.save_weights(weights_path)
                best_loss = val_loss
                nb_no_impovements = 0
            else:
                nb_no_impovements += 1
                if nb_no_impovements>10:
                    print('early stopping on {} epochs with no improvements'.format(nb_no_impovements))
                    break


    # Загружаем последние лучшие веса модели
    ae_model.load_weights(weights_path)

    # Теперь можем сохранить декодер, что позволит
    # потом использовать ее для генерации текста.
    with open(arch_filepath, 'w') as f:
        f.write(decoder_model.to_json())

    decoder_model.save_weights(weights_path)

    # Проверим, что модели кодера и декодера дают при последовательном применении
    # тот же результат, что и полная модель автоэнкодера.
    test_data = vtexts[0:BATCH_SIZE]
    y_ae = ae_model.predict_on_batch(test_data)
    y1 = encoder_model.predict_on_batch(test_data)
    y_decoder = decoder_model.predict_on_batch(y1)

    decoded_ae = decode_output(y_ae, v2w)
    decoded_12 = decode_output(y_decoder, v2w)

    for phrase1, phrase2 in zip(decoded_ae, decoded_12):
        print(u'ae={} decoder={}'.format(phrase1, phrase2))



if do_vizualize:
    # Загружаем конфигурацию модели с данными, необходимыми для
    # восстановления архитектуры сетки и формирования входных данных.
    with open(os.path.join(model_folder, 'lstm_ae.config'), 'r') as f:
        model_config = json.load(f)

        max_seq_len = model_config['max_seq_len']
        arch_filepath = model_config['arch_filepath']
        weights_path = model_config['weights_path']
        word_dims = model_config['word_dims']
        latent_dim = model_config['latent_dim']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    nb_generated = 10  # столько случайных фраз сгенерируем и покажем

    # Предполагаем, что скрытые переменные на выходе энкодера имеют диапазон 0..1
    # Это может быть не так, если там поставить relu вместо sigmoid активации.
    # В таком случае надо при тренировке модели оценить диапазон значений скрытых переменных
    # и записать в конфигурацию модели вектор максимальных и минимальных значений.
    X_probe = np.random.uniform(0.0, 1.0, (nb_generated, latent_dim))
    y_probe = model.predict(X_probe)

    # декодируем результаты работы модели
    result_phrases = decode_output(y_probe, v2w)
    for phrase in result_phrases:
        print(u'{}'.format(phrase))
