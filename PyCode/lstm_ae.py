# -*- coding: utf-8 -*-
"""
Для baseline оценки - автоэнкодер и вариационный автоэнкодер предложений
на базе LSTM+Conv (точная конфигурация нейросети настраивается).
Тренируется на подготовленном заранее датасете - см. скрипты prepare_phrases.py
и prepare_vae_dataset.py

Тренировка модели состоит из 2х этапов, которые выполняются отдельным запуском
данного скрипта. Сначала происходит собственно тренировка автоэнкодера (--train), веса
и архитектура нейросеток сохраняются на диск. Во второй части (--estimate) подгружается кодирующая
нейросетка, через нее прогоняются все исходные данные, и происходит
оценка распределения значений компонентов скрытого вектора на выходе кодера. Полученные
гистограммы также сохраняются на диск.

После того, как модель прошла первые два этапа, ее можно использовать для генерации
новых предложений (--generate 1).
"""

from __future__ import print_function, division

from future.utils import iteritems

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
from keras import regularizers

import tqdm
import sklearn.model_selection
import scipy.spatial.distance
from scipy import stats
import pickle
import numpy as np
import sys
import json
import os
import argparse

from word_decoder import W2V_Decoder


# длина вектора на выходе кодирующей части автоэнкодера, фактически это
# длина вектора, представляющего предложение.
latent_dim = 64

NB_EPOCHS = 200  # макс. кол-во эпох обучения

batch_size = 32

# Конфигурация нейросетки:
# arch - общая архитектура: 'ae' для простого сжимающего автоэнкодера, 'vae' для
#        вариационного автоэнкодера
# encoder - структура кодирующей части
# decoder - структура декодера
NET_CONFIG={'arch': 'ae', 'encoder': 'lstm(cnn)', 'decoder': 'lstm,dense'}
#NET_CONFIG={'arch': 'vae', 'encoder': 'lstm(cnn)', 'decoder': 'lstm,dense'}


# Путь к папке с файлами датасетов, подготовленных
# скриптом prepare_vase_dataset.py
data_folder = '../data'

# В этом каталоге будем сохранять файлы с конфигурацией и весами натренированной модели.
model_folder = '../tmp'


def create_ae(net_config, max_seq_len, word_dims, latent_dim, l1, l2):
    """
    Создается классический сжимающий автоэнкодер с архитектурой seq2seq для упаковки
    предложения в вектор фиксированного размера latent_dim.
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

        encoder_final = keras.layers.concatenate(inputs=convs)
    elif net_config['encoder'] == 'lstm':
        encoder_final = recurrent.LSTM(latent_dim, return_sequences=False)(encoder_input)

    if l1 == 0.0 and l2 == 0.0:
        encoder_final = Dense(units=int(latent_dim),
                              activation='tanh',
                              )(encoder_final)
    elif l1 > 0.0 and l2 == 0.0:
        encoder_final = Dense(units=int(latent_dim),
                              activation='tanh',
                              activity_regularizer=regularizers.l1(l1)
                              )(encoder_final)
    else:
        encoder_final = Dense(units=int(latent_dim),
                              activation='tanh',
                              activity_regularizer=regularizers.l1l2(l1, l2)
                              )(encoder_final)

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
    Исходный код взят из https://github.com/twairball/keras_lstm_vae/blob/master/lstm_vae/vae.py

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
        encoder_final = Dense(units=int(latent_dim), activation='tanh')(encoder_merged)
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



def v_cosine(v1, v2):
    s = scipy.spatial.distance.cosine(v1, v2)
    if np.isinf(s) or np.isnan(s):
        s = 0.0

    return s


def find_word_l2(word_vector, words, vectors):
    """
    Поиск слова, максимально близкого к заданному вектору word_vector,
    с использованием евклидового расстояния.
    """
    deltas = vectors - word_vector
    l2 = np.linalg.norm(deltas, axis=-1)
    imin = np.argmin(l2)
    return words[imin]

    #min_dist = 1e38
    #best_word = u''
    #for v, w in v2w:
    #    d = np.linalg.norm(v - word_vector)
    #    if d < min_dist:
    #        min_dist = d
    #        best_word = w
    #return best_word


def decode_output(y, words, vectors):
    """
    Декодируем выходной тензор автоэнкодера, получаем читабельные
    предложения в том же порядке, как входные.
    """
    decoded_phrases = []
    null_word = u'(null)'
    for iphrase in range(y.shape[0]):
        phrase_vectors = y[iphrase]
        phrase_words = []
        for iword in range(y.shape[1]):
            word_vector = phrase_vectors[iword]
            l2 = np.linalg.norm(word_vector)
            best_word = null_word
            if l2>0.1:
                best_word = find_word_l2(word_vector, words, vectors)

            phrase_words.append(best_word)

        while len(phrase_words) > 0 and phrase_words[-1] == null_word:
            phrase_words = phrase_words[:-1]

        decoded_phrases.append(u' '.join(phrase_words))

    return decoded_phrases


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):

    def __init__(self, ae_model, X_data, w2v_decoder, batch_size):
        self.epoch = 0
        self.X_data = X_data
        self.model = ae_model
        self.w2v_decoder = w2v_decoder
        self.batch_size = batch_size
        self.instance_acc_hist = []

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1

        print('\n')
        # Выберем подмножество паттернов, которые мы прогоним через модель
        idx = np.random.permutation( range(self.batch_size) )
        #idx = list(range(self.batch_size))
        X_test = self.X_data[idx]
        y_test = self.model.predict(X_test, verbose=0)

        # Расчет точности восстановления полного теста предложения работает достаточно
        # медленно, так как надо делать полный перебор записей в словаре для каждого слова.
        # Поэтому ограничиваем количество проверяемых сэмплов.
        nb_max_tested = 5000
        nb_tested = 0
        nb_errors = 0

        for iphrase in range(min(nb_max_tested,X_test.shape[0])):
            input_phrase = self.w2v_decoder.decode_output(X_test[iphrase:iphrase+1])[0]
            output_phrase = self.w2v_decoder.decode_output(y_test[iphrase:iphrase+1])[0]

            nb_tested += 1
            if input_phrase != output_phrase:
                nb_errors += 1

            if iphrase<10:
                print(colors.ok + '☑ ' + colors.close if input_phrase == output_phrase else colors.fail + '☒ ' + colors.close,
                      end='')

                print(u'{} ==> {}'.format(input_phrase, output_phrase))

        acc = float(nb_tested-nb_errors)/nb_tested
        print('\nPer sample accuracy={}'.format(acc))
        self.instance_acc_hist.append(acc)

    def save_instance_acc(self, filepath):
        with open(filepath, 'w') as wrt:
            for acc in self.instance_acc_hist:
                wrt.write('{}\n'.format(acc))

# -----------------------------------------------------------

parser = argparse.ArgumentParser(description='Training autoencoders and generating random phrases')
parser.add_argument('--train', default=0, type=int, help='train autoencoder model')
parser.add_argument('--estimate', default=0, type=int, help='estimate latent PDFs with trained model')
parser.add_argument('--generate', default=0, type=int, help='use trained model and PDFs estimations for random text generation')
parser.add_argument('--epochs', default=NB_EPOCHS, type=int, help='max number of epochs when training the model')
parser.add_argument('--batch_size', default=32, type=int, help='size of minibatch when training the model')
parser.add_argument('--latent_dim', default=64, type=int, help='length of encoder output vectors')
parser.add_argument('--l1', default=0.0, type=float, help='L1 regularization value, e.g. 1e-6')
parser.add_argument('--l2', default=0.0, type=float, help='L2 regularization value, e.g. 1e-4')
parser.add_argument('--arch', default='ae', type=str, help='"ae" or "vae"')


args = parser.parse_args()

do_train = args.train
do_estimate_pdfs = args.estimate
do_vizualize = args.generate
nb_epochs = args.epochs
batch_size = args.batch_size
latent_dim = args.latent_dim
l1 = args.l1
l2 = args.l2
NET_CONFIG['arch'] = args.arch

# -----------------------------------------------------------

# Словарь с парами слово-вектор для печати читабельных результатов
w2v_decoder = W2V_Decoder(os.path.join(data_folder, 'word2vec.pkl'))

# -----------------------------------------------------------

while not do_train and not do_estimate_pdfs and not do_vizualize:
    print('1 - train the model\n2 - estimate latent PDFs\n3 - generate random texts with trained model')
    run_mode = raw_input('? ').decode(sys.stdout.encoding).strip().lower()
    if run_mode == '1':
        do_train = True
        break
    elif run_mode == '2':
        do_vizualize = True
        break
    elif run_mode == '3':
        do_estimate_pdfs = True
        break
    else:
        print('Invalid selection, please re-enter your choice')


if do_train:
    # Загружаем подготовленный датасет с векторизованными фразами.
    vtexts = np.load(os.path.join(data_folder,'vtexts.npz'))
    vtexts = vtexts['arr_0']

    w2v_dims = vtexts.shape[2]
    print('w2v_dims={0}'.format(w2v_dims))

    max_seq_len = vtexts.shape[1]
    print('max_seq_len={}'.format(max_seq_len))

    # ---------------------------------------------------------------
    # Создание нейросетки

    if NET_CONFIG['arch'] == 'ae':
        ae_model, encoder_model, decoder_model = create_ae(NET_CONFIG,
                                                           max_seq_len,
                                                           w2v_dims,
                                                           latent_dim,
                                                           l1, l2)
    elif NET_CONFIG['arch'] == 'vae':
        ae_model, encoder_model, decoder_model = create_vae(NET_CONFIG, max_seq_len, w2v_dims, latent_dim, batch_size)
    else:
        raise NotImplemented()

    print('ae_model:')
    ae_model.summary()

    print('\nencoder_model:')
    encoder_model.summary()

    print('\ndecoder_model:')
    decoder_model.summary()

    decoder_weights_path = os.path.join(model_folder, 'lstm_ae_decoder.weights')
    decoder_arch_filepath = os.path.join(model_folder, 'lstm_ae_decoder.arch')

    encoder_weights_path = os.path.join(model_folder, 'lstm_ae_encoder.weights')
    encoder_arch_filepath = os.path.join(model_folder, 'lstm_ae_edcoder.arch')

    monitor_metric = 'val_loss'
    model_checkpoint = ModelCheckpoint(decoder_weights_path, monitor=monitor_metric,
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    viz = VisualizeCallback(ae_model, vtexts, w2v_decoder, batch_size)

    callbacks = [viz, model_checkpoint, early_stopping]

    val_size = (int(0.2*vtexts.shape[0]) // batch_size) * batch_size
    train_data, val_data = sklearn.model_selection.train_test_split( vtexts, test_size=val_size, random_state=123456)

    # для VAE нужно, чтобы данные были точно выровнены на BATCH_SIZE.
    if (train_data.shape[0]%batch_size) != 0:
        train_data = train_data[: (train_data.shape[0] // batch_size) * batch_size]

    if nb_epochs>0:
        hist = ae_model.fit(x=train_data,
                            y=train_data,
                            validation_data=(val_data, val_data),
                            batch_size=batch_size,
                            epochs=nb_epochs,
                            verbose=1,
                            callbacks=callbacks,
                            )

    viz.save_instance_acc(os.path.join(model_folder, 'instance_accuracy.l1={}.l2={}.dat'.format(l1, l2)))

    # Загружаем последние лучшие веса полной модели
    ae_model.load_weights(decoder_weights_path)

    # Теперь можем сохранить декодер, что позволит
    # потом использовать ее для генерации текста.
    with open(decoder_arch_filepath, 'w') as f:
        f.write(decoder_model.to_json())

    decoder_model.save_weights(decoder_weights_path)

    # Сохраняем также энкодер, так как он может понадобиться для экспериментов
    with open(encoder_arch_filepath, 'w') as f:
        f.write(encoder_model.to_json())

    encoder_model.save_weights(encoder_weights_path)

    # сохраним конфиг модели в json файлике.
    model_config = {
        'max_seq_len': max_seq_len,
        'latent_dim': latent_dim,
        'decoder_arch_filepath': decoder_arch_filepath,
        'decoder_weights_path': decoder_weights_path,
        'encoder_arch_filepath': encoder_arch_filepath,
        'encoder_weights_path': encoder_weights_path,
        'word_dims': w2v_dims,
        'NET_CONFIG': NET_CONFIG
    }

    with open(os.path.join(model_folder, 'lstm_ae.config'), 'w') as f:
        json.dump(model_config, f)


    if False:
        # Проверим, что модели кодера и декодера дают при последовательном применении
        # тот же результат, что и полная модель автоэнкодера.
        test_data = vtexts[0:batch_size]
        y_ae = ae_model.predict_on_batch(test_data)
        y1 = encoder_model.predict_on_batch(test_data)
        y_decoder = decoder_model.predict_on_batch(y1)

        decoded_ae = decode_output(y_ae, v2w)
        decoded_12 = decode_output(y_decoder, v2w)

        for phrase1, phrase2 in zip(decoded_ae, decoded_12):
            print(u'ae={} decoder={}'.format(phrase1, phrase2))

    if False:
        # оценка точности модели с лучшими весами
        # для больших датасетов валидация может растянутся на часы, так что
        # ограничиваем кол-во проверяемых сэмплов.
        nb_val_batches = min(100, val_data.shape[0] // batch_size)
        print('\nFinal validation on {} batches, {} samples'.format(nb_val_batches, val_data.shape[0]))
        l2_loss = 0.0
        cos_loss = 0.0
        nb_tested = 0
        nb_errors = 0
        for ibatch in tqdm.tqdm(range(nb_val_batches), total=nb_val_batches):
            batch_data = val_data[ibatch * batch_size:(ibatch + 1) * batch_size]
            #batch_loss = ae_model.evaluate(x=batch_data, y=batch_data)
            y_batch = ae_model.predict_on_batch(batch_data)
            l2 = np.linalg.norm(batch_data-y_batch, axis=-1)
            batch_loss = np.sum(l2)
            l2_loss += batch_loss

            for iphrase in range(batch_data.shape[0]):
                for iword in range(batch_data.shape[1]):
                    c = v_cosine(batch_data[iphrase, iword, :], y_batch[iphrase, iword, :])
                    cos_loss += abs(1.0-c)

                input_phrase = decode_output(batch_data[iphrase:iphrase + 1], v2w)[0]
                output_phrase = decode_output(y_batch[iphrase:iphrase + 1], v2w)[0]
                nb_tested += 1
                if input_phrase != output_phrase:
                    nb_errors += 1

        print('final L2 loss={}'.format(l2_loss))
        print('final cos loss={}'.format(cos_loss))

        acc = float(nb_tested-nb_errors)/nb_tested
        print('final accuracy per sample={}'.format(acc))


if do_estimate_pdfs:
    print('Estimating PDFs of latent variables...')
    # Будем оценивать распределение значений на скрытом слое (выход кодирующей
    # части автоэнкодера), пропуская через него все исходные данные.
    vtexts = np.load(os.path.join(data_folder,'vtexts.npz'))
    vtexts = vtexts['arr_0']

    # Загружаем конфигурацию модели, натренированной на предыдущем этапе --train
    with open(os.path.join(model_folder, 'lstm_ae.config'), 'r') as f:
        model_config = json.load(f)

        max_seq_len = model_config['max_seq_len']
        decoder_arch_filepath = model_config['decoder_arch_filepath']
        decoder_weights_path = model_config['decoder_weights_path']
        encoder_arch_filepath = model_config['encoder_arch_filepath']
        encoder_weights_path = model_config['encoder_weights_path']
        word_dims = model_config['word_dims']
        latent_dim = model_config['latent_dim']
        NET_CONFIG = model_config['NET_CONFIG']

    # восстанавливаем кодер
    with open(encoder_arch_filepath, 'r') as f:
        encoder_model = model_from_json(f.read())

    encoder_model.load_weights(encoder_weights_path)

    # Оценим распределение значений переменных в скрытом слое (на выходе энкодера)
    # Для этого прогоняем через кодер исходные данные.
    nb_batches = vtexts.shape[0] // batch_size
    nb_rec = nb_batches * batch_size
    latents = np.zeros((nb_rec, latent_dim))

    print('\nGenerating latent vectors on {} batches'.format(nb_batches))
    for ibatch in tqdm.tqdm(range(nb_batches), total=nb_batches):
        batch_data = vtexts[ibatch * batch_size:(ibatch + 1) * batch_size]
        y_batch = encoder_model.predict_on_batch(batch_data)
        latents[ibatch * batch_size:(ibatch + 1) * batch_size, :] = y_batch

    print('Storing {} latent vectors'.format(nb_rec))
    with open(os.path.join(model_folder, 'latents.npz'), 'wb') as f:
        np.savez_compressed(f, latents)

    latent_mins = np.zeros((latent_dim))
    latent_maxs = np.zeros((latent_dim))
    for idim in range(latent_dim):
        latent_mins[idim] = np.amin(latents[:, idim])
        latent_maxs[idim] = np.amax(latents[:, idim])

    # распределение значений по каждой переменной будем аппроксимировать
    # гистограммой.
    latent_histos = []
    for idim in range(latent_dim):
        h = np.histogram(a=latents[:, idim], range=(latent_mins[idim], latent_maxs[idim]), bins=20)
        latent_histos.append(h)

    latent_histos_path = os.path.join(model_folder, 'latent_histos.pkl' )
    with open(latent_histos_path, 'wb') as f:
        pickle.dump(latent_histos, f)

    # дополняем конфиг модели данными о гистограммах
    model_config['latent_histos_path'] = latent_histos_path
    with open(os.path.join(model_folder, 'lstm_ae.config'), 'w') as f:
        json.dump(model_config, f)


if do_vizualize:

    # Загружаем конфигурацию модели с данными, необходимыми для
    # восстановления архитектуры сетки и формирования входных данных.
    with open(os.path.join(model_folder, 'lstm_ae.config'), 'r') as f:
        model_config = json.load(f)

        max_seq_len = model_config['max_seq_len']
        decoder_arch_filepath = model_config['decoder_arch_filepath']
        decoder_weights_path = model_config['decoder_weights_path']
        encoder_arch_filepath = model_config['encoder_arch_filepath']
        encoder_weights_path = model_config['encoder_weights_path']
        word_dims = model_config['word_dims']
        latent_dim = model_config['latent_dim']
        latent_histos_path = model_config['latent_histos_path']
        NET_CONFIG = model_config['NET_CONFIG']

    with open(decoder_arch_filepath, 'r') as f:
        decoder_model = model_from_json(f.read())

    decoder_model.load_weights(decoder_weights_path)

    nb_generated = 10  # столько случайных фраз сгенерируем и покажем

    if False:  #NET_CONFIG['arch'] == 'vae':
        # Для вариационного автоэнкодера нужно на вход декодера подавать
        # нормально распределенный шум с единичной дисперсией.
        X_probe = np.random.normal(loc=1.0, scale=1.0, size=(nb_generated, latent_dim))
    else:
        # Мы должны подавать на входе декодера вектор скрытых переменных с разбросом
        # значений отдельных компонентов, примерно соответствующим распределению
        # для тренировочных данных. Мы уже собрали гистограммы для каждой скрытой переменной,
        # их надо загрузить и использовать.
        with open(latent_histos_path, 'rb') as f:
            latent_histos = pickle.load(f)

        pdfs = []
        for idim, histo in enumerate(latent_histos):
            pdfs.append(stats.rv_histogram(histogram=histo))

        X_probe = np.zeros((nb_generated, latent_dim))
        for idim in range(latent_dim):
            p = pdfs[idim].rvs(size=nb_generated)
            X_probe[:, idim] = p

    # Пропускаем подготовленный вектор через декодер.
    y_probe = decoder_model.predict(X_probe)

    # декодируем результаты работы модели
    result_phrases = w2v_decoder.decode_output(y_probe)
    for phrase in result_phrases:
        print(u'{}'.format(phrase))
