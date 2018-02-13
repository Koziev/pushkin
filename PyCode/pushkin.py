# -*- coding: utf-8 -*-
"""
Character Language Model с использованием RNN (LSTM) библиотеки keras.
Модель учится предсказывать следующий символ в предложении (teacher forcing).
"""

from __future__ import print_function, division

import os
import numpy as np
import codecs
import future.utils
import itertools
import tqdm
import future.utils

import sklearn.model_selection

import keras.callbacks
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding, Bidirectional, LSTM
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Путь к текстовому файлу, на котором модель обучается.
# chars_path = u'../data/ЕвгенийОнегин.txt'
chars_path = u'../data/phrases.txt'

START_CHAR = u'\b'
END_CHAR = u'\t'

# спецсимвол для выравнивания длины предложений.
SENTINEL_CHAR = u'\a'

NB_EPOCHS = 100

BATCH_SIZE = 64

MAX_NB_PATTERNS = 100000

ENCODER_ARCH = 'lstm(conv)'
#ENCODER_ARCH = 'lstm'


def pad_seq(s, max_len):
    return s.rjust(max_len, SENTINEL_CHAR)
    # return s.ljust(max_len, SENTINEL_CHAR)


class Alphabet(object):
    def __init__(self, chars):
        self.chars = chars

        if SENTINEL_CHAR in chars:
            # для получения символа по его индексу при визуализации сгенерированного текста
            # символ-заполнитель должен получить код 0, чтобы работало маскирование обучения.
            self.index2char = dict(
                (i, c) for i, c in
                enumerate(itertools.chain([SENTINEL_CHAR], filter(lambda c: c != SENTINEL_CHAR, chars))))

            # 1-hot кодирование символов для целевого тензора
            self.char2index = dict((c, i) for i, c in future.utils.iteritems(self.index2char))
        else:
            self.char2index = dict((c, i) for (i, c) in enumerate(chars))
            self.index2char = dict((i, c) for (c, i) in future.utils.iteritems(self.char2index))

    def count(self):
        return len(self.chars)

    def get_index(self, char):
        return self.char2index[char]

    def get_char(self, index):
        return self.index2char[index]


# экземпляр класса вызывается в конце каждой эпохи обучения модели
# и выполняет генерацию текста, позволяя оперативно оценить качество
# генерации.
class TextGenerator(keras.callbacks.Callback):
    # в этот файл будем записывать генерируемые моделью строки
    output_samples_path = '../tmp/pushkin.generated.txt'

    def __init__(self, max_seq_len, input_chars, output_chars, model, nb_generated_phrases=5):
        self.max_seq_len = max_seq_len
        self.input_chars = input_chars
        self.output_chars = output_chars
        self.model = model
        self.nb_generated_phrases = nb_generated_phrases
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.epoch = 0
        # удалим тексты, сгенерированные в предыдущих запусках скрипта
        if os.path.isfile(self.output_samples_path):
            os.remove(self.output_samples_path)

    # helper function to sample an index from a probability array
    # взято из https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    def sample(self, a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / (np.sum(np.exp(a)) + 1e-5)
        a = np.random.multinomial(1, a, 1)
        return np.argmax(a)

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        print('')
        if True:  # (self.epoch%10) == 0:
            with codecs.open(self.output_samples_path, 'a', 'utf-8') as fsamples:
                fsamples.write(u'\n' + '=' * 50 + '\nepoch=' + str(self.epoch) + u'\n')

                # генерируем по 4 цепочки (строфы) для нескольких температур
                for temp in [0.3, 0.6, 0.9, 1.0, 1.1]:
                    for igener in range(0, 1):
                        # сделаем сэмплинг цепочки символов.
                        # начинаем всегда с символа START_CHAR.
                        self.model.reset_states()

                        # буфер для накопления сгенерированной строки
                        # Инициализируем затравочным значением
                        #sample_seq = START_CHAR
                        sample_seq = START_CHAR+u'я вижу '

                        X_gener = np.zeros((1, self.max_seq_len), dtype='int32')
                        while len(sample_seq) < max_seq_len:
                            X_gener.fill(0)
                            for itime, uch in enumerate(pad_seq(sample_seq, self.max_seq_len)):
                                X_gener[0, itime] = self.input_chars.get_index(uch)

                            y_gener = self.model.predict(X_gener, batch_size=1, verbose=0)[0]
                            yv = y_gener

                            selected_index = self.sample(yv, temp)
                            selected_char = self.output_chars.get_char(selected_index)

                            if selected_char == END_CHAR:
                                break

                            sample_seq += selected_char

                        sample_seq = sample_seq.replace(START_CHAR, u'').strip()
                        print(u'sample t={} str={}'.format(temp, sample_seq))
                        fsamples.write('\nt={}\n\n'.format(temp))
                        fsamples.write(sample_seq + u'\n')


# Соберем список встретившихся символов.
# Токены начала и конца цепочки добавляем
# руками, так как в явном виде их в исходном корпусе нет.
input_chars_set = set([SENTINEL_CHAR, START_CHAR, END_CHAR])
output_chars_set = set([START_CHAR, END_CHAR])
with codecs.open(chars_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        input_chars_set.update(line)
        output_chars_set.update(line)

input_chars = Alphabet(input_chars_set)
output_chars = Alphabet(output_chars_set)
print('Number of unique output chars={}'.format(output_chars.count()))


# количество накопленных в samples последовательностей
sample_count = 0

# здесь получим макс. длину последовательности символов без учета добавляемых токенов <s> и </s>
max_seq_len = 0

# идем по файлу с предложениями, извлекаем сэмплы для обучения
print(u'Loading samples from {}'.format(chars_path))

sample_count = 0
samples = []

if u'ЕвгенийОнегин' in chars_path:
    endchars = set(u'!;.?')
    sample_buf = u''
    with codecs.open(chars_path, 'r', 'utf-8') as f:
        for line in f:
            charseq = line.strip()

            if len(charseq) > 0:
                if len(sample_buf) > 0:
                    sample_buf += u'\n'
                sample_buf += charseq

                if len(sample_buf) > 0 and sample_buf[-1] in endchars:
                    sample_buf = START_CHAR + sample_buf + END_CHAR
                    xlen = len(sample_buf)
                    if xlen > 3:
                        samples.append(sample_buf)
                        sample_count += 1
                        max_seq_len = max(max_seq_len, xlen)

                    sample_buf = u''
else:
    with codecs.open(chars_path, 'r', 'utf-8') as f:
        for line in f:
            sample_buf = line.strip()
            if len(sample_buf) > 0:
                sample_buf = START_CHAR + sample_buf + END_CHAR
                xlen = len(sample_buf)
                if xlen > 3:
                    samples.append(sample_buf)
                    sample_count += 1
                    max_seq_len = max(max_seq_len, xlen)

print('sample_count={}'.format(sample_count))
print('max_seq_len={}'.format(max_seq_len))

# Генерируем паттерны для обучения и валидации.
# Для этого у каждого исходного сэмпла перебираем позиции символов,
# и берем символ и цепочку символов слева от него. Получается соответственно
# целевое значение и входные данные.
# Хитрые манипуляции по разделению датасета на тренировочный и валидационный набор
# вызваны необходимость не допустить утечку тренировочных данных в валидацию, но при
# этом данные могут в принципе повторяться, особенно для первых символов каждого
# исходного предложения.
data_set = set()
data = []
for sample in samples:
    for itime in range(1, len(sample) - 1):
        x = sample[:itime]
        y = sample[itime]
        pattern = (x, y)
        data_set.add(pattern)
        data.append(pattern)

        if len(data) >= MAX_NB_PATTERNS:
            break

train_set, val_set = sklearn.model_selection.train_test_split(list(data_set),
                                                              test_size=0.05,
                                                              random_state=123456)

train_set = set(train_set)
val_set = set(val_set)

nb_train = len(train_set)
nb_val = len(val_set)

print('nb_train={} nb_val={}'.format(nb_train, nb_val))

xlen = max_seq_len

# тензоры для входных последовательностей и выходных эталонных данных
X_train = np.zeros((nb_train, xlen), dtype=np.int32)
y_train = np.zeros((nb_train, output_chars.count()), dtype=np.bool)

X_val = np.zeros((nb_val, xlen), dtype=np.int32)
y_val = np.zeros((nb_val, output_chars.count()), dtype=np.bool)

itrain = 0
ival = 0
for pattern in tqdm.tqdm(data_set, total=len(data_set), desc='Vectorization'):
    if pattern in train_set:
        X_data = X_train
        y_data = y_train
        idata = itrain
        itrain += 1
    else:
        X_data = X_val
        y_data = y_val
        idata = ival
        ival += 1

    x = pattern[0]
    y = pattern[1]

    # слева или справа дополняем символами \a, чтобы все последовательности имели одинаковую длину
    seq = pad_seq(x, max_seq_len)
    for i, c in enumerate(seq):
        X_data[idata, itime] = input_chars.get_index(c)
    y_data[idata, output_chars.get_index(y)] = True

# -------------------------------------------------------------
# Собираем нейросетку, которая будет предсказывать следующий символ, взяв цепочку ранее
# выбранных символов.

# на вход поступают коды символов
model_input = Input(shape=(max_seq_len,), dtype='int32', name='input_chars')

# этот шаг преобразует порядковый код каждого символа в вектор заданной длины,
# причем можно включить настройку векторов символов.
if True:
    char_dims = 8
    model_output = Embedding(input_dim=input_chars.count(),
                            output_dim=char_dims,
                            mask_zero=ENCODER_ARCH == 'lstm',
                            trainable=True)(model_input)
else:
    # вариант с 1-hot кодированием символов
    char_dims = input_chars.count()
    model_output = Embedding(input_dim=char_dims,
                            output_dim=char_dims,
                            embeddings_initializer=keras.initializers.Identity(),
                            mask_zero=ENCODER_ARCH == 'lstm',
                            trainable=False)(model_input)

# Следующий слой упаковывает цепочку символов в действительный вектор заданной длины.
rnn_size = 128

if ENCODER_ARCH == 'lstm':
    # output_char = Bidirectional(LSTM(rnn_size // 2, return_sequences=False))(output_char)
    model_output = LSTM(rnn_size, return_sequences=False)(model_output)
elif ENCODER_ARCH == 'lstm(conv)':
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

        lstm = Bidirectional(LSTM(rnn_size, return_sequences=False))

        conv_layer = conv(model_output)
        conv_layer = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer)
        conv_layer = lstm(conv_layer)
        convs.append(conv_layer)

        encoder_size += rnn_size

    encoder_merged = keras.layers.concatenate(inputs=convs)
    model_output = Dense(units=int(encoder_size))(encoder_merged)

# Далее идет классификатор для выбора номера следующего символа.
# output_char = Dense(units=rnn_size, activation='relu')(output_char)
# output_char = Dense(units=rnn_size, activation='relu')(output_char)
model_output = Dense(units=output_chars.count(), activation='softmax')(model_output)

model = Model(inputs=model_input, outputs=model_output)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

weights_path = '../tmp/pushkin.weights'
monitor_metric = 'val_acc'
model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                   verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

text_generator = TextGenerator(max_seq_len, input_chars, output_chars, model)

# Тренировка модели
acc = model.fit(x=X_train,
                y=y_train,
                validation_data=(X_val, y_val),
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCHS,
                callbacks=[text_generator, model_checkpoint, early_stopping])

# Загрузим лучшие веса
model.load_weights(weights_path)

# сгенерируем кусок текста с лучшим вариантом модели
text_generator = TextGenerator(max_seq_len, input_chars, output_chars, model, 100)
text_generator.on_epoch_end(None)
