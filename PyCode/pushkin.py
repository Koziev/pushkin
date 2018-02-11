# -*- coding: utf-8 -*- 
"""
Character Language Model с использованием RNN (LSTM) библиотеки keras

Входной корпус - "Евгений Онегин" с удаленными номерами глав и т.п.
Файл с корпусом лежит рядом с исходником.

Модель учится предсказывать следующий символ в предложении (teacher forcing).

Корпус разбиваем на строфы. Каждая строфа превращается в одну входную
цепочку, символ \n задает границы строк в строфе.
 
В режиме генерации модель дает вероятности для каждого из символов
в следующей позиции.
 
(c) kelijah 2016
"""

from __future__ import print_function, division

import os
import numpy
import codecs
import future.utils
import itertools
import tqdm

import sklearn.model_selection

import keras.callbacks
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding, Bidirectional, LSTM
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Путь к текстовому файлу, на котором модель обучается.
chars_path = u'../data/ЕвгенийОнегин.txt'

START_CHAR = u'\b'
END_CHAR = u'\t'

# спецсимвол для выравнивания длины предложений.
SENTINEL_CHAR = u'\a'

NB_EPOCHS = 100

BATCH_SIZE = 32

# экземпляр класса вызывается в конце каждой эпохи обучения модели
# и выполняет генерацию текста, позволяя оперативно оценить качество
# генерации.
class TextGenerator(keras.callbacks.Callback):

    # в этот файл будем записывать генерируемые моделью строки
    output_samples_path = '../tmp/samples.txt'

    def __init__(self, id2char,model):
        self.id2char = id2char
        self.model = model

    def on_train_begin(self, logs={}):
        self.epoch = 0
        # удалим тексты, сгенерированные в предыдущих запусках скрипта
        if os.path.isfile(self.output_samples_path):
            os.remove(self.output_samples_path)

    # helper function to sample an index from a probability array
    # взято из https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    def sample(self, a, temperature=1.0):
        a = numpy.log(a) / temperature
        a = numpy.exp(a) / numpy.sum(numpy.exp(a))
        return numpy.argmax(numpy.random.multinomial(1, a, 1))

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        if (self.epoch%10) == 0:
            with codecs.open(self.output_samples_path, 'a', 'utf-8') as fsamples:
                fsamples.write( u'\n' + '='*50 + '\nepoch=' + str(self.epoch) + u'\n' )
                        
                # генерируем по 4 цепочки (строфы) для нескольких температур
                for temp in [0.3, 0.6, 0.9, 1.0, 1.1]:
                    for igener in range(0, 4):
                        # сделаем сэмплинг цепочки символов
                        # начинаем всегда с символа <s>
                        last_char = START_CHAR
                        self.model.reset_states();
                                
                        # буфер для накопления сгенерированной строки
                        sample_str = u''
                        sample_seq = last_char
                            
                        while len(sample_str) < 300:
                            xlen = len(sample_seq)
                            X_gener = numpy.zeros((1,xlen,input_size))
                                
                            for itime, uch in enumerate(list(sample_seq)):
                                X_gener[0, itime,:] = char2vector[uch]
                            
                            # получаем результат - цепочка предсказаний, из которой нам нужен только
                            # последний вектор
                            Y_gener = self.model.predict(X_gener, batch_size=1, verbose=0)[0, :]
                            yv = Y_gener[xlen-1, :]
    
                            selected_index = self.sample(yv, temp)
                            selected_char = id2char[selected_index]
    
                            if selected_char == END_CHAR:
                                break
                                
                            sample_str = sample_str + selected_char
                            sample_seq = sample_seq + selected_char
                            last_char = selected_char
                                    
                        print(u'sample t={} str={}'.format(temp, sample_str))
                        fsamples.write('\nt={}\n\n'.format(temp))
                        fsamples.write( sample_str + u'\n' )



# Соберем список встретившихся символов.
# Токены начала и конца цепочки, а также перевода строки добавляем
# руками, так как в явном виде их в корпусе нет.
chars_set = set([SENTINEL_CHAR, START_CHAR, u'\n', END_CHAR])
print(u'Reading char sequences from '.format(chars_path))
with codecs.open(chars_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        chars_set.update(line.strip())

nchar = len(chars_set)
print('Number of unique chars={}'.format(len(chars_set)))

# для получения символа по его индексу при визуализации сгенерированного текста
# символ-заполнитель должен получить код 0, чтобы работало маскирование обучения.
index2char = dict((i, c) for i, c in enumerate(itertools.chain([SENTINEL_CHAR], filter(lambda c: c != SENTINEL_CHAR, chars_set))))

# 1-hot кодирование символов для целевого тензора
char2index = dict((c, i) for i, c in future.utils.iteritems(index2char))

features_size = 150 # кол-во элементов в RNN
input_size = nchar
output_size = nchar

# количество накопленных в samples последовательностей
sample_count = 0

# здесь получим макс. длину последовательности символов без учета добавляемых токенов <s> и </s>
max_seq_len = 0

# идем по файлу с предложениями, извлекаем сэмплы для обучения
print(u'Loading samples from {}'.format(chars_path))

endchars = set(u'!;.?')
sample_buf = u''
sample_count=0
samples = []
with codecs.open(chars_path, 'r', 'utf-8') as f:
    for line in f:
        charseq = line.strip()
        
        if len(charseq) > 0:
            if len(sample_buf)>0:
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
    for itime in range(len(sample)-1):
        x = sample[:itime]
        y = sample[itime]
        pattern = (x, y)
        data_set.add(pattern)
        data.append(pattern)

train_set, val_set = sklearn.model_selection.train_test_split( list(data_set),
                                                               test_size=0.2,
                                                               random_state=123456)

train_set = set(train_set)
val_set = set(val_set)

nb_train = len(train_set)
nb_val = len(val_set)

print('nb_train={} nb_val={}'.format(nb_train, nb_val))

xlen = max_seq_len

# тензоры для входных последовательностей и выходных эталонных данных
X_train = numpy.zeros((nb_train, xlen), dtype=numpy.int32)
y_train = numpy.zeros((nb_train, input_size), dtype=numpy.bool)

X_val = numpy.zeros((nb_val, xlen), dtype=numpy.int32)
y_val = numpy.zeros((nb_val, input_size), dtype=numpy.bool)

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
    seq = x.rjust(max_seq_len, SENTINEL_CHAR)

    for i, c in enumerate(seq):
        X_data[idata, itime] = char2index[c]
        y_data[idata, :] = char2index[y]

# -------------------------------------------------------------
# Собираем нейросетку, которая будет предсказывать следующий символ, взяв цепочку ранее
# выбранных символов.

# на вход поступают коды символов
input_chars = Input(shape=(max_seq_len,), dtype='int32', name='input_chars')

# этот шаг преобразует порядковый код каждого символа в вектор заданной длины,
# причем можно включить настройку векторов символов.
char_dims = 16
output_char = Embedding(input_dim=nchar, output_dim=char_dims, mask_zero=True, trainable=True)(input_chars)

# Следующий слой упаковывает цепочку символов в действительный вектор заданной длины.
rnn_size = 64
output_char = Bidirectional(LSTM(rnn_size // 2, return_sequences=False))(output_char)

# Далее идет классификатор для выбора номера следующего символа.
output_char = Dense(units=rnn_size)(output_char)
output_char = Dense(units=nchar, activation='softmax')(output_char)

model = Model(inputs=input_chars, outputs=output_char)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

weights_path = '../tmp/pushkin.weights'
monitor_metric = 'val_acc'
model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                   verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

text_generator = TextGenerator(index2char, model)

# Тренировка модели
acc = model.fit( x=X_train,
                 y=y_train,
                 validation_data=(X_val, y_val),
                 batch_size=BATCH_SIZE,
                 nb_epoch=NB_EPOCHS,
                 callbacks=[text_generator, model_checkpoint, early_stopping] )


# Загрузим лучшие веса
model.load_weights(weights_path)

# todo - сгенерируем кусок текста
pass
