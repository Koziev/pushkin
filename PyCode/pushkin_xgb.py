# -*- coding: utf-8 -*-
"""
Генеративная Character Language Model с использованием XGBoost
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
import xgboost
from scipy.sparse import lil_matrix

import sklearn.model_selection

# Путь к текстовому файлу, на котором модель обучается.
# chars_path = u'../data/ЕвгенийОнегин.txt'
chars_path = u'../data/phrases.txt'

START_CHAR = u'\b'
END_CHAR = u'\t'

# спецсимвол для выравнивания длины предложений.
SENTINEL_CHAR = u'\a'

MAX_NB_PATTERNS = 1000000


def get_shingles(s):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(3)])]


def pad_seq(s, max_len):
    return s.rjust(max_len, SENTINEL_CHAR)


# Соберем список встретившихся символов.
# Токены начала и конца цепочки, а также перевода строки добавляем
# руками, так как в явном виде их в корпусе нет.
chars_set = set([START_CHAR, END_CHAR])
with codecs.open(chars_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        chars_set.update(line.strip())

nchar = len(chars_set)
print('Number of unique chars={}'.format(len(chars_set)))

index2char = dict((i, c) for i, c in enumerate(chars_set))
char2index = dict((c, i) for i, c in future.utils.iteritems(index2char))

# количество накопленных в samples последовательностей
sample_count = 0

# здесь получим макс. длину последовательности символов без учета добавляемых токенов <s> и </s>
max_seq_len = 0

# идем по файлу с предложениями, извлекаем сэмплы для обучения
print(u'Loading samples from {}'.format(chars_path))

sample_count = 0
samples = []

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
    for itime in range(len(sample)):
        x = sample[:itime]
        y = sample[itime]
        pattern = (x, y)
        data_set.add(pattern)
        data.append(pattern)

        if len(data) >= MAX_NB_PATTERNS:
            break

train_set, val_set = sklearn.model_selection.train_test_split(list(data_set),
                                                              test_size=0.2,
                                                              random_state=123456)

train_set = set(train_set)
val_set = set(val_set)

nb_train = len(train_set)
nb_val = len(val_set)

print('nb_train={} nb_val={}'.format(nb_train, nb_val))


# Цепочку символов слева от генерируемого символа представляем
# как bag-of-shingles. Нам нужен словарь шинглов, который мы получим
# анализом предложений в датасете.
all_shingles = set()
for x, y in data_set:
    all_shingles.update(get_shingles(x))

shingle2index = dict((s, i) for (i, s) in enumerate(all_shingles))

# Входная матрица должна представлять достаточную информацию о символах
# справа. В том числе: предыдущие NB_PREV_CHAR, шинглы в левой цепочке
NB_PREV_CHAR = 4
nb_features = NB_PREV_CHAR*nchar + len(all_shingles)

print('nb_features={}'.format(nb_features))

X_train = lil_matrix((nb_train, nb_features), dtype='bool')
y_train = []

X_val = lil_matrix((nb_val, nb_features), dtype='bool')
y_val = []

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

    lenx = len(x)
    prev_chars = x[max(0, lenx-NB_PREV_CHAR):lenx][::-1]

    idim = 0
    for i, c in enumerate(prev_chars):
        X_data[idata, idim + i*nchar + char2index[c]] = True
    idim += NB_PREV_CHAR*nchar

    for shingle in get_shingles(x):
        if shingle in shingle2index:
            X_data[idata, idim + shingle2index[shingle]] = True

    y_data.append(char2index[y])

D_train = xgboost.DMatrix(X_train, y_train, silent=0)
D_val = xgboost.DMatrix(X_val, y_val, silent=0)

xgb_params = {
    'booster': 'gbtree',
    # 'n_estimators': _n_estimators,
    'subsample': 1.0,
    'max_depth': 6,
    'seed': 123456,
    'min_child_weight': 1,
    'eta': 0.10,
    'gamma': 0.01,
    'colsample_bytree': 1.0,
    'colsample_bylevel': 1.0,
    'eval_metric': 'merror',
    'objective': 'multi:softmax',
    'num_class': nchar,
    'silent': 1,
    # 'updater': 'grow_gpu'
}

print('Train model...')
cl = xgboost.train(xgb_params,
                   D_train,
                   evals=[(D_val, 'val')],
                   num_boost_round=8000,
                   verbose_eval=50,
                   early_stopping_rounds=50)

print('Training is finished')
y_pred = cl.predict(D_val)
score = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred)
print('accuracy={}'.format(score))


