# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
import pickle
import numpy as np
import zipfile
import re
import itertools

data_folder = './data'


def rpad(s, n):
    return s + u' '*(n-len(s))


rx1 = re.compile( u'[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]+' )
dict_words = set()
all_chars = set(u' ')
with zipfile.ZipFile( os.path.join(data_folder,'ruwords.txt.zip')) as z:
    with z.open('ruwords.txt') as rdr:
        for line in rdr:
            word = line.decode('utf-8').strip()
            if rx1.match( word) is not None:
                dict_words.add(word)
                all_chars.update(word)

nb_words = len(dict_words)
print('{} vocabulary words'.format(nb_words))

nb_chars = len(all_chars)
print('{} chars'.format(nb_chars))

max_word_len = max(map(len,dict_words))
print('max_word_len={}'.format(max_word_len))

c2i = dict([(c, i) for (i, c) in enumerate(itertools.chain([u' '], filter(lambda z: z != u' ', all_chars)))])

data = np.zeros((nb_words, max_word_len, nb_chars), dtype='bool')
for iword, word in enumerate(dict_words):
    word = rpad(word, max_word_len)
    for ichar, c in enumerate(word):
        data[iword, ichar, c2i[c]] = True

print('Storing dataset...')
with open('./data/words_4gan.npz', 'wb') as f:
    np.savez_compressed(f, data)

with open('./data/c2i_4gan.pkl', 'wb') as f:
    pickle.dump(c2i, f)



