# -*- coding: utf-8 -*-
"""
Подготовка датасета для экспериментов с автоэнкодерами и вариационными
автоэнкодерами для русскогоязычных предложений.
На входе используется файл со списком предложений.
На выходе - два файла с векторизованными предложениями и словарь
с соответствиями слов и векторов.
"""
from __future__ import print_function, division

import gensim
import codecs
import os
import pickle
import numpy as np
from future.utils import iteritems


data_folder = '../data'

#w2v_path = r'f:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt'
w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=48.txt'

corpus_path = '../data/phrases.txt'

MAX_SENT_LEN = 6




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


if __name__ == '__main__':

    print('Loading the w2v model {}'.format(w2v_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))

    phrases = []
    all_words = set()

    with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            words = line.strip().split()
            if len(words) <= MAX_SENT_LEN:
                all_words_known = True
                for word in words:
                    if word not in w2v:
                        all_words_known = False
                        break

                if all_words_known:
                    phrases.append(words)
                    all_words.update(words)
                    #if len(phrases) >= 10000:
                    #    break

    nb_phrases = len(phrases)

    max_sent_len = max(map(len, phrases))
    print('max_sent_len={}'.format(max_sent_len))

    vtexts = np.zeros((nb_phrases, max_sent_len, w2v_dims))
    for iphrase, phrase_words in enumerate(phrases):
        for iword, word in enumerate(phrase_words):
            vtexts[iphrase, iword, :] = w2v[word]

    # выполним нормализацию векторов
    vmin = np.amin(vtexts)
    vmax = np.amax(vtexts)
    scale = 1.0 / max(vmax, -vmin)  # приводим к диапазону -1..+1
    vtexts *= scale
    print('scale={}'.format(scale))

    word2vec = dict([(word, w2v[word]*scale) for word in all_words])

    vmin = np.amin(vtexts)
    vmax = np.amax(vtexts)
    print('After scaling: vmin={} vmax={}'.format(vmin, vmax))


    # тестируем векторизацию
    v2w = [(v, w) for w, v in iteritems(word2vec)]

    X_probe = vtexts[0:10]
    probe_phrases = decode_output(X_probe, v2w)
    for phrase in probe_phrases:
        print(u'{}'.format(phrase))


    print('Storing dataset...')
    with open('../data/vtexts.npz', 'wb') as f:
        np.savez_compressed(f, vtexts)

    with open('../data/word2vec.pkl', 'wb') as f:
        pickle.dump(word2vec, f)



