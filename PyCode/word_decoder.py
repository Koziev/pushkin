# -*- coding: utf-8 -*-
"""
Вспомогательный класс для декодирования вектора слова в ближайший
элемент w2v с использованием L2 метрики.
"""

from __future__ import print_function, division

import pickle
import numpy as np


class W2V_Decoder:
    def find_word_l2(self, word_vector):
        """
        Поиск слова, максимально близкого к заданному вектору word_vector,
        с использованием евклидового расстояния.
        """
        deltas = self.vectors - word_vector
        l2 = np.linalg.norm(deltas, axis=-1)
        imin = np.argmin(l2)
        return self.words[imin]

    def decode_output(self, y):
        """
        Декодируем выходной тензор автоэнкодера, получаем читабельные
        предложения в том же порядке, как входные.
        
        Возвращается список слов.
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
                if l2 > 0.1:
                    best_word = self.find_word_l2(word_vector)

                phrase_words.append(best_word)

            while len(phrase_words) > 0 and phrase_words[-1] == null_word:
                phrase_words = phrase_words[:-1]

            decoded_phrases.append(u' '.join(phrase_words))

        return decoded_phrases

    def __init__(self, w2v_path):
        # Словарь с парами слово-вектор для печати читабельных результатов
        with open(w2v_path, 'r') as f:
            word2vec = pickle.load(f)

        self.words = list(word2vec.keys())
        w2v_dim = len(word2vec[self.words[0]])
        self.vectors = np.zeros((len(self.words), w2v_dim))
        for i, word in enumerate(self.words):
            self.vectors[i, :] = word2vec[self.words[i]]
