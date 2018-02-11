# -*- coding: utf-8 -*-

from __future__ import print_function
from os import getenv
import sys
import collections
import codecs
import os
from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers.wrappers import TimeDistributed
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
import numpy as np
import itertools
import pickle
from Alphabet import Alphabet

class CharSequenceFinalizer(object):
    def __init__(self):
        pass

    def load(self, model_folder):
        self.seq_len = 64

        with open(os.path.join(model_folder,'char_predictor_alphabet.pkl'),'r') as f:
            self.alphabet = pickle.load(f)

        with open(os.path.join(model_folder,'char_predictor.arch'),'r') as f:
            self.model = model_from_json(f.read())

        self.model.load_weights( os.path.join(model_folder,'char_predictor.model') )


    # helper function to sample an index from a probability array
    # взято из https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    def sample(self, a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / (np.sum(np.exp(a))+1e-5)
        return np.argmax(np.random.multinomial(1, a, 1))


    def finalize_str(self, seed_str, nb_add_chars):
        bits_per_char = self.alphabet.nb_chars
        X_query = np.zeros( (1, self.seq_len, bits_per_char), dtype=np.bool )

        temp = 0.6

        cur_str = seed_str

        for _ in range(nb_add_chars):
            left_seq = cur_str[ -self.seq_len : ]
            X_query.fill(0)
            for i,c in enumerate(left_seq):
                X_query[0, i, self.alphabet.char2index[c]] = True

            y = self.model.predict(X_query)[0]
            y = y/y.sum()

            selected_index = self.sample(y, temp)
            selected_char = self.alphabet.index2char[selected_index]
            cur_str += selected_char

        return cur_str



finalizer = CharSequenceFinalizer()
finalizer.load('../tmp')

while True:
    seed_str = raw_input('\n>: ').strip().decode(sys.stdout.encoding)

    if len(seed_str) == 0:
        break;

    new_str = finalizer.finalize_str(seed_str, 20)
    print(u'{}'.format(new_str))

