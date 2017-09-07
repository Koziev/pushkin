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
import numpy as np
import itertools
import pickle
from Alphabet import Alphabet





class CharLevelEstimator(object):
    def __init__(self):
        self.sentinel_char = u'\b'
        self.seq_len = 64

    def build_model(self):
        assert self.seq_len>1
        assert len(self.alphabet.alphabet)>0
        bits_per_char = self.alphabet.nb_chars
        rnn_size = bits_per_char
        model = Sequential()
        model.add( Masking( mask_value=0, input_shape=(self.seq_len, bits_per_char), name='input_layer' ) )
        model.add( recurrent.LSTM( rnn_size, input_shape=(self.seq_len, bits_per_char), return_sequences=False ) )
        model.add( Dense( units=rnn_size, activation='sigmoid') )
        model.add( Dense( units=bits_per_char, activation='softmax', name='output_layer') )
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def generate_rows(self, corpus_path, batch_size, for_validation):
        bits_per_char = self.alphabet.nb_chars

        X_batch = np.zeros( (batch_size, self.seq_len, bits_per_char), dtype=np.bool )
        y_batch = np.zeros( (batch_size, bits_per_char), dtype=np.bool )

        with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
            lines_count = 0
            required_remainder = 1 if for_validation else 0
            batch_index = 0
            for line in rdr:
                lines_count += 1
                if (lines_count%2)==required_remainder:
                    l = len(line)
                    for right_pos in range(1, l):
                        x_chars = line[max(0, right_pos-self.seq_len) : right_pos]
                        y_char = line[right_pos]

                        for i,x in enumerate(x_chars):
                            if x in self.alphabet.char2index:
                                X_batch[batch_index, i, self.alphabet.char2index[x]] = True

                        if y_char in self.alphabet.char2index:
                            y_batch[batch_index, self.alphabet.char2index[y_char]] = True

                        batch_index += 1

                        if batch_index==batch_size:
                            yield ({'input_layer_input': X_batch}, {'output_layer': y_batch})

                            # очищаем матрицы порции для новой порции
                            X_batch.fill(0)
                            y_batch.fill(0)
                            batch_index = 0


    def fit(self, corpus_pass, model_folder):

        # Заранее построим список всех возможных символов - алфавит.
        # Для этого проанализируем достаточно большой кусок тренировочного корпуса
        # с надеждой на то, что появление нового символа за пределами этого куска - маловероятно.
        self.alphabet = Alphabet()
        self.alphabet.fit(corpus_path)
        print('{} chars'.format(self.alphabet.nb_chars))
        self.model = self.build_model()

        with open(os.path.join(model_folder,'char_predictor_alphabet.pkl'),'w') as f:
            pickle.dump( self.alphabet, f )

        with open(os.path.join(model_folder,'char_predictor.arch'),'w') as f:
            f.write(self.model.to_json())

        weights_filename = os.path.join(model_folder, 'char_predictor.model' )

        print('Train...')

        # Генерируем батчи из обучающего набора.
        # Перед каждой эпохой тасуем обучающие N-граммы.
        nb_patterns = 1000000
        batch_size = 200

        model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_acc', verbose=1,
                                           save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')

        self.model.fit_generator(generator=self.generate_rows(corpus_path, batch_size, for_validation=False),
                            steps_per_epoch=int(nb_patterns / batch_size),
                            epochs=100,
                            verbose=1,
                            callbacks=[model_checkpoint, early_stopping],
                            validation_data=self.generate_rows(corpus_path, batch_size, for_validation=True),
                            validation_steps=int(nb_patterns / batch_size),
                            )
        return



#corpus_path = r'f:\Corpus\Raw\ru\text_blocks.txt'
corpus_path = r'/home/eek/Corpus/Raw/ru/text_blocks.txt'
model_folder = '.'

char_model = CharLevelEstimator()
char_model.fit(corpus_path, model_folder)




