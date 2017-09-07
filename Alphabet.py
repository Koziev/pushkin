# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs


class Alphabet(object):
    def __init__(self):
        pass

    def fit(self, corpus_path):
        self.alphabet = set()
        with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
            line_count = 0
            for line in rdr:
                self.alphabet.update(line)
                line_count += 1
                if line_count>1000000:
                    break

        self.nb_chars = len(self.alphabet)
        self.char2index = dict([(c,i) for (i,c) in enumerate(self.alphabet)])
        self.index2char = dict([(i,c) for (c,i) in self.char2index.iteritems()])

