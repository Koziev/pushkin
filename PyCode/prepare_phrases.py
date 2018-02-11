# -*- coding: utf-8 -*-
"""
Подготовка списка качественных предложений для тренировки генеративных NLP моделей.
Используются различные корпуса, наполненные вручную или тщательно промодерированные.
Результат работы - текстовый файл, каждое предложений на отдельной строке.
"""

from __future__ import print_function, division

import codecs
import os
import pickle
import numpy as np
import zipfile
import re
import glob

from utils.tokenizer import Tokenizer

data_folder = '../data'

MAX_SENT_LEN = 4

if __name__ == '__main__':

    # список словарных слов.
    # они нужны, чтобы не рассматривать предложения, содержащие
    # искаженную лексику и т.д.
    rx1 = re.compile(u'[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]+')
    dict_words = set()
    with zipfile.ZipFile(os.path.join(data_folder, 'ruwords.txt.zip')) as z:
        with z.open('ruwords.txt') as rdr:
            for line in rdr:
                word = line.decode('utf-8').strip()
                if rx1.match(word) is not None:
                    dict_words.add(word)

    uniq_phrases = set()
    phrases = []
    all_words = set()

    tokenizer = Tokenizer()

    if True:
        for corpus_filepath in glob.glob(os.path.join(data_folder, r'e:\MVoice\lem\dictionary.src\corpus\syntax-ru.*.xml')):
            with codecs.open(corpus_filepath, 'r', 'utf-8') as rdr:
                for line in rdr:
                    if line.startswith(u'<text>'):
                        line = line.replace(u'<text>', u'').replace(u'</text>', u'').strip()
                        if line not in uniq_phrases:
                            uniq_phrases.add(line)

                            words = tokenizer.tokenize(line)
                            if len(words) <= MAX_SENT_LEN:
                                all_words_known = True
                                for word in words:
                                    if word not in dict_words:
                                        all_words_known = False
                                        break

                                if all_words_known:
                                    phrases.append(words)
                                    all_words.update(words)
                                    if len(phrases) >= 1000000:
                                        break

        with codecs.open(r'e:\polygon\paraphrasing\data\paraphrases.txt', 'r', 'utf-8') as rdr:
            for line in rdr:
                if line not in uniq_phrases:
                    uniq_phrases.add(line)

                    words = tokenizer.tokenize(line.strip())
                    if len(words) <= MAX_SENT_LEN:
                        all_words_known = True
                        for word in words:
                            if word not in dict_words:
                                all_words_known = False
                                break

                        if all_words_known:
                            phrases.append(words)
                            all_words.update(words)
                            if len(phrases) >= 1000000:
                                break
    else:
        # Составляем список ответов. Предполагается что получившийся
        # датасет будет использован для тренировки генератора ответов чат-бота.
        corpora = ['e:/polygon/paraphrasing/data/qa.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer_names4.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer_neg4.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer_neg4_1s.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer_neg4_2s.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer_neg5.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer_neg5_1s.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer_neg5_2s.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer6.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer5.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer4.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer4_1s.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer4_2s.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer5_1s.txt',
                   'e:/polygon/paraphrasing/data/premise_question_answer5_2s.txt',
                   ]

        for corpus in corpora:
            print(u'Start processing {}...'.format(corpus))
            with codecs.open(corpus, 'r', 'utf-8') as rdr:
                for line in rdr:
                    if line not in uniq_phrases:
                        uniq_phrases.add(line)
                        if line.startswith(u'A:'):
                            line = line.replace(u'A:', u'').strip()
                            words = tokenizer.tokenize(line)
                            if len(words) <= MAX_SENT_LEN:
                                all_words_known = True
                                for word in words:
                                    if word not in dict_words:
                                        all_words_known = False
                                        break

                                if all_words_known:
                                    phrases.append(words)
                                    all_words.update(words)
                                    if len(phrases) >= 1000000:
                                        break

    nb_phrases = len(phrases)
    print('Storing {} phrases...'.format(nb_phrases))
    with codecs.open('../data/phrases.txt', 'w', 'utf-8') as wrt:
        for phrase in phrases:
            wrt.write(u'{}\n'.format(u' '.join(phrase)))
