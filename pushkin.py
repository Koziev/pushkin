# -*- coding: utf-8 -*- 

''' Character Language Model с использованием RNN (LSTM) библиотеки keras

 Входной корпус - "Евгений Онегин" с удаленными номерами глав и т.п.
 Файл с корпусом лежит рядом с исходником.

 Модель учится предсказывать следующий символ в предложении.
 Корпус разбиваем на строфы. Каждая строфа превращается в одну входную
 цепочку, символ \n задает границы строк в строфе.
 
 В режиме генерации модель дает вероятности для каждого из символов
 в следующей позиции.
 
 (c) kelijah 2016
''' 

import os
import numpy
import random
import copy
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import keras.callbacks

# Путь к текстовому файлу, на котором модель обучается.
chars_path = 'ЕвгенийОнегин.txt'

START_CHAR = u'\b'
END_CHAR = u'\t'

# спецсимвол для выравнивания длины предложений.
SENTINEL_CHAR = u'\a'


# вызывается в конце каждой эпохи и выполняет генерацию текста
class TextGenerator(keras.callbacks.Callback):

    # в этот файл будем записывать генерируемые моделью строки
    output_samples_path = 'samples.txt'

    def __init__(self,id2char,model):
        self.id2char = id2char
        self.model = model

    def on_train_begin(self, logs={}):
        self.epoch = 0
        # удалим тексты, сгенерированные в предыдущих запусках скрипта
        if os.path.isfile(self.output_samples_path):
            os.remove(self.output_samples_path)

    # helper function to sample an index from a probability array
    # взято из https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    def sample( self, a, temperature=1.0):
        a = numpy.log(a) / temperature
        a = numpy.exp(a) / numpy.sum(numpy.exp(a))
        return numpy.argmax(numpy.random.multinomial(1, a, 1))


    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        if (self.epoch%10)==0:
            with open( self.output_samples_path, 'a' ) as fsamples:
                        
                fsamples.write( u'\n' + '='*50 + '\nepoch=' + str(self.epoch) + u'\n' )
                        
                # генерируем по 4 цепочки (строфы) для нескольких температур
                for temp in [0.3, 0.6, 0.9, 1.0, 1.1]:
                    for igener in range(0,4):
                        # сделаем сэмплинг цепочки символов
                        # начинаем всегда с символа <s>
                        last_char = START_CHAR
                        self.model.reset_states();
                                
                        # буфер для накопления сгенерированной строки
                        sample_str = u''
                        sample_seq = last_char
                            
                        while len(sample_str)<300:
                
                            xlen = len(sample_seq)
                            X_gener = numpy.zeros( (1,xlen,input_size) )
                                
                            for itime,uch in enumerate( list( sample_seq ) ):
                                X_gener[0,itime,:] = char2vector[ uch ]
                            
                            # получаем результат - цепочка предсказаний, из которой нам нужен только
                            # последний вектор
                            Y_gener = self.model.predict( X_gener, batch_size=1, verbose=0 )[0,:]
                            yv = Y_gener[xlen-1,:]
    
                            selected_index = self.sample( yv, temp )
                            selected_char = id2char[selected_index]
    
                            if selected_char==END_CHAR:
                                break
                                
                            sample_str = sample_str + selected_char
                            sample_seq = sample_seq + selected_char
                            last_char = selected_char
                                    
                        print 'sample t=', temp,  ' str=', sample_str
                        fsamples.write( '\nt=' + str(temp) + '\n\n' )
                        fsamples.write( sample_str.encode('utf-8') + '\n' )







print 'Reading char sequences from ', chars_path

chars_set = set() # список встретившихся символов без повтора

# токены начала и конца цепочки, а также перевода строки добавляем
# руками, так как в явном виде их в корпусе нет
chars_set = set( [SENTINEL_CHAR,START_CHAR, u'\n', END_CHAR] )

with open( chars_path, 'r' ) as f:
    for num,line in enumerate(f):
        chars_set.update( list(line.strip().decode("utf-8")) )

nchar = len(chars_set)
print 'Number of unique chars=', len(chars_set)

# для получения символа по его индексу
id2char = dict( (i,c) for i,c in enumerate(chars_set) ) 

# преобразование символа во входной вектор 0|1's
char2vector = {}
for i,c in enumerate(chars_set):
    v = numpy.zeros(nchar)
    if c!=SENTINEL_CHAR:
        v[i] = 1
    char2vector[c] = v


features_size = 150 # кол-во элементов в RNN
batch_len = 16
NUMBER_OF_EPOCH = 500 # кол-во повторов тренировки по одному набору данных
input_size = nchar
output_size = nchar

model = Sequential()

rnn_layer = LSTM( input_dim=input_size, output_dim=features_size, activation='tanh', return_sequences=True )
model.add(rnn_layer)
model.add( Dropout(0.1))

#rnn_layer2 = LSTM( output_dim=features_size, activation='tanh', return_sequences=True )
#model.add(rnn_layer2)
#model.add( Dropout(0.1))

model.add(TimeDistributedDense(output_dim=output_size))
model.add(Activation('softmax'))
#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

print 'Compiling the model...'
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# сохраним архитектуру сети в текстовый файл
#open('model_architecture.json', 'w').write( model.to_json() )

print( 'Start training...' )

# количество накопленных в samples последовательностей
sample_count = 0

# здесь получим макс. длину последовательности символов без учета добавляемых токенов <s> и </s>
max_seq_len=0

# идем по файлу с предложениями, извлекаем сэмплы для обучения
print u'Loading', chars_path

endchars = set( u'!;.?' )
sample_buf = ''
sample_count=0
samples = []
with open( chars_path, 'r' ) as f:
    for line in f:
        charseq = line.strip().decode("utf-8")
        
        if len(charseq)>0:
            if len(sample_buf)>0:
                sample_buf = sample_buf + u'\n'
            sample_buf = sample_buf + charseq
            
            if len(sample_buf)>0 and sample_buf[-1] in endchars:
                xlen = len(sample_buf)
                if xlen>1:
                    samples.append(sample_buf)
                    sample_count = sample_count+1
                    max_seq_len = max( max_seq_len, xlen )
                sample_buf = ''

print 'sample_count=', sample_count

# из общего числа отберем треть для проверки
#n_test = int(sample_count*0.3)
n_test = 0
            
# остальное - тренировка
n_train = sample_count-n_test
            
xlen = max_seq_len+1            
            
# тензоры для входных последовательностей и выходных эталонных данных
X_train = numpy.zeros( (n_train,xlen,input_size), dtype=numpy.bool )
Y_train = numpy.zeros( (n_train,xlen,input_size), dtype=numpy.bool )

X_test = numpy.zeros( (n_test,xlen,input_size), dtype=numpy.bool )
Y_test = numpy.zeros( (n_test,xlen,input_size), dtype=numpy.bool )

itrain=0;
itest=0


print 'Vectorization...'            
# заполняем тензоры
for isample,rawseq in enumerate(samples):
    
    # слева или справа дополняем символами \a, чтобы все последовательности имели одинаковую длину
    seq = (START_CHAR + rawseq + END_CHAR).rjust(max_seq_len+2,SENTINEL_CHAR)

    is_training = True
    if itrain>=n_train:
        is_training = False
                     
    for itime in range(0,len(seq)-1):
        x = seq[itime]
        y = seq[itime+1]
                    
        if is_training:
            X_train[itrain,itime,:] = char2vector[ x ]
            Y_train[itrain,itime,:] = char2vector[ y ]
        else:
            X_test[itest,itime,:] = char2vector[ x ]
            Y_test[itest,itime,:] = char2vector[ y ]
                
    if is_training:        
        itrain = itrain+1
    else:    
        itest = itest+1

text_generator = TextGenerator(id2char,model)

print 'training sample_count=', sample_count, 'itrain=', itrain, 'itest=', itest
#acc = model.fit( X_train, Y_train, batch_size=batch_len, nb_epoch=NUMBER_OF_EPOCH, validation_data=[X_test,Y_test], callbacks=[text_generator] )
acc = model.fit( X_train, Y_train, batch_size=batch_len, nb_epoch=NUMBER_OF_EPOCH, callbacks=[text_generator] )
          
