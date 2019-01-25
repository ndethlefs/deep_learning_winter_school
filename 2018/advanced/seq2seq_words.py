'''Trains a recurrent deep neural net on to generate image descriptions from their input patterns.

Currently trains an LSTM, but RNN and GRU also work. 

Input sequences are padded with zeros to be of equal size (facilitates training). 

'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, RepeatVector, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU
from bleu_score import *
import time
import sys


max_features = 20000 # needed for embeddings layer
max_len = 0  # depends on the length of input vectors.
batch_size = 32
epochs = 50
layers = 3
hidden_size = 128
input_file = './GRE3D7-1.0/GRE3D7-descriptions.csv'
output_file = input_file.split('.txt')[0] + '-weights.h5'

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
    highlight = '\033[94m'    

print('Started at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))

text = 'mask_zeros '

X = []
Y = []
scenes = []

for line in open(input_file, 'r'):
	l = line.split(",")
	scene = l[1] 
	scenes.append(scene)
	pattern = l[6]
	if len(pattern.split()) > max_len:
		max_len = len(pattern.split())
	text = text + pattern + ' '
	X.append(pattern.split())
	description = l[4]
	text = text + description + ' '	
	Y.append(description.split())	

X_set = np.asarray(X[1:])
Y_set = np.asarray(Y[1:])
	
print('Max len of input sequence: ', max_len, '(all sequences will be padded to this).')
start_time = time.time()
chars = set(text.split())
print('Total number of input symbols:', len(chars))

chars1 = list(chars)
chars1.insert(0, 'mask_zeros')	
print('chars', chars1)
#sys.exit(0)


# Get mapping dicts for input representations.
char_indices = dict((c, i) for i, c in enumerate(chars1))
indices_char = dict((i, c) for i, c in enumerate(chars1))    


print('Vectorisation...')
X = np.zeros((len(X_set), max_len, len(chars1)), dtype=np.bool)
Y = np.zeros((len(Y_set), max_len, len(chars1)), dtype=np.bool)


# encode X into a boolean matrix
for i, item in enumerate(X_set):
	X1 = np.zeros((max_len, len(chars1)))
	for j, c in enumerate(item):
		X1[j, char_indices[c]] = 1
	X[i] = X1

# encode Y into a boolean matrix
for i, item in enumerate(Y_set):
	Y1 = np.zeros((max_len, len(chars1)))
	for j, c in enumerate(item):
		Y1[j, char_indices[c]] = 1
	Y[i] = Y1	
	
	
def decode(X, calc_argmax=True):
	if calc_argmax:
		X = X.argmax(axis=-1)
	return ' '.join(indices_char[x] for x in X)

	

# Explicitly set apart 20% for validation data that we never train over.
split_at = len(X) - len(X) // 20
(X_train, X_val) = X[:split_at], X[split_at:]
(Y_train, Y_val) = Y[:split_at], Y[split_at:]
                                                                                                                                                                                                                  
print(len(X_train), 'train sequences with shape: ', X_train.shape)
print(len(X_val), 'test sequences with shape: ', X_val.shape)

# Pad sequences to be of equal length for training (adds zeros as padding values).
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_val = sequence.pad_sequences(X_val, maxlen=max_len)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_val.shape)
Y_train = sequence.pad_sequences(Y_train, maxlen=max_len)
Y_val = sequence.pad_sequences(Y_val, maxlen=max_len)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_val.shape)


print('Build model...')
'''
model = Sequential()
model.add(Embedding(max_features, hidden_size, input_length=max_len, dropout=0.2))
model.add(LSTM(hidden_size, dropout_W=0.2, dropout_U=0.2))  # can replace LSTM by RNN or GRU
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
model = Sequential()
model.add(LSTM(hidden_size, input_shape=(max_len, len(chars1))))
model.add(RepeatVector(max_len))
for _ in range(layers):
    model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(len(chars1))))
model.add(Activation('softmax'))


model.summary()

# can use different optimisers and optimiser configurations.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
# save a picture of the model architecture.              
#plot(model, to_file='./model.png', show_shapes=True)

print("--- %s seconds ---" % (time.time() - start_time))
print('Started training at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))

for iteration in range(1, epochs):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
              validation_data=(X_val, Y_val))
    score, acc = model.evaluate(X_val, Y_val, batch_size=batch_size)
                            
                            
    # Select 10 samples from the validation set at random to visualise and inspect. 
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], Y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = decode(rowX[0])       
        correct = decode(rowy[0])
        guess = decode(preds[0], calc_argmax=False)
        print()
        print('Input vector: ', colors.highlight, q.replace("mask_zeros", ""), colors.close)
        print('Correct label: ', colors.highlight, correct.replace("mask_zeros", ""), colors.close)
        print('Predicted label: ' + str(guess) + colors.ok + ' (good)' + colors.close if correct == guess else 'Predicted label: ' + str(guess) + colors.fail + ' (bad)' + colors.close)
        guess_bleu = guess.split(" mask_zeros")[0].split()
        ref_bleu = correct.split(" mask_zeros")[0].split()        
        print("BLEU", 2, "score:", getBleu(guess_bleu, [ref_bleu], [0.25, 0.25]))        
        print("BLEU", 3, "score:", getBleu(guess_bleu, [ref_bleu], [0.25, 0.25, 0.25]))
        print("BLEU", 4, "score:", getBleu(guess_bleu, [ref_bleu], [0.25, 0.25, 0.25, 0.25]))                
        print('---')
        json_string = model.to_json()
        model.save_weights(output_file, overwrite=True)                             
                          
    print('Test score:', score)
    print('Test accuracy:', acc)

print('Finished at:', (time.strftime("%d/%m/%Y")), (time.strftime("%H:%M:%S")))
