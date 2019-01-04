# Downloaded the Shakespeare sonnents from Project Gutenburg. Find the file at http://www.gutenberg.org/ebooks/1041?msg=welcome_stranger

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils

sonnets = (open('Sonnets.txt.utf-8.txt')).read()

# Word embeddings

sonnets = sonnets.lower()
characters = sorted (list(set(sonnets)))
# Assigning an integer to every character that appears in this
# whole text
int_to_vocab = {key: word for key, word in enumerate(characters)}
vocab_to_int = {word: key for key, word in enumerate(characters)}


# Let's go creating the data

X = []
Y = []
length = len(sonnets)
seq_length = 100 # Hyperparameter probably

for i in range(0, length - seq_length, 1):
    sequence = sonnets[i : i + seq_length]
    label = sonnets [i + seq_length]
    X.append([vocab_to_int[char] for char in sequence])
    Y.append ( [vocab_to_int[label]])

final_X = np.reshape(X, (len(X), seq_length, 1) )
final_X = final_X / float(len(characters)) # converts to sorts of fraction
final_Y = np_utils.to_categorical(Y)


model = Sequential()
model.add(LSTM(700, input_shape=(final_X.shape[1], final_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(final_Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#model.fit(final_X, final_Y, epochs=100, batch_size=50)
