from sonnets import X, seq_length, characters,int_to_vocab, model
from keras.models import load_model
import numpy as np
from keras.models import Sequential


model = load_model('/Users/nimishmishra/Desktop/Sonnets22.model')
print('Lets go')
string_mapped = X[99]

full_string = [int_to_vocab[value] for value in string_mapped]
# generating characters
for i in range(1000):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))
    pred_index = np.argmax(model.predict(x, verbose=0))
    full_string.append(int_to_vocab[pred_index])
    string_mapped.append(pred_index)
    # Let's remove the first character. Logic flows like:
    # Pick the first 100 characters(0-99) and predict the next character.
    # Then pick the (1-100) characters and predict the next
    string_mapped = string_mapped[1:len(string_mapped)]

txt=""
for char in full_string:
    txt = txt+char

print(txt)
