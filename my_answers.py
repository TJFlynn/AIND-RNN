import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    #staring and stopping indices
    start = 0
    stop = window_size
    
    #loops over the series and breaks when the start or stop index goes out of range
    #relative to the length of the series input
    while True:
        if start >= len(series):
            break
        elif stop >= len(series):
            break
        else:
            X.append(series[start:stop])
            y.append(series[stop])
        start += 1
        stop += 1
        

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    characters = sorted(list(set(text)))
    
    punctuation = ['!', ',', '.', ':', ';', '?']
    
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

    symbols_to_keep = punctuation + letters
    
    for char in characters:
        if char not in symbols_to_keep:
            text = text.replace(char, ' ')
   
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    #staring and stopping indices
    start = 0
    stop = window_size
    
    #loops over the text and breaks when the start or stop index goes out of range
    #relative to the length of the text input
    while True:
        if start >= len(text):
            break
        elif stop >= len(text):
            break
        else:
            inputs.append(text[start:stop])
            outputs.append(text[stop])
        start += step_size
        stop += step_size
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
