# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-RNN 
-IMDB review classification

"""
#### Import the libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data preprocessing
#importing the libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

#loading the dataset
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=10000)
#we are taking most recent 10000 words
x_train.shape
#no fixed length of arrays

#apply padding to fix the lenghth
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

#check the shape
x_train.shape

#### Building the model
#define an object (initialising RNN)
model = tf.keras.models.Sequential()

#Adding Embedding layer, matrix multiplication layer to tranform the words to embedding
#it is added to compress input features into smaller ones 
#it turns positive integers to dense vectors of fixed size
model.add(tf.keras.layers.Embedding(input_dim=20000, output_dim=128, input_shape=(100,)))
#unique words are 20000

#add lstm layers, used to understand the relation between different elements of sequence
#i.e ralation between words and reviews
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))

#output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

#compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#### TRaining the model
history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data = (x_test, y_test))

#predictions
y_pred = model.predict(x_test)
y_pred = np.round(y_pred,0)

print(y_pred[15], y_test[15])

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_train, y_pred)
print(cm)

acc_score = accuracy_score(y_train, y_pred)
print(acc_score)

#lets see the learning curve

def learning_curve(history, epochs):
    #training vs validation accuracy
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Epoch')
    plt.legend(['Train', 'val'], loc = 'upper left')
    plt.show()
    
    #training vs validation accuracy
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.legend(['Train', 'val'], loc = 'upper left')
    plt.show()


learning_curve(history, 5)

#model is not overfitting 

#### End of the script #####