# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-RNN and LSTM for image classification

"""
### Import the libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Data preprocessing

from tensorflow.keras.datasets import mnist

# loading the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
    
print(x_train.min(), x_train.max())
print(y_train.min(), y_train.max())
#  0 to 9 classes

### normalise the dataset
x_train = x_train/255
x_test = x_test/255
print(x_train.min(), x_train.max())

plt.imshow(x_train[5])
print(y_train[5])

### Building the model
#initialising RNN
model = tf.keras.models.Sequential()

#First LSTM layer, it is used to understand the relation btw elements of sequence
model.add(tf.keras.layers.LSTM(units=128, activation='relu', return_sequences=True, input_shape=(28,28)))

## add droput layer
model.add(tf.keras.layers.Dropout(0.2))

#Second LSTM layer, it is used to understand the relation btw elements of sequence,
#here we removing return_sequence=True, since after this we are not going to add another LSTM layer
model.add(tf.keras.layers.LSTM(units=128, activation='relu'))
## add droput layer
model.add(tf.keras.layers.Dropout(0.2))

## Adding fully connected layer (first dense layer)
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

# Add output layer
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


### Training the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)

print(y_pred[3], y_test[3])

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

#lets see the learning curve

def learning_curve(history, epochs):
    #training vs validation accuracy
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['sparse_categorical_accuracy'])
    plt.plot(epoch_range, history.history['val_sparse_categorical_accuracy'])
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

############## end of the script ############