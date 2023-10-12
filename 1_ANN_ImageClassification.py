# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

- ANN for multiclass image classification

"""
#### step 1: import the libraries
import tensorflow as tf
#print(tf.__version__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the data sets
#fashion mnist dataset
from tensorflow.keras.datasets import fashion_mnist
#This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, 
#along with a test set of 10,000 images.

######## step2: Data Preprocessing
#Loading the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#x_train.shape, x_test.shape
#y_train.shape, y_test.shape
#np.max(x_train), np.min(x_train), np.mean(x_train)
class_names = ['0 Top/T-shirt', '1 Trouser', '2 Pullover', '3 Dress', '4 Coat', 
               '5 Sandal', '6 Shirt', '7 sneaker', '8 Bag', '9 Ankle boot']

#Data Exploration
plt.figure()
plt.imshow(x_train[1])
plt.colorbar()

class_names[y_train[1]]

#Normalise the data sets, neural network learns faster
x_train = x_train / 255.0
x_test = x_test / 255.0

plt.figure()
plt.imshow(x_train[1])
plt.colorbar()

#Flattening the data
x_train = x_train.reshape(-1, 28*28)

x_test = x_test.reshape(-1, 28*28)

#### step 3: Build the model ######

#Define the object, sequential class with sequence of dense layer
model = tf.keras.models.Sequential()
#Sequence of layers
#Adding fully connected hidden layer firts
# 1) units (Number of units) = 128
# 2) activation function as ReLU
# 3) input shape 784, because we have flatten the input data
model.add(tf.keras.layers.Dense(units=128, activation = 'relu', input_shape = (784,)))
# Adding second layer as drop out, we need to att dropout layer to avoid overfitting
model.add(tf.keras.layers.Dropout(0.3))
#regularisation technique
# Add output layer
# 1. No of output units = 10
# 2. Activation function = softmax, for multiple output (multi-classificaion)
#if it binary classification, use sigmoid function
model.add(tf.keras.layers.Dense(units=10, activation = 'softmax'))

##### Step 4: TRaining the model ####
#Compiling the model i.e. learning configeration
# 1) optimizer = adam, (minimise the loss function)
# 2) loss function = sparse_categorical_crossentropy, (acts as guide to optimiser)
# 3) metrics = sparse_categorical_accuracy, we use this for multiple ouput.
#if it is binary, accuracy is enough.
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics =['sparse_categorical_accuracy'])

model.summary()

#train the modle
model.fit(x_train, y_train, epochs = 10)
#epochs =10 means number of times we are going to train our model


#### Step 5: Model evaluation and prediction
# Model Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
# model prediction
y_pred1 = model.predict(x_test)
y_pred2 = np.argmax(model.predict(x_test), axis=-1)
#np.argmax returns the indices of the maximum value

#print(y_pred2)
#y_pred2[10], y_test[10]

#Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred2)
print(cm)
# diogonal elements show the all correct predictions

acc_score = accuracy_score(y_test, y_pred2)
print(acc_score)

### The end ####