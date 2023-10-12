# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-CNN 
-project digit recognition

"""
### import the libraries ###
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the data set, here built in dataset of digits
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[3])
#it is one
print(y_train[3])

#check the size of the data
x_train.shape
#60000 images with size 28 by 28
y_train.shape

x_test.shape
y_test.shape

print(x_train.min(), x_train.max())
print(y_train.min(), y_train.max())

#normalisation of the pixel data so that CNN will run fast
x_train = x_train/255.0
x_test = x_test/255.0

#reshape the dimension to 3D
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

input_shape = x_train[0].shape
print(input_shape)

#define an object 
model = tf.keras.models.Sequential()

#Adding first convolution layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

#Adding Second convolution layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

#Adding maxpool layer, polling extracts dominant features of the image and reduces the size of convolt features
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#adding dropout layer, to regularisation so that avoid over fitting
model.add(tf.keras.layers.Dropout(0.4))

#adding flatten layer
model.add(tf.keras.layers.Flatten())

#Add fully connected layer, i.e. ANN
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

#output layer
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()

#adam is stochastic gradeint descnet algorithm
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

#Lets train the model
history=model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)
#size of y_pred is 10000,10. -1 as the axis argument means that it will operate along the last axis of the array.
y_pred[15], y_test[15]

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

#Lets plot the learning the curve to understand whether model is overfitting or underfitting

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
#it is observed that training as well as validation accuracy is increasing with no of epochs,
#this means the model is not overfiltted.
    
######### End of the script ######    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
