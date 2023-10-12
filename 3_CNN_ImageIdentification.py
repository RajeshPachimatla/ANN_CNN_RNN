# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Convoluted Neural Network application
-

"""
### 1. Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# Importing the dataset
from tensorflow.keras.datasets import cifar10

#### 2: Data preprocessing
# Loading the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['0: airplane', '1: automobile', '2: bird', '3: cat', '4: deer', '5: dog', '6: frog', '7: horse', '8: ship', '9: truck']

print(class_names) 

print(x_train.max(), x_train.min(), x_train.mean())
#255, 0, 120.7

print(y_train.max(), y_train.min())
#10 classe

#Noramlizing the images
x_train = x_train/255.0
x_test = x_test/255.0

print(x_train.shape, x_test.shape)

plt.imshow(x_train[0])
print(y_train[0])

#Building the CNN model
#Define the object as model
model = tf.keras.models.Sequential()

#Adding the first CNN layer
# 1) filters (kernel) = 32
# 2) kernel size = 3
# 3) padding = same
# 4) activation function = relu
# 5) input shape = (32,32,3)

#SAME padding Image ==> filter ==> output
#adding the column or row of zeros called padding
#VALID padding (no addinf zero column/row), we may loose information in last column/row

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size = 3, padding ='same', activation = 'relu', input_shape=[32,32,3]))

# Adding second layer and maxpool layer
# 1) filters (kernel) = 32
# 2) kernel size = 3
# 3) padding = same
# 4) activation function = relu

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size = 3, padding ='same', activation = 'relu'))

#maxpool layer parameters,
# 1) pool size = 2
# 2) strides = 2
# 3) padding = valid

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding third layer 
# 1) filters (kernel) = 64
# 2) kernel size = 3
# 3) padding = same
# 4) activation function = relu

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size = 3, padding ='same', activation = 'relu'))

# Adding fourth layer and maxpool layer
# 1) filters (kernel) = 64
# 2) kernel size = 3
# 3) padding = same
# 4) activation function = relu

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size = 3, padding ='same', activation = 'relu'))

#2nd maxpool layer parameters,
# 1) pool size = 2
# 2) strides = 2
# 3) padding = valid

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#Adding dropout layers is like a regularization technique
model.add(tf.keras.layers.Dropout(0.4))

#Adding flattening layer, converting array into vector
model.add(tf.keras.layers.Flatten())


#Adding first Dense layer
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Adding second dense layer (output)
model.add(tf.keras.layers.Dense(units=10, activation ='softmax'))

model.summary()

#Compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam',metrics=['sparse_categorical_accuracy'])
#for binary classification we use accuracy, but multiclassification we use sparce_categorical_accuracy

#Fitting the model
model.fit(x_train, y_train, batch_size=10, epochs=1)

#Evaluate the model performance
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print('Test accuracy is {}'.format(test_accuracy))

#Predictions
y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=-1)

print(y_pred)

print(y_pred[12], y_test[12])

#Confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc_cm = accuracy_score(y_test, y_pred)
#accuracy based on confusion matrix 
print(acc_cm)


#### End of the script ####