# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-CNN 
-project breast cancer prediction

"""
#### Import the libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Data preprocessing

from sklearn import datasets, metrics
#import dataset
cancer = datasets.load_breast_cancer()

print(cancer.DESCR)
#it is data set not dataframe

#matrix of features
x = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
x.head()

#selecting dependent variable
y = cancer.target
print(y)

cancer.target_names
x.shape, y.shape
#(569, 30), (30,)

#splitting the dataset in train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

print(x_train.shape, x_test.shape)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#reshape
x_train = x_train.reshape(455,30,1)
x_test = x_test.reshape(114,30,1)

x_train.shape
x_test.shape

#Define an object
model = tf.keras.models.Sequential()

#Add first CNN layer
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(30,1)))

#Add batch normalization, allows each of layer of network to be independent of other layers
model.add(tf.keras.layers.BatchNormalization())

#Adding dropout layer
model.add(tf.keras.layers.Dropout(0.2))

#Add second CNN layer
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))

#Add batch normalization, allows each of layer of network to be independent of other layers
model.add(tf.keras.layers.BatchNormalization())

#Adding dropout layer
model.add(tf.keras.layers.Dropout(0.4))

#Add flatten layer
model.add(tf.keras.layers.Flatten())

#Add Dense layer / fully connected layer / ANN
model.add(tf.keras.layers.Dense(units=64, activation='relu'))

#Add output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

#compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.00005)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

### Model prediction

y_pred=model.predict(x_test)

y_pred.shape

y_pred = np.round(y_pred,0)

y_pred.shape
print(y_pred[12], y_test[12])

cancer.target_names

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc_score=accuracy_score(y_test, y_pred)
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


learning_curve(history, 50)

#model is not overfitting 


############ End of the script ######