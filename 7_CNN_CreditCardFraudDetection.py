# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-CNN 
-credit card fraud detection

"""
#### Import the libraries
import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# data preprocessing

#import inbult data set
#training_data = pd.read_csv('\training_set.csv', delimiter=',', encoding='utf-8', header=0)

dataset_1 = pd.read_csv('creditcard.csv')

dataset_1.head()
dataset_1.tail()
#we have data from november 2011 to October 2019
dataset_1.info()

#Checking the null values
dataset_1.isnull().sum()
#no null values

#observations in each class
dataset_1['Class'].value_counts()
#data is highly unbalanced

#balance the dataset
fraud = dataset_1[dataset_1['Class']==1]
non_fraud = dataset_1[dataset_1['Class']==0]

fraud.shape, non_fraud.shape

#random selection of sample
non_fraud_t = non_fraud.sample(n=492)
non_fraud_t.shape

# merge the data set
dataset = fraud.append(non_fraud_t, ignore_index = True)

#observations in each class
dataset['Class'].value_counts()

#matrix of features
x= dataset.drop(labels=['Class'], axis=1)

#dependent variable
y = dataset['Class']
x.shape, y.shape

### splitting the dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)

## feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
#since there is only one column in 'y', better to write it in numpy array form

#reshape the dataset
x_train = x_train.reshape(787, 30, 1)
x_test = x_test.reshape(197, 30, 1)
#since, takes CNN 3D size only

### Building the model
model = tf.keras.models.Sequential()

# add first CNN layer
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu',input_shape=(30,1)))

#batch normalization
model.add(tf.keras.layers.BatchNormalization())

#max pool and droput layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))

# add second CNN layer
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))

#batch normalization
model.add(tf.keras.layers.BatchNormalization())

#max pool and droput layer
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

#flatten layers
model.add(tf.keras.layers.Flatten())

#first Dense layer
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
#dropout layer
model.add(tf.keras.layers.Dropout(0.3))

#output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

#compile the model to configure the learning process
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

### Model fitting /training
history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))

#model prediction
y_pred = model.predict(x_test)
y_pred = np.round(y_pred,0)

print(y_pred[11], y_test[11])

### Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc_score = accuracy_score(y_test, y_pred)
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


learning_curve(history, 25)

#model is neither overfitted nor undefitted
############## End of the script #####















