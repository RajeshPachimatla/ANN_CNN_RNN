# -*- coding: utf-8 -*-
"""

@author: USER

- ANN for binary data classification

"""
### 1: import the libraries
import tensorflow as tf
#print(tf.__version__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### 2: Data importing and preparation
data_set = pd.read_csv('Churn_Modelling.csv')
data_set.head(5)

data_set.corr()

#x1 = data_set.drop(labels = ['RowNumber','CustomerId','Surname','Exited'], axis = 1)

x = data_set.drop(labels = ['RowNumber','CustomerId','Surname','Exited'], axis = 1)
#matrix of features, indepedent variables
y = data_set['Exited']
#dependent variables

x.describe(include = 'all')
#y.head()

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
Label_1 = LabelEncoder()
#LabelEncoder() is instance method; if it is like this LabelEncoder is class method

x['Geography'] = Label_1.fit_transform(x['Geography'])

x.head()

Label_2 = LabelEncoder()
#LabelEncoder() is instance method; if it is like this LabelEncoder is class method

x['Gender'] = Label_2.fit_transform(x['Gender'])

x.head()

#Avoid the dummy variable trap
x = pd.get_dummies(x, drop_first = True, columns =['Geography'])
x.head(8)

 # splitiing the dataset into traina dn test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#Here x_train is numpy array
#x_test and y_test are vector. to convert them to numpy array use x_test.to_numpy() and y_test.to_numpy()

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

### 3. Building the model
#Creating an object (or initialisation of neural networking)
#since we want to build fully connceted neural network, use sequential()
model = tf.keras.models.Sequential()
#Adding input layer and hidden layer
# 1) units = 6 ((trick input dimension 11 + output dimension 1)/2 = 6)
# 2) activation function = Relu, brings non-lineary
# 3) input dimension = 11
model.add(tf.keras.layers.Dense(units = 6, activation = 'relu', input_dim = 11))
#Adding second hidden layer
model.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#Adding out put layer
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# compile the model, configuring the learning process
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = 'accuracy')
model.summary()

### 4: Training the model
#fit model for training
model.fit(x_train, y_train.to_numpy(), batch_size=10, epochs = 20)
#data is modeled 20 times in the batches of 10

### 5. Model evaluation and prediction
#evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test.to_numpy())
print(test_loss, test_accuracy)
y_pred1 = model.predict(x_test)
y_pred2 = np.argmax(model.predict(x_test), axis=-1)
y_pred2
y_test = y_test.to_numpy()
print(y_pred2[11], y_test[11])
#Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred2)
print(cm)

acc_2 = accuracy_score(y_test, y_pred2)
print(acc_2)

########  End of the script ###























