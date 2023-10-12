# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-RNN and LSTM for stock price prediction

"""
######## Import the libraries
import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

## data preprocessing

#import inbult data set
#training_data = pd.read_csv('\training_set.csv', delimiter=',', encoding='utf-8', header=0)

training_data = pd.read_csv('training_set.csv')

training_data.head()
training_data.tail()
#we have data from november 2011 to October 2019
training_data.info()

#select 'open' feature from the dataset as training_set
training_set = training_data.iloc[:, 1:2].values
#training_set.shape

#feature scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)

#Creating a datastructure with 60 timesteps and 1 output
x_train = []
y_train = []

for i in range(60, 1257):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])

#
#x_train.info()
#y_train

#converting x_train and y_train into numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape
#1197, 60
#reshape to 3D since RNN takes 3D

x_train = x_train.reshape(1197,60,1)

#Define an object (RNN)
model = tf.keras.models.Sequential()

#Add LSTM layer
model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True, input_shape=(60,1)))

#Add dropout layer
model.add(tf.keras.layers.Dropout(0.2))

#Second LSTM layer
model.add(tf.keras.layers.LSTM(units=60, activation='relu', return_sequences=True))

#Add dropout layer
model.add(tf.keras.layers.Dropout(0.2))

#Third LSTM layer
model.add(tf.keras.layers.LSTM(units=80, activation='relu', return_sequences=True))

#Add dropout layer
model.add(tf.keras.layers.Dropout(0.2))

#Fourth LSTM layer
model.add(tf.keras.layers.LSTM(units=120, activation='relu'))

#Add dropout layer
model.add(tf.keras.layers.Dropout(0.2))

#Add output layer
model.add(tf.keras.layers.Dense(units=1))

model.summary()

#compile the model
model.compile(optimizer='adam', loss = 'mean_squared_error')
#here adam is better than rmsprop after checking
#since we are dealing with regression probelm we use mean_squared_error

model.fit(x_train, y_train, batch_size=32, epochs=10)
#32 observations for batch

#### Making predictions
#getting the real stock price of the month nov 2019
test_data = pd.read_csv('test_set.csv')
training_data2 = pd.read_csv('training_set.csv')

test_data.shape
#20,7

test_data.info()
real_stock_price = test_data.iloc[:,1:2].values

#Getting predicting stock pricess of month nov 2019
#concatination
dataset_total = pd.concat([training_data2['Open'], test_data['Open']], axis=0)
 
#stock prices of previous 60 days for each day of Nov 2019
inputs = dataset_total[len(dataset_total)-len(test_data)-60:].values
#If we dont put .values, it will be a series otherwise its an array of float64

#reshape ( converting into numpy array)
inputs = inputs.reshape(-1,1)
#reshape(-1, 1) means that you want to reshape inputs into a 2D array with one column, and the -1 is a placeholder for the number of rows.

#feature scaling
inputs = sc.transform(inputs)

#creating a test_set
x_test = []

for i in range(60,80):
    x_test.append(inputs[i-60:i,0])

#converting it to numpy array from list
x_test = np.array(x_test)

#convert it to 3D which required to process
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#getting predicted stock price
predicted_stock_price = model.predict(x_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

########  End of the script #######






    


