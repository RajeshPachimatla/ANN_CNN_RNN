# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Convoluted Neural Netwwork Application
-

"""
## import libraries
import tensorflow as tf
#print(tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating an object (initialising an object)
model = tf.keras.models.Sequential()


# Adding first CCN layer
# 1) filters (kernel/feature detectors) = 64
# 2) kernal size = 3
# 3) padding = same
# 4) activation = ReLU
# 5) input shape = (32, 32, 3)

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=[32, 32, 3]))

# adding max pool layer
# 1) pool size = 2 (we are selecting 2by2 array for selecting features)
# 2) strides = 2 (units on which filter is moving)
# 3) padding = valid 

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

#Adding second CNN layer and max pool layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

#Adding flattening layer
model.add(tf.keras.layers.Flatten())

#Adding the dropount layer
model.add(tf.keras.layers.Dropout(0.4))

#Adding fully connected layer i.e. ANN
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Adding output layer 
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#compiling the model i.e. configure the process
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting CNN To images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

training_set = datagen.flow_from_directory('training_set', target_size=(32,32),classes = ['dogs','cats'], class_mode='binary',batch_size=20)

testing_set = datagen.flow_from_directory('test_set', target_size=(32,32),classes = ['dogs','cats'], class_mode='binary',batch_size=20)

len(training_set)
testing_set.batch_size

model.fit_generator(generator=training_set, steps_per_epoch=401, epochs=3, validation_data=testing_set, validation_steps=102)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('D:/Rajesh P/ANN_simplilearn/Neural Network/single_prediction/cat_or_dog_2.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1 :
    prediction = 'dog'
else:
    predicton = 'cat'
    
print(prediction)

### End of the script ####