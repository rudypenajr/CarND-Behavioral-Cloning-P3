

from import_data import ImportData
from nvidia_model import build

import csv
import os
import random

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import tensorflow as tf

from sklearn.model_selection import train_test_split


# Udacity Assistance Data
udacity_path = 'udacity_data/driving_log.csv'

# My Data
data_path = 'trained_data/left_track/driving_log.csv'

ch, row, col = 3, 64, 64


###############
# Read Data
###############
samples = ImportData()
samples.load_csv(udacity_path)
samples.load_csv(data_path)
###############
# End - Read Data
###############


###############
# Data Modifications:
# Take out MOST of the 0 steering angle images
###############
samples.remove_partial_steering_angles();
train_samples, validation_samples = samples.train_test_split()
###############
# End - Data Modifications:
###############


###############
# Data Pre-processing:
# Shift, Flip, Change Brightness, Crop
###############
# Compile and Train Model using the Generator
train_generator = samples.train_generator()
# print("train_generator", train_generator)
valid_generator = samples.valid_generator()
# print("valid_generator", valid_generator)
###############
# End - Generators
##############




###############
# Keras Model
##############
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mse')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=valid_generator, nb_epoch=5, verbose=1)
###############
# End - Keras Model
##############

###############
# Save Model and Weights
###############
model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
K.clear_session()