# https://github.com/ahmadchatha/CarND-Behavioral-Cloning-P3/blob/master/model.py

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
# samples = []
# with open(udacity_path) as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
#
# with open(data_path) as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         samples.append(line)
# print('samples: ', samples)
###############
# End - Read Data
###############


###############
# Data Modifications:
# Take out MOST of the 0 steering angle images
###############
samples.remove_partial_steering_angles();
# print('total images: ' + str(len(samples)))
# count = 0
# new_sample = []
# for sample in samples:
#     print('sample 3: ', sample[3])
#     exit()
# 	center_angle = float(sample[3])
# 	if center_angle == 0.0:
# 		count = count + 1
# 		if np.random.uniform() < 0.2:
# 			new_sample.append(sample)
# 	else:
# 		new_sample.append(sample)
#
# samples = new_sample
# print('0 count: ' + str(count))
# print('After images: ' + str(len(samples)))

train_samples, validation_samples = samples.train_test_split()
###############
# End - Data Modifications:
###############


###############
# Data Pre-processing:
# Shift, Flip, Change Brightness, Crop
###############
# def pre_process(image, angle):
#     image, angle = shift_img(image, angle)
#     image, angle = flip_img(image, angle)
#     image = brightness_img(image)
#     img = image[60:136,0:image.shape[1],:]
#     return cv2.resize(img, (64, 64), cv2.INTER_AREA),angle

# def flip_img(image, steering):
# 	""" randomly flip image to gain right turn data (track1 is biaed in left turn)
# 		source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py#L89"""
# 	flip_image = image.copy()
# 	flip_steering = steering
# 	num = np.random.randint(2)
# 	if num == 0:
# 	    flip_image, flip_steering = cv2.flip(image, 1), -steering
# 	return flip_image, flip_steering

# def brightness_img(image):
# 	"""
# 	randomly change brightness by converting Y value
# 	source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py
# 	"""
# 	br_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# 	coin = np.random.randint(2)
# 	if coin == 0:
# 	    random_bright = 0.2 + np.random.uniform(0.2, 0.6)
# 	    br_img[:, :, 2] = br_img[:, :, 2] * random_bright
# 	br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
# 	return br_img

# def shift_img(image, steer):
# 	""" shift image randomly
# 		source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py """
# 	max_shift = 55
# 	max_ang = 0.14  # ang_per_pixel = 0.0025
#
# 	rows, cols, _ = image.shape
#
# 	random_x = np.random.randint(-max_shift, max_shift + 1)
# 	dst_steer = steer + (random_x / max_shift) * max_ang
# 	if abs(dst_steer) > 1:
# 	    dst_steer = -1 if (dst_steer < 0) else 1
#
# 	mat = np.float32([[1, 0, random_x], [0, 1, 0]])
# 	dst_img = cv2.warpAffine(image, mat, (cols, rows))
# 	return dst_img, dst_steer

# # function to grab the actual image
# def grab_image(sample):
# 	rand = np.random.randint(3)
# 	name = sample[rand].strip()
# 	# special case for udacity data
# 	if sample[rand].split('/')[0].strip() == 'IMG':
# 		name = './data/IMG/'+sample[rand].split('/')[-1]
# 	angle = float(sample[3])
# 	if rand == 1:
# 		angle = angle + 0.22
# 	if rand == 2:
# 		angle = angle - 0.22
# 	return name,angle
###############
# End - Data Pre-processing
###############


###############
# Generators
###############
# # generator for train images
# def generator(samples, batch_size=32):
# 	num_samples = len(samples)
# 	while 1: # Loop forever so the generator never terminates
# 		shuffle(samples)
# 		for offset in range(0, num_samples, batch_size):
# 			batch_samples = samples[offset:offset+batch_size]
#
# 			images = []
# 			angles = []
# 			for batch_sample in batch_samples:
# 				name,angle = grab_image(batch_sample)
# 				#print(name,angle)
# 				image = cv2.imread(name)
# 				#print(image.shape)
# 				image,angle = pre_process(image,angle)
# 				images.append(image)
# 				angles.append(angle)
#
# 			X_train = np.array(images)
# 			y_train = np.array(angles)
# 			yield sklearn.utils.shuffle(X_train, y_train)

# # function to grab validation images.
# def generate_valid(validation_samples, batch_size=32):
# 	img_set = np.zeros((len(validation_samples), 64, 64, 3))
# 	steer_set = np.zeros(len(validation_samples))
# 	for i in range(len(validation_samples)):
# 		sample = validation_samples[i]
# 		name = sample[0].strip()
# 		# special case for udacity data
# 		if sample[0].split('/')[0].strip() == 'IMG':
# 			name = './data/IMG/'+sample[0].split('/')[-1]
# 		steer_set[i] = float(sample[3])
# 		image = cv2.imread(name)
# 		img = image[60:136,0:image.shape[1],:]
# 		img_set[i] =  cv2.resize(img, (64, 64), cv2.INTER_AREA)
#
# 	return img_set, steer_set


# # compile and train the model using the generator function
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generate_valid(validation_samples, batch_size=32)
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

# model.summary()
# print('modal summary: ', model.summary())
# adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer='adam', loss='mse')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=valid_generator, nb_epoch=5, verbose=1)
###############
# End - Keras Model
##############


# # Save the model and weights
# model_json = model.to_json()
# with open('model.json', "w") as json_file:
#     json_file.write(model_json)
model.save('model.h5')
K.clear_session()