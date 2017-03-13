import os
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
udacity_csv = 'udacity_data/driving_log.csv'
trained_csv = 'trained_data/left_track/driving_log.csv'
trained_reverse_csv = 'trained_data/left_track_reverse/driving_log.csv'
corrections_csv = 'trained_data/edge_turns/driving_log.csv'
ch, row, col = 3, 64, 64
############################################
# Step 1: Read the CSV File
lines = []
with open(udacity_csv) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)
print('length of 1: ', len(lines))

with open(trained_csv) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)
print('length of 2: ', len(lines))

with open(trained_reverse_csv) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

with open(corrections_csv) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

count = 0
new_sample = []
for line in lines:
    # print('sample 3: ', sample[3])
    # exit()
	center_angle = float(line[3])
	if center_angle == 0.0:
		count = count + 1
		if np.random.uniform() < 0.2:
			new_sample.append(line)
	else:
		new_sample.append(line)

lines = new_sample

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


############################################
# Step 2: Build Array for Camera Views/Steering Angles
def get_image_and_angle(batch_sample):
    # Image
    random = np.random.randint(3)
    path = batch_sample[random].strip()
    if path.split('/')[0] == 'IMG':
        path = dir_path + '/udacity_data/' + path

    # Angle
    angle = float(batch_sample[3])
    if random == 1:
        angle = angle + 0.22
    if random == 2:
        angle = angle - 0.22

    return path, angle

def preprocess(image, angle):
    # image, angle = shift_img(image, angle)
    image, angle = flip_image(image, angle)
    image = augment_brightness(image)
    img = image[60:136, 0:image.shape[1], :]

    # return image, angle
    return cv2.resize(img, (64, 64), cv2.INTER_AREA), angle

def shift_img(image, angle):
    """ shift image randomly
    	source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py """
    max_shift = 55
    max_ang = 0.14  # ang_per_pixel = 0.0025

    rows, cols, _ = image.shape

    random_x = np.random.randint(-max_shift, max_shift + 1)
    steer = angle + (random_x / max_shift) * max_ang
    if abs(steer) > 1:
        dst_steer = -1 if (steer < 0) else 1

    mat = np.float32([[1, 0, random_x], [0, 1, 0]])
    dst_img = cv2.warpAffine(image, mat, (cols, rows))
    return dst_img, steer

def flip_image(image, angle):
    flip_image = image.copy()
    flip_angle = angle
    num = np.random.randint(2)
    if num == 0:
        flip_image, flip_angle = cv2.flip(image, 1), -angle

    return flip_image, flip_angle

def augment_brightness(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random = np.random.randint(2)

    if random == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        hsv_img[:,:,2] = hsv_img[:,:,2] * random_bright
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    return hsv_img

def train_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                image, angle = get_image_and_angle(batch_sample)
                image = cv2.imread(image)

                image, angle = preprocess(image, angle)
                images.append(image)
                steering_angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

def valid_generator(samples, batch_size=32):
    # image_set = np.zeros((len(validation_samples), 160, 320, 3))
    image_set = np.zeros((len(validation_samples), 64, 64, 3))
    angles_set = np.zeros(len(validation_samples))

    for i in range(len(samples)):
        # Image
        sample = samples[i]
        path = sample[0].strip()
        if path.split('/')[0] == 'IMG':
            path = dir_path + '/udacity_data/' + path

        image = cv2.imread(path)
        img = image[60:136,0:image.shape[1],:]
        image_set[i] = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        # image_set[i] = image

        # Angle
        angles_set[i] = float(sample[3])

        return image_set, angles_set

#######
# Old
# trained_image_path = 'trained_data/left_track/IMG/'
# images = []
# # measurements = []
# for idx, line in enumerate(lines):
#     for i in range(3):
#         # get image
#         source_path = line[i]
#         tokens = source_path.split('/')
#         filename = tokens[-1]
#
#         # format to image
#         # local_path = 'IMG/' + filename
#         local_path = trained_image_path + filename
#         image = cv2.imread(local_path)
#         images.append(image)
#
#     # grab measurement
#     correction = 0.2
#     measurement = float(line[3])
#     # center image measurement
#     measurements.append(measurement)
#     # left image measurement
#     measurements.append(measurement + correction)
#     # rightimage measurement
#     measurements.append(measurement - correction)

# print("Print images length: ", len(images))
# print("Print measurements length: ", len(measurements))

# augmented_images = []
# augmented_measurements = []
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     # flip around the vertical axis
#     flipped_image = cv2.flip(image, 1)
#     flipped_measurement = float(measurement) * -1.0
#
#     augmented_images.append(flipped_image)
#     augmented_measurements.append(flipped_measurement)
# End - Old
#######


############################################
# Step 3: Transform Training Data Into Numpy Arrays (Keras Expects That)
# X_train = np.array(images)
# y_train = np.array(measurements)
train_generator = train_generator(train_samples, batch_size=32)
valid_generator = valid_generator(validation_samples, batch_size=32)

# Step 4: Build Small Keras Model
model = Sequential()
# normalization
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((72,25),(0,0))))


# nvidia nn
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu', name="conv0"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu', name="conv1"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu', name="conv2"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(64,3,3,activation='relu', name="conv3"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(64,3,3, activation='relu', name="conv4"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1))

# model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Convolution2D(64, 5, 5, activation="relu"))
# model.add(Convolution2D(64, 5, 5, activation="relu"))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

# original nn
# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=valid_generator, nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
K.clear_session()