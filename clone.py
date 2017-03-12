import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


lines = []
csv_file = 'driving_log.csv'
desktop_csv_file = 'trained_data/left_track/driving_log.csv'

# Step 1: Read the CSV File
with open(desktop_csv_file) as f:
    reader = csv.reader(f)
    for line in reader:
        # 'line' is essentially a token
        lines.append(line)


# Step 2: Build Array for Camera Views/Steering Angles
desktop_img_path = 'trained_data/left_track/IMG/'
images = []
measurements = []
for idx, line in enumerate(lines):
    # Udacity's Example Dataset has Labels for [0]
    if idx != 0:
        for i in range(3):
            # get image
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]

            # format to image
            # local_path = 'IMG/' + filename
            local_path = desktop_img_path + filename
            image = cv2.imread(local_path)
            images.append(image)

        # grab measurement
        correction = 0.2
        measurement = float(line[3])
        # center image measurement
        measurements.append(measurement)
        # left image measurement
        measurements.append(measurement + correction)
        # rightimage measurement
        measurements.append(measurement - correction)

# print("Print images length: ", len(images))
# print("Print measurements length: ", len(measurements))

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    # flip around the vertical axis
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0

    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)


# Step 3: Transform Training Data Into Numpy Arrays (Keras Expects That)
X_train = np.array(images)
y_train = np.array(measurements)


# Step 4: Build Small Keras Model
model = Sequential()
# normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((72,25),(0,0))))

# nvidia nn
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Convolution2D(64,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), border_mode='same'))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
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

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')