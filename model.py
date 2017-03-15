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

dir_path = os.path.dirname(os.path.realpath(__file__))
udacity_csv = 'udacity_data/driving_log.csv'
# trained_csv = 'trained_data/left_track/driving_log.csv'
# trained_reverse_csv = 'trained_data/left_track_reverse/driving_log.csv'
# corrections_csv = 'trained_data/corrections/driving_log.csv'


############################################
# Step 1: Read the CSV File
############################################
lines = []
with open(udacity_csv) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


############################################
# Step 2: Helper Methods for Images, Pass through Generators
############################################
def get_image_and_angle(batch_sample):
    """
    :param batch_sample: Row from CSV
    :return: String (path, angle are strings from row)
    """
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

def shift_img(image, angle):
    """
    Random Shift Image
    Inspiration for this Edition:
    source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py

    :param image: Numpy Array
    :param angle: Float
    :return: Numpy Array, Float (Altered)
    """
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
    """
    :param image: Numpy Array
    :param angle: Float
    :return: Numpy Array, Float (Altered)
    """
    flip_image = image.copy()
    flip_angle = angle
    num = np.random.randint(2)
    if num == 0:
        flip_image, flip_angle = cv2.flip(image, 1), -angle

    return flip_image, flip_angle

def augment_brightness(image):
    """
    :param image: Numpy Array
    :return: Numpy Array, (Altered)
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random = np.random.randint(2)

    if random == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        hsv_img[:,:,2] = hsv_img[:,:,2] * random_bright
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    return hsv_img

def preprocess(image, angle):
    """
    :param image: Numpy Array
    :param angle: Float
    :return: Numpy Array, Float (Altered)
    """
    image, angle = shift_img(image, angle)
    image, angle = flip_image(image, angle)
    image = augment_brightness(image)

    return image, angle

def train_generator(samples, batch_size=32):
    """
    Generators can be a great way to work with large amounts of data.
    Instead of storing the preprocessed data in memory all at once,
    using a generator you can pull pieces of the data and process them
    on the fly only when you need them, which is much more memory-efficient.

    :param samples: Array
    :param batch_size: Integer
    :return: Numpy Array (Images and Steering)
    """
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
    """
    :param samples: Array
    :param batch_size: Integer
    :return: Numpy Array (Images and Steering
    """
    image_set = np.zeros((len(validation_samples), 160, 320, 3))
    angles_set = np.zeros(len(validation_samples))

    for i in range(len(samples)):
        # Image
        sample = samples[i]
        path = sample[0].strip()
        if path.split('/')[0] == 'IMG':
            path = dir_path + '/udacity_data/' + path
        image = cv2.imread(path)
        # img = image[60:136,0:image.shape[1],:]
        image_set[i] = image

        # Angle
        angles_set[i] = float(sample[3])

        return image_set, angles_set

train_generator = train_generator(train_samples, batch_size=32)
valid_generator = valid_generator(validation_samples, batch_size=32)


############################################
# Step 3: Build Models and Execute
############################################

model = Sequential()

# normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((72,25),(0,0))))

# NVIDIA Inspiration
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
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

history_object = model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples),
    validation_data=valid_generator,
    nb_epoch=5,
    verbose=1)

model.save('model.h5')
K.clear_session()