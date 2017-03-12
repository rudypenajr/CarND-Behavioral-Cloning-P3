from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

def build():
    model = Sequential()

    ## Preprocessing:
    ## Center around 0 w/ small standard deviation
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(64, 64, 3)))


    # Network
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
    return model


