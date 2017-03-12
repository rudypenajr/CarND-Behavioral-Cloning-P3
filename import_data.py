import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os


class ImportData():
    def __init__(self):
        self.samples = []
        self.train_samples = []
        self.validation_samples = []
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def load_csv(self, csv_path):
        with open(csv_path) as f:
            reader = csv.reader(f)

            for line in reader:
                self.samples.append(line)

        print('Total Sample: ', len(self.samples))
        return self.samples

    def remove_partial_steering_angles(self):
        count = 0
        modified_samples = []

        for sample in self.samples:
            # Udacity Data has LABELS
            steering_angle = float(sample[3])


            if steering_angle == 0.0:
                count = count + 1
                if np.random.uniform() < 0.2:
                    modified_samples.append(sample)
            else:
                modified_samples.append(sample)


        # print('Total Count for Center Steering Angles: ', count)
        # print('Total Samples After Steering Angle Modifications: ', len(modified_samples))
        self.samples = modified_samples
        return self.samples

    def train_test_split(self):
        self.train_samples, self.validation_samples = train_test_split(self.samples, test_size=0.2)
        print('Train Samples: ', len(self.train_samples))
        print('Validation Samples:', len(self.validation_samples))

        return self.train_samples, self.validation_samples

    def get_img_and_angle(self, sample):
        rand = np.random.randint(3)
        path = sample[rand].strip()

        # Fix Udacity Data Path for Image
        if sample[rand].split('/')[0].strip() == 'IMG':
            # print('path updated: ', self.dir_path, path)
            path = self.dir_path + '/udacity_data/' + path

        angle = float(sample[3])

        if rand == 1:
            angle += 0.22
        elif rand == 2:
            angle -= 0.22

        if not path.endswith('.jpg'):
            raise ValueError('{path} is not an actual image path.')
            exit()

        return path, angle

    def train_generator(self, batch_size=32):
        samples = self.train_samples
        num_samples = len(samples)
        # Loop 4-ever so the generator never terminates
        while 1:
            shuffle(self.samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    path, st_angle = self.get_img_and_angle(batch_sample)

                    image = cv2.imread(path)
                    image, st_angle = self.preprocess_data(image, st_angle)
                    images.append(image)
                    angles.append(st_angle)
                    # print('image taco: ', image.shape)
                    # exit()

                X_train = np.array(images)
                y_train = np.array(angles)
                # print('X_train', X_train.shape) returning (32, 76, 320, 3)
                yield shuffle(X_train, y_train)

    def preprocess_data(self, image, st_angle):
        # Step 1: Shift Image
        image, angle = self.get_schwifty(image, st_angle)

        # Step 2: Flip Image
        image, angle = self.flip_image(image, st_angle)

        # Step 3: Augment Brightness
        image = self.augment_brightness(image)

        # Step 4: Crop Before
        crop = image[60:136, 0:image.shape[1] ,:]
        # plt.imshow(crop)
        # plt.show()
        # exit()
        # return crop, st_angle
        return cv2.resize(image, (64, 64), cv2.INTER_AREA), angle

    def get_schwifty(self, image, st_angle):
        """
        Tribute to Rick and Morty : )
        Purpose is to shift image
        :param image: image multi-dimensional array in RBG
        :param angle: float
        :return: image, angle in similar format
        """
        max_shift = 55
        max_angle = 0.14
        # Expected Image Shape: (160, 320, 3)
        rows, columns, _ = image.shape

        random_x = np.random.randint(-max_shift, max_shift + 1)
        disrupt_str = st_angle + (random_x / max_shift) * max_angle
        if abs(disrupt_str) > 1:
            disrupt_str = -1 if (disrupt_str < 0) else 1

        mat = np.float32([[1, 0, random_x], [0, 1, 0]])
        warp_image = cv2.warpAffine(image, mat, (columns, rows))
        # plt.subplot(1,2,1)
        # plt.imshow(image)
        # plt.subplot(1,2,2)
        # plt.imshow(warp_image)
        # plt.show()
        # exit()
        return warp_image, disrupt_str

    def flip_image(self, image, st_angle):
        """
        :param image: Image (created from cv2), Matrix
        :param angle: Steering Angle, Float
        :return: Image (Matrix), Steering Angle (Float)
        """
        flip_img = image.copy()
        flip_angle = st_angle

        num = np.random.randint(2)
        if num == 0:
            flip_img = cv2.flip(image, 1)
            flip_angle = -st_angle

        # plt.subplot(1,2,1)
        # plt.imshow(image)
        # plt.subplot(1,2,2)
        # plt.imshow(flip_img)
        # plt.show()
        # exit()
        return flip_img, flip_angle

    def augment_brightness(self, image):
        """
        Randomly Change Brightness by Converting Y Value
        :param image: Matrix
        :param angle: Float
        :return: Matrix, Float (Image, Steering Angle)
        """
        image_br = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        num = np.random.randint(2)
        # plt.subplot(1,2,1)
        # plt.imshow(image_br)

        if num == 0:
            randomize_brightness = 0.2 + np.random.uniform(0.2, 0.6)
            image_br[:,:,0.2] = image_br[:,:,0.2] * randomize_brightness

        image_br = cv2.cvtColor(image_br, cv2.COLOR_HSV2RGB)
        # plt.subplot(1,2,2)
        # plt.imshow(image_br)
        # plt.show()
        return image_br

    def valid_generator(self, batch_size=32):
        image_set = np.zeros((len(self.validation_samples), 64, 64, 3))
        steer_set = np.zeros(len(self.validation_samples))
        # print('Generate valid Image Set', image_set)
        # print('Generate valid Steering Set', steer_set)

        for i in range(len(self.validation_samples)):
            row = self.validation_samples[i] #row
            image_path = row[0].strip()
            # print('row', row)
            # print('image path', image_path)

            if row[0].split('/')[0].strip() == 'IMG':
                image_path = self.dir_path + '/udacity_data/' + image_path
                # print('path has been updated:', image_path)

            steer_set[i] = float(row[3])
            # print('steer_set', steer_set[i])
            image = cv2.imread(image_path)
            # plt.subplot(1,2,1)
            # plt.imshow(image)
            # plt.show()
            image = image[60:136,0:image.shape[1],:]
            # plt.subplot(1, 2, 2)
            # plt.imshow(image)
            # plt.show()
            image_set[i] = cv2.resize(image, (64, 64), cv2.INTER_AREA)

            # plt.imshow(image_set[i])
            # plt.show()
            # exit()
        # print("image_set", image_set.shape)
        # exit()
        return image_set, steer_set