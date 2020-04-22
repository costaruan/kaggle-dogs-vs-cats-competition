import cv2
import numpy as np
import os

from random import shuffle
from tqdm import tqdm


def create_testing_data(directory, file, size):
    testing_data = []

    for image in tqdm(os.listdir(directory)):
        path = os.path.join(directory, image)
        image_number = image.split('.')[0]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (size, size))

        testing_data.append([np.array(image), image_number])

    shuffle(testing_data)
    np.save(os.path.join(directory, file), testing_data)

    return testing_data


def create_training_data(directory, file, size):
    training_data = []

    for image in tqdm(os.listdir(directory)):
        label = label_image(image)
        path = os.path.join(directory, image)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (size, size))

        training_data.append([np.array(image), np.array(label)])

    shuffle(training_data)
    np.save(os.path.join(directory, file), training_data)

    return training_data


def label_image(image):
    label = image.split('.')[-3]

    if label == 'cat':
        return [1, 0]
    elif label == 'dog':
        return [0, 1]
