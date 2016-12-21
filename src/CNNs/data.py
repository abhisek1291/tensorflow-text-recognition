############################################
#
# Author : Abhisek Mohanty
# Description : This files contain the methods that provide training data for the two CNNs
#
############################################

import os
import re

import cv2
import numpy as np

from images.resizeImage import easy_resize


def load_detection_data(positive_data_path, negative_data_path):
    """
    :param positive_data_path: folder path for positive training images
    :param negative_data_path: folder path for negative training images
    :return: list_of_images, list_of_image_labels, mean of pixels in each column, standard deviation of pixels in each column
    """

    print '\nloading train data :'
    cwd = os.getcwd()
    positive_directory = os.path.join(cwd, positive_data_path)
    negative_directory = os.path.join(cwd, negative_data_path)
    directories = [positive_directory, negative_directory]

    images = []
    labels = []

    for directory in directories:
        if 'positive' in directory:
            label = [0, 1]
        elif 'negative' in directory:
            label = [1, 0]
        else:
            raise

        # files = [f for f in os.listdir(directory)]
        files = [f for f in os.listdir(directory) if re.match(r'.*\.png', f)]
        for file_name in files:
            # print file_name
            file_path = os.path.join(directory, file_name)
            img = cv2.imread(file_path, 0)
            # sobel operator on img
            # img = cv2.Laplacian(img,cv2.CV_64F)
            img = img.flatten()
            images.append(img)
            labels.append(label)

    mean = np.mean(images, axis=0)
    images -= np.mean(images, axis=0)
    sd = np.std(images, axis=0)
    images /= np.std(images, axis=0)
    return [images, labels, mean, sd]


def load_recognition_data(data_path):
    """
    This method provides training data to the Text Recognition CNN
    :param data_path: Path of data files
    :return: List of images and their labels, mean and standard deviation
    """
    print '\nloading train data :'
    cwd = os.getcwd()
    directory = os.path.join(cwd, data_path)
    # print directory

    images = []
    labels = []

    for x in os.walk(directory):
        subdirectory = x[0]
        if 'Sample' not in subdirectory:
            continue

        str_number = subdirectory[-3:]
        number = int(str_number.lstrip('0'))

        # label = np.zeros(36)
        n_output = 37
        label = [0] * n_output
        if number == 36:
            index = number - 1
        elif number > 36:
            index = number % 36 + 9
        else:
            index = number % 36 - 1

        if 'negative' in subdirectory:
            index = 36

        label[index] = 1
        # print subdirectory + '\t' + str_number + '\t' + str(number) + '\t' + str(index)
        # print os.path.join(directory, subdirectory)

        files = [f for f in os.listdir(os.path.join(directory, subdirectory)) if re.match(r'.*\.png', f)]
        for file_name in files:
            # print file_name
            file_path = os.path.join(directory, subdirectory, file_name)
            img = cv2.imread(file_path, 0)
            img = easy_resize(img, 32)
            # sobel operator on img
            # img = cv2.Laplacian(img,cv2.CV_64F)
            img = img.flatten()
            images.append(img)
            labels.append(label)

    mean = np.mean(images, axis=0)
    images -= np.mean(images, axis=0)
    sd = np.std(images, axis=0)
    images /= np.std(images, axis=0)
    # return [images, labels, mean, sd]
    return images, labels, mean, sd


def batch(data, batch_size, num_epochs, shuffle=True):
    """
    This method is used to provide batches of data to the CNN. Following are the parameters:
    :param data: The training data [matrix of images in our case]
    :param batch_size: The size of each batch {Integer value, for ex. 256 or 512}
    :param num_epochs: The total epochs for which the data needs to be provided
    :param shuffle: Shuffle the data in each batch. Has better training results compared to unshuffled data
    :return: Batches of data based on the batch size
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
