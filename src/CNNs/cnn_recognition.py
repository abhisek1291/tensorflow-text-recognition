############################################
#
# Author : Abhisek Mohanty
# Description : This file contains methods that train the CNN for text recognition.
#               Using tflearn instead of tensorflow. TFLearn is a library based on tensorflow
#
############################################

from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
from sklearn.cross_validation import train_test_split
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from CNNs import data
from images import resizeImage


def train_tflearn_cnn_recognition():
    x_data, y_data, mean, sd = data.load_recognition_data("rec_files/english/img")
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    X = X_train
    Y = y_train

    print('data fetch done.')
    # # Make sure the data is normalized
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # # Create extra synthetic training data by flipping, rotating and blurring the
    # images on our data set.
    img_aug = ImageAugmentation()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    # Define our network architecture:

    # Input is a 32x32 image with 1 color channel
    network = input_data(shape=[None, 32, 32, 1])

    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 32, 3, activation='relu')
    network = conv_2d(network, 64, 5, activation='relu')
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 37, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='path/to/digit-recognizer.tfl.ckpt')

    # Train it for 400 epochs
    model.fit(np.reshape(X, (-1, 32, 32, 1)), Y, n_epoch=400, shuffle=True,
              validation_set=(np.reshape(X_test, (-1, 32, 32, 1)), y_test),
              show_metric=True, batch_size=512,
              snapshot_epoch=True,
              run_id='digit-classifier')

    # Save model when training is complete to a file
    model.save("path/to/digit-recognizer.tfl")
    print("Network trained and saved as digit-recognizer.tfl!")

    print("start predictions : \n")
    count = 0
    prediction = model.predict(np.reshape(X_test, (-1, 32, 32, 1)))
    for x, y in zip(prediction, y_test):
        if x == y:
            count += 1

    print('accuracy : '),
    print(str(count / float(len(y_test))))

    return model


def classify(image, model):
    print('loading model.')
    model.load('path/to/digit-classifier.tfl')
    print('model loaded.')
    image = resizeImage.add_padding(image)
    prediction = model.predict(np.reshape(image, (-1, 32, 32, 1)))

    return prediction
