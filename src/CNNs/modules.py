############################################
#
# Author : Abhisek Mohanty
# Description : This has helper files that are used in the text detection cnn
#
############################################

import tensorflow as tf


def weights(shape, name):
    return tf.Variable(
        tf.truncated_normal(shape, stddev=0.1),
        name=name
    )


def biases(shape, name):
    return tf.Variable(
        tf.truncated_normal(shape, stddev=0.1),
        name=name
    )


def conv2d(x, shape, bias, stride=1):
    layer = tf.nn.conv2d(x, shape, [1, stride, stride, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, bias)
    return tf.nn.relu(layer)


def pool(x, k=2, mode='max'):
    if mode == 'max':
        layer = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    elif mode == 'avg':
        layer = tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return layer
