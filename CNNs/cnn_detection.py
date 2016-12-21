############################################
#
# Author : Abhisek Mohanty
# Description : This file contains methods that train the CNN for text recognition.
#               Uses tensorflow. Creates a ConvNet from scratch using tensorflow.
#
############################################

import tensorflow as tf
from sklearn.cross_validation import train_test_split

import data
import modules as m

tf.flags.DEFINE_string("positive_data_loc", "files/positive", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_loc", "files/negative", "Data source for the positive data.")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 512)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

FLAGS = tf.flags.FLAGS


class CNN:
    def __init__(self, n_inputs, n_outputs, dropout, learning_rate):
        self.input_shape = n_inputs
        self.output_shape = n_outputs
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.x, self.y, self.weights, self.biases, self.keep_prob = self.initialize()

    def initialize(self):
        x = tf.placeholder(tf.float32, [None, self.input_shape])
        y = tf.placeholder(tf.float32, [None, self.output_shape])
        keep_prob = tf.placeholder(tf.float32)

        weights = {
            'conv1': m.weights([3, 3, 1, 32], 'conv1_w'),
            'conv2': m.weights([3, 3, 32, 32], 'conv2_w'),
            'conv3': m.weights([5, 5, 32, 64], 'conv3_w'),
            'conv4': m.weights([3, 3, 64, 256], 'conv4_w'),
            'fc1': m.weights([4 * 4 * 256, 1024], 'fc1_w'),
            'output': m.weights([1024, self.output_shape], 'output_w')
        }

        biases = {
            'conv1': m.biases([32], 'conv1_b'),
            'conv2': m.biases([32], 'conv2_b'),
            'conv3': m.biases([64], 'conv3_b'),
            'conv4': m.biases([256], 'conv4_b'),
            'fc1': m.biases([1024], 'fc1_b'),
            'output': m.biases([self.output_shape], 'output_b')
        }

        return x, y, weights, biases, keep_prob

    def cnn_init(self):
        """
        Initialize the Text detection CNN
        :return: The convolutional Neural Network
        """
        network = self.x
        y = self.y

        # shape = l*b*channels
        network = tf.reshape(network, shape=[-1, 32, 32, 1])

        # first Layer, conv + pool
        network = m.conv2d(network, self.weights['conv1'], self.biases['conv1'])
        print 'conv1 : ' + str(network.get_shape())
        network = m.pool(network)
        print 'pool1 : ' + str(network.get_shape())

        # second layer, conv + pool
        network = m.conv2d(network, self.weights['conv2'], self.biases['conv2'])
        print 'conv2 : ' + str(network.get_shape())
        # network = network
        network = m.pool(network)
        # print 'pool2 : ' + str(pool2.get_shape())

        # third layer, conv + pool
        # pool2 = np.reshape(pool2, (-1, 1024))
        network = m.conv2d(network, self.weights['conv3'], self.biases['conv3'])
        print 'conv3 : ' + str(network.get_shape())
        network = m.pool(network)
        print 'pool3 : ' + str(network.get_shape())

        # fourth layer, conv + pool
        network = m.conv2d(network, self.weights['conv4'], self.biases['conv4'])
        print 'conv4 : ' + str(network.get_shape())
        network = m.pool(network)
        print 'pool3 : ' + str(network.get_shape())

        # fully connected
        network = tf.reshape(network, [-1, 4 * 4 * 256])
        network = tf.nn.relu(tf.matmul(network, self.weights['fc1']) + self.biases['fc1'])
        network = tf.nn.dropout(network, keep_prob=self.keep_prob)
        print 'fc1 : ' + str(network.get_shape())

        # output layer
        network = tf.matmul(network, self.weights['output']) + self.biases['output']

        return network

    def train_cnn(self):
        """
        Train the CNN.
        :return: None
        """
        batches, X_test, y_test = self.train_data()

        print '\ninitializing CNN...\n'
        y_out = self.cnn_init()

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, self.y))
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # print [v.op.name for v in tf.all_variables()]
            iteration = 1

            print '\nstarting training...\n'

            for batch in batches:
                x_batch, y_batch = zip(*batch)

                if iteration % 50 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        self.x: x_batch, self.y: y_batch, self.keep_prob: self.dropout})

                    print("\n\tstep %d, training accuracy %g" % (iteration, train_accuracy))

                iteration += 1

                train_step.run(feed_dict={
                    self.x: x_batch, self.y: y_batch, self.keep_prob: self.dropout})

            save_path = saver.save(sess, "files/model.ckpt")
            print("Model saved in file: %s\n" % save_path)

            print("test accuracy %g" % accuracy.eval(feed_dict={
                self.x: X_test, self.y: y_test, self.keep_prob: 1.}))
        print 'training done'

    def train_data(self):
        """
        This module calls the batch method in the data.py file to create batches of data for training the CNN
        :return: Batches of data, Test Dataset
        """
        print '\nreading training data...\n'
        x_data, y_data, mean, sd = data.load_detection_data(FLAGS.positive_data_loc, FLAGS.negative_data_loc)
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

        batches = data.batch(
            list(zip(X_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        print '\ndata fetch and batch creation done...\n'
        return batches, X_test, y_test

