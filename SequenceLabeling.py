# coding: utf8
import functools

import tensorflow as tf


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class SequenceLabelling(object):
    def __init__(self, data, target, dropout, session, num_hidden=200, num_layers=1):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.session = session
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.nn.rnn_cell.LSTMCell(self._num_hidden)
        network = tf.nn.rnn_cell.DropoutWrapper(network, output_keep_prob=self.dropout)
        network = tf.nn.rnn_cell.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction), reduction_indices=1)
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.01
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        # print(mistakes)
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
