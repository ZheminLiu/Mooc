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


class BiRNN(object):
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
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])

        # Permuting batch_size and n_steps
        x = tf.transpose(self.data, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, 15])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, max_length, x)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._num_hidden, forget_bias=1.0)

        outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(outputs, [-1, 2*self._num_hidden])

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
        mistakes = tf.not_equal(tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    # 输出平均准确率
    def correct(self):
        correct = tf.equal(tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    @lazy_property
    # 动态地输出每周的预测准确率
    def dynamic_correct(self):
        max_len = self.target.get_shape()[1]
        predicts = tf.split(1, max_len, self.prediction)
        targets = tf.split(1, max_len, self.target)
        print targets
        results = []
        for i in xrange(max_len):
            correct = tf.equal(tf.argmax(targets[i], 2), tf.argmax(predicts[i], 2))
            results.append(tf.reduce_mean(tf.cast(correct, tf.float32)))
        return results

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([2*in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)