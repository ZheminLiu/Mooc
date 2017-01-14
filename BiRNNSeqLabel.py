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
        max_length = int(self.target.get_shape()[1])  # max_length = week_num
        num_classes = int(self.target.get_shape()[2])  # num_classes = label_class_num

        # Permuting batch_size and week_num
        x = tf.transpose(self.data, [1, 0, 2])
        # Reshape to (week_num*batch_size, feature_num)
        x = tf.reshape(x, [-1, int(self.data.get_shape()[2])])
        # Split to get a list of 'week_num' tensors of
        # Partition by Weeks
        x = tf.split(0, max_length, x)

        # Forward direction cell
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._num_hidden, forget_bias=1.0, state_is_tuple=True)
        # Backward direction cell
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self._num_hidden, forget_bias=1.0, state_is_tuple=True)

        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout)

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * self._num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * self._num_layers, state_is_tuple=True)

        # outputs = [week_num * stu_num * 2Hidden_num] is list of tensor
        outputs, output_state_fw, output_state_bw = tf.nn.bidirectional_rnn(fw_cell, bw_cell, x,
                                                                            dtype=tf.float32)
        # transfer list of 2D tensor into a 3D tensor
        outputs = tf.pack(outputs)
        # [week_num, stu_num, 2*hidden_num] => [stu_num, week_num, 2*hidden_num]
        outputs = tf.transpose(outputs, [1, 0, 2])
        # weight==[2*hidden_num, num_classes], bias==[num_classes]
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # [stu_num, week_num, 2*hidden_num] => [stu_num * week_num, 2*hidden_num]
        output = tf.reshape(outputs, [-1, 2 * self._num_hidden])
        # 然后计算每个分类的softmax概率值
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])

        return prediction

    @lazy_property
    def cost(self):
        # Compute cross entropy for each week.
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction), [1, 2])
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
        results = []
        for i in xrange(max_len):
            correct = tf.equal(tf.argmax(targets[i], 2), tf.argmax(predicts[i], 2))
            results.append(tf.reduce_mean(tf.cast(correct, tf.float32)))
        return results, targets

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([2 * in_size, out_size], mean=0.0, stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
