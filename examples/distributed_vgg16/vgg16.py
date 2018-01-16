"""
implementation of vgg network with TensorFlow

date: 9/17
author: arabian9ts
"""

import tensorflow as tf
import numpy as np
import vgg16_structure as vgg

from functools import reduce
from activation import Activation

class VGG16:
    def __init__(self):
        pass

    def build(self, input, is_training=True):
        """
        input is the placeholder of tensorflow
        build() assembles vgg16 network
        """

        # flag: is_training? for tensorflow-graph
        self.train_phase = tf.constant(is_training) if is_training else None

        with tf.device('/gpu:0'):
            self.conv1_1 = self.convolution(input, 'conv1_1')
            self.conv1_2 = self.convolution(self.conv1_1, 'conv1_2')
            self.pool1 = self.pooling(self.conv1_2, 'pool1')

        with tf.device('/gpu:1'):
            self.conv2_1 = self.convolution(self.pool1, 'conv2_1')
            self.conv2_2 = self.convolution(self.conv2_1, 'conv2_2')
            self.pool2 = self.pooling(self.conv2_2, 'pool2')

        with tf.device('/gpu:0'):
            self.conv3_1 = self.convolution(self.pool2, 'conv3_1')
            self.conv3_2 = self.convolution(self.conv3_1, 'conv3_2')
            self.conv3_3 = self.convolution(self.conv3_2, 'conv3_3')
            self.pool3 = self.pooling(self.conv3_3, 'pool3')

        with tf.device('/gpu:1'):
            self.conv4_1 = self.convolution(self.pool3, 'conv4_1')
            self.conv4_2 = self.convolution(self.conv4_1, 'conv4_2')
            self.conv4_3 = self.convolution(self.conv4_2, 'conv4_3')
            self.pool4 = self.pooling(self.conv4_3, 'pool4')

        with tf.device('/gpu:0'):
            self.conv5_1 = self.convolution(self.pool4, 'conv5_1')
            self.conv5_2 = self.convolution(self.conv5_1, 'conv5_2')
            self.conv5_3 = self.convolution(self.conv5_2, 'conv5_3')
            self.pool5 = self.pooling(self.conv5_3, 'pool5')

        with tf.device('/gpu:1'):
            self.fc6 = self.fully_connection(self.pool5, tf.nn.relu, 'cifar')

        with tf.device('/gpu:0'):
            self.fc7 = self.fully_connection(self.fc6, tf.nn.relu, 'fc7')

        with tf.device('/gpu:1'):
            self.fc8 = self.fully_connection(self.fc7, tf.nn.relu, 'fc8')

        self.prob = self.fc6

        return self.prob



    def pooling(self, input, name):
        """
        Args: output of just before layer
        Return: max_pooling layer
        """
        return tf.nn.max_pool(input, ksize=vgg.ksize, strides=vgg.pool_strides, padding='SAME', name=name)

    def convolution(self, input, name):
        """
        Args: output of just before layer
        Return: convolution layer
        """
        print('Current input size in convolution layer is: '+str(input.get_shape().as_list()))
        with tf.variable_scope(name):
            size = vgg.structure[name]
            kernel = self.get_weight(size[0], name='w_'+name)
            bias = self.get_bias(size[1], name='b_'+name)
            conv = tf.nn.conv2d(input, kernel, strides=vgg.conv_strides, padding='SAME', name=name)
            out = tf.nn.relu(tf.add(conv, bias))
        return self.batch_normalization(out)

    def fully_connection(self, input, activation, name):
        """
        Args: output of just before layer
        Return: fully_connected layer
        """
        size = vgg.structure[name]
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = reduce(lambda x, y: x * y, shape[1:])
            x = tf.reshape(input, [-1, dim])

            weights = self.get_weight([dim, size[0][0]], name=name)
            biases = self.get_bias(size[1], name=name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            fc = activation(fc)

            print('Input shape is: '+str(shape))
            print('Total nuron count is: '+str(dim))
            
            return self.batch_normalization(fc)
    
    def batch_normalization(self, input, decay=0.9, eps=1e-5):
        """
        Batch Normalization
        Result in:
            * Reduce DropOut
            * Sparse Dependencies on Initial-value(e.g. weight, bias)
            * Accelerate Convergence
            * Enable to increase training rate

        Args: output of convolution or fully-connection layer
        Returns: Normalized batch
        """
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        beta = tf.Variable(tf.zeros([n_out]))
        gamma = tf.Variable(tf.ones([n_out]))

        if len(shape) == 2:
            batch_mean, batch_var = tf.nn.moments(input, [0])
        else:
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(self.train_phase, mean_var_with_update,
          lambda: (ema.average(batch_mean), ema.average(batch_var)))

        return tf.nn.batch_normalization(input, mean, var, beta, gamma, eps)


    def get_weight(self, shape, name):
        """
        generate weight tensor

        Args: weight size
        Return: initialized weight tensor
        """
        initial = tf.truncated_normal(shape, 0.0, 1.0) * 0.01
        return tf.Variable(initial, name='w_'+name)

    def get_bias(self, shape, name):
        """
        generate bias tensor

        Args: bias size
        Return: initialized bias tensor
        """
        return tf.Variable(tf.truncated_normal(shape, 0.0, 1.0) * 0.01, name='b_'+name)
