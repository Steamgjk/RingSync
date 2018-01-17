# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import t_ring.tensorflow as tr

# escape tensorflow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
import re
import pickle
import tensorflow as tf
import numpy as np
import datetime
import time
import math

from vgg16 import *
from util import *

layers = tf.contrib.layers
learn = tf.contrib.learn

tf.logging.set_verbosity(tf.logging.INFO)

# global variables
DATASET_NUM = 10000
BATCH = 100
EPOCH = 32
learning_rate = 0.05
loss_loop = DATASET_NUM/BATCH/10

images = []
labels = []
global epoch_loss

def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = layers.conv2d(
            feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = layers.conv2d(
            h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dropout(
        layers.fully_connected(
            h_pool2_flat, 1024, activation_fn=tf.nn.relu),
        keep_prob=0.5,
        is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss


def gen_onehot_list(label=0):
    """
    generate one-hot label-list based on ans-index
    e.g. if ans is 3, return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Args: answer-index
    Returns: one-hot list
    """
    return [1 if l==label else 0 for l in range(0, 10)]


def load_data():
    """
    open cifar-dataset
    segregate images-data and answers-label to images and labels
    """
    with open('dataset/data_batch_1', 'rb') as f:
        data = pickle.load(f)
        slicer = int(DATASET_NUM*0.8)
        train_images = np.array(data['data'][:slicer]) / 255
        train_labels = np.array(data['labels'][:slicer])
        test_images = np.array(data['data'][slicer:]) / 255
        test_labels = np.array(data['labels'][slicer:])
        reshaped_train_images = np.array([x.reshape([32, 32, 3]) for x in train_images])
        reshaped_train_labels = np.array([gen_onehot_list(i) for i in train_labels])
        reshaped_test_images = np.array([x.reshape([32, 32, 3]) for x in test_images])
        reshaped_test_labels = np.array([gen_onehot_list(i) for i in test_labels])

    return reshaped_train_images, reshaped_train_labels, reshaped_test_images, reshaped_test_labels
        

def get_next_batch(images, labels, max_length, length=BATCH, is_training=True):
    """
    extract next batch-images

    Returns: batch sized BATCH
    """
    if is_training:
        indicies = np.random.choice(max_length, length)
        next_batch = images[indicies]
        next_labels = labels[indicies]
    else:
        indicies = np.random.choice(max_length, length)
        next_batch = images[indicies]
        next_labels = labels[indicies]

    return np.array(next_batch), np.array(next_labels)

def test(sess, images, labels, predict, x_input):
    """
    do test
    """
    images, labels = get_next_batch(images, labels, max_length=len(labels), length=100, is_training=False)
    result = sess.run(predict, feed_dict={x_input: images})

    correct = 0
    total = 100

    for i in range(len(labels)):
        pred_max = result[i].argmax()
        ans = labels[i].argmax()

        if ans == pred_max:
            correct += 1

    correct_float = float(correct)
    print(str(correct_float/total))


def main(_):
    # Initialize
    print("ok")
    tr.init()
    train_images, train_labels, test_images, test_labels = load_data()
    print("ok2")
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.name_scope('input'):
        # input #
        x_input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        # output #
        ans = tf.placeholder(shape=None, dtype=tf.float32)
        ans = tf.squeeze(tf.cast(ans, tf.float32))
    print("ok3")
    # use VGG16 network
    vgg = VGG16()
    # params for converting to answer-label-size
    w = tf.Variable(tf.truncated_normal([512, 10], 0.0, 1.0) * 0.01, name='w_last')
    b = tf.Variable(tf.truncated_normal([10], 0.0, 1.0) * 0.01, name='b_last')
    print("ok4")  
    fmap = vgg.build(x_input, is_training=True)
    predict = tf.nn.softmax(tf.add(tf.matmul(fmap, w), b))
    loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(predict), reduction_indices=[1]))
    print("ok5")
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #add our code
        optimizer =  tr.DistributedOptimizer(optimizer)
        '''
        if FLAGS.sync_replicas:
            optimizer = tf.train.SyncReplicasOptimizer(
                optimizer,
                replicas_to_aggregate = num_workers,
                total_num_replicas = num_workers,
                name = "vgg16_sync_replicas")
        '''
        train_step = optimizer.minimize(loss, global_step = global_step)

    print("ok7")

    #add our code
    hooks = [tr.BroadcastGlobalVariablesHook(0),
             tf.train.StopAtStepHook(last_step=10000),
             tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss}, every_n_iter=10),
             ]
    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.visible_device_list = str(bq.local_rank())

    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    #checkpoint_dir = './checkpoints' if tr.rank() == 0 else None
    checkpoint_dir = None
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    print("ok 9")
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            batch, actuals = get_next_batch(train_images, train_labels, len(train_labels))
            _, step = mon_sess.run([train_step, global_step], feed_dict={x_input: batch, ans: actuals})
            #_, step = mon_sess.run(train_step, feed_dict={x_input: batch, ans: actuals})

            #image_, label_ = mnist.train.next_batch(100)
            #mon_sess.run(train_op, feed_dict={image: image_, label: label_})


'''
    # Download and load MNIST dataset.
    mnist = learn.datasets.mnist.read_data_sets('MNIST-data-%d' % tr.rank())

    # Build model...
    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32, [None, 784], name='image')
        label = tf.placeholder(tf.float32, [None], name='label')
    predict, loss = conv_model(image, label, tf.contrib.learn.ModeKeys.TRAIN)

    opt = tf.train.RMSPropOptimizer(0.01)

    # Add Bcube Distributed Optimizer.
    opt = tr.DistributedOptimizer(opt)

    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    # BroadcastGlobalVariablesHook broadcasts initial variable states from rank 0
    # to all other processes. This is necessary to ensure consistent initialization
    # of all workers when training is started with random weights or restored
    # from a checkpoint.
    hooks = [tr.BroadcastGlobalVariablesHook(0),
             tf.train.StopAtStepHook(last_step=10000),
             tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                        every_n_iter=10),
             ]

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.visible_device_list = str(bq.local_rank())

    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    #checkpoint_dir = './checkpoints' if tr.rank() == 0 else None
    checkpoint_dir = None
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = mnist.train.next_batch(100)
            mon_sess.run(train_op, feed_dict={image: image_, label: label_})
'''

if __name__ == "__main__":
    tf.app.run()
