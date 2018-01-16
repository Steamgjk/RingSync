"""
mnist tester (train and test accuracy)

date: 9/24
author: arabian9ts
"""

# escape tensorflow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
import tensorflow as tf
import pickle
import numpy as np
import datetime
import time

from vgg16 import *
from util import *

# global variables
DATASET_NUM = 10000
BATCH = 10
EPOCH = 32
learning_rate = 0.05
loss_loop = DATASET_NUM/BATCH/10

images = []
labels = []

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
        

def get_next_batch(max_length, length=BATCH, is_training=True):
    """
    extract next batch-images

    Returns: batch sized BATCH
    """
    if is_training:
        indicies = np.random.choice(max_length, length)
        next_batch = train_images[indicies]
        next_labels = train_labels[indicies]
    else:
        indicies = np.random.choice(max_length, length)
        next_batch = test_images[indicies]
        next_labels = test_labels[indicies]

    return np.array(next_batch), np.array(next_labels)

def test():
    """
    do test
    """
    images, labels = get_next_batch(max_length=len(test_labels), length=100, is_training=False)
    result = sess.run(predict, feed_dict={input: images})

    correct = 0
    total = 100

    for i in range(len(labels)):
        pred_max = result[i].argmax()
        ans = labels[i].argmax()

        if ans == pred_max:
            correct += 1

    correct_float = float(correct)
    print(str(correct_float/total))


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    """
    TensorFlow session
    """
    args = sys.argv

    # use VGG16 network
    vgg = VGG16()
    # params for converting to answer-label-size
    w = tf.Variable(tf.truncated_normal([512, 10], 0.0, 1.0) * 0.01, name='w_last')
    b = tf.Variable(tf.truncated_normal([10], 0.0, 1.0) * 0.01, name='b_last')

    # input image's placeholder and output of VGG16
    input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
    fmap = vgg.build(input, is_training=True)
    predict = tf.nn.softmax(tf.add(tf.matmul(fmap, w), b))

    # params for defining Loss-func and Training-step
    ans = tf.placeholder(shape=None, dtype=tf.float32)
    ans = tf.squeeze(tf.cast(ans, tf.float32))

    # cross-entropy
    loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(predict), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    # load image data
    train_images, train_labels, test_images, test_labels = load_data()

    ### restoring saved parameters ###
    if 2 == len(args) and 'eval' == args[1]:
        # parameter saver
        saver = tf.train.Saver()
        saver.restore(sess, './params.ckpt')
        test()
        sys.exit()
    # ========= Loading END ======== #

    ## Perform training ##
    print('\nSTART LEARNING')
    print('==================== '+str(datetime.datetime.now())+' ====================')
    time_begin = time.time()
    print("Training begins @ %s" % time.ctime(time_begin))

    # Training-loop
    lossbox = []
    for e in range(EPOCH):
        epoch_time_begin = time.time()
        for b in range(int(DATASET_NUM/BATCH)):
            batch, actuals = get_next_batch(len(train_labels))
            sess.run(train_step, feed_dict={input: batch, ans: actuals})

            if b == int(DATASET_NUM/BATCH) - 1:
                print(str(sess.run(loss, feed_dict={input: batch, ans: actuals})).ljust(7)[:7] + '  '),

        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_begin
        print("    %3d        %5f        " % (e + 1, epoch_time)),
        test()
        lossbox.append(sess.run(loss, feed_dict={input: batch, ans: actuals})) 

    print('==================== '+str(datetime.datetime.now())+' ====================')
    print('\nEND LEARNING')
    time_end = time.time()
    print("Training ends @ %s" % time.ctime(time_end))
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)
