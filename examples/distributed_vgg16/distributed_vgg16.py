# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""
Distributed MNIST training and validation, with model replicas.
A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.
The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

# global variables
DATASET_NUM = 10000
BATCH = 100
EPOCH = 32
learning_rate = 0.05
loss_loop = DATASET_NUM/BATCH/10

images = []
labels = []
global epoch_loss

'''
flags = tf.app.flags
flags.DEFINE_string("job_name", "","One of 'ps' or 'worker'")
flags.DEFINE_string("ps_hosts", "12.12.10.11:7777, 12.12.10.12:7777, 12.12.10.13:7777, 12.12.10.14:7777, 12.12.10.15:7777, 12.12.10.16:7777, 12.12.10.17:7777, 12.12.10.18:7777, 12.12.10.19:7777",
                    "List of hostname:port for ps jobs."
                    "This string should be the same on every host!!")
flags.DEFINE_string("worker_hosts", "12.12.10.11:2222, 12.12.10.12:2222, 12.12.10.13:2222, 12.12.10.14:2222, 12.12.10.15:2222, 12.12.10.16:2222, 12.12.10.17:2222, 12.12.10.18:2222, 12.12.10.19:2222",
                    "List of hostname:port for worker jobs."
                    "This string should be the same on every host!!")
flags.DEFINE_integer("task_index", None,
                     "Ps task index or worker task index, should be >= 0. task_index=0 is "
                     "the master worker task that performs the variable "
                     "initialization ")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_integer("train_steps", 1000,
                     "Number of (global) training steps to perform")
flags.DEFINE_boolean("allow_soft_placement", True, "True: allow")
flags.DEFINE_boolean("log_device_placement", False, "True: allow")
FLAGS = flags.FLAGS
'''
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

'''
def get_device_setter(num_parameter_servers, num_workers):
    """ 
    Get a device setter given number of servers in the cluster.
    Given the numbers of parameter servers and workers, construct a device
    setter object using ClusterSpec.
    Args:
        num_parameter_servers: Number of parameter servers
        num_workers: Number of workers
    Returns:
        Device setter object.
    """

    ps_hosts = re.findall(r'[\w\.:]+', FLAGS.ps_hosts) # split address
    worker_hosts = re.findall(r'[\w\.:]+', FLAGS.worker_hosts) # split address

    assert num_parameter_servers == len(ps_hosts)
    assert num_workers == len(worker_hosts)

    cluster_spec = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})

    # Get device setter from the cluster spec #
    return tf.train.replica_device_setter(cluster=cluster_spec)
'''

def main(unused_argv):
    # Initialize .
    tr.init()
    # load image data
    train_images, train_labels, test_images, test_labels = load_data()
'''
    ps_hosts = re.findall(r'[\w\.:]+', FLAGS.ps_hosts)
    num_parameter_servers = len(ps_hosts)
    if num_parameter_servers <= 0:
        raise ValueError("Invalid num_parameter_servers value: %d" % 
                         num_parameter_servers)
    worker_hosts = re.findall(r'[\w\.:]+', FLAGS.worker_hosts)
    num_workers = len(worker_hosts)
    if FLAGS.task_index >= num_workers:
        raise ValueError("Worker index %d exceeds number of workers %d " % 
                         (FLAGS.task_index, num_workers))
    server = tf.train.Server({"ps":ps_hosts,"worker":worker_hosts}, job_name=FLAGS.job_name, task_index=FLAGS.task_index,protocol='grpc+verbs')

    #global BATCH 
    #BATCH = BATCH/num_workers
    print("GRPC URL: %s" % server.target)
    print("Task index = %d" % FLAGS.task_index)
    print("Number of workers = %d" % num_workers)
    print("Number of ps = %d" % num_parameter_servers)
    print("batch size = %d" % BATCH)

    if FLAGS.job_name == "ps":
        server.join()
    else:
        is_chief = (FLAGS.task_index == 0)

    if FLAGS.sync_replicas:
        if FLAGS.replicas_to_aggregate is None:
            replicas_to_aggregate = num_workers
        else:
            replicas_to_aggregate = FLAGS.replicas_to_aggregate

    # Construct device setter object #
    device_setter = get_device_setter(num_parameter_servers,
                                      num_workers)

    # The device setter will automatically place Variables ops on separate        #
    # parameter servers (ps). The non-Variable ops will be placed on the workers. #
    with tf.device(device_setter):
'''
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.name_scope('input'):
        # input #
        x_input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        # output #
        ans = tf.placeholder(shape=None, dtype=tf.float32)
        ans = tf.squeeze(tf.cast(ans, tf.float32))

    # use VGG16 network
    vgg = VGG16()
    # params for converting to answer-label-size
    w = tf.Variable(tf.truncated_normal([512, 10], 0.0, 1.0) * 0.01, name='w_last')
    b = tf.Variable(tf.truncated_normal([10], 0.0, 1.0) * 0.01, name='b_last')
        
    fmap = vgg.build(x_input, is_training=True)
    predict = tf.nn.softmax(tf.add(tf.matmul(fmap, w), b))
    loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(predict), reduction_indices=[1]))

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
    init_op = tf.global_variables_initializer()
    sv = tf.train.Supervisor(is_chief=is_chief,
                                 init_op=init_op,
                                 recovery_wait_secs=1,
                                 global_step=global_step)

    sess_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                     log_device_placement=FLAGS.log_device_placement,
                                     device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    sess_config.gpu_options.allow_growth = True
'''
    if FLAGS.sync_replicas and is_chief:
        chief_queue_runner = optimizer.get_chief_queue_runner()
        init_tokens_op = optimizer.get_init_tokens_op()
    init_op = tf.global_variables_initializer()
    sv = tf.train.Supervisor(is_chief=is_chief,
                            init_op=init_op,
                            recovery_wait_secs=1,
                            global_step=global_step)


        sess_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                     log_device_placement=FLAGS.log_device_placement,
                                     device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

        sess_config.gpu_options.allow_growth = True
        # The chief worker (task_index==0) session will prepare the session,   #
        # while the remaining workers will wait for the preparation to complete. #
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)

        sess = sv.prepare_or_wait_for_session(server.target,
                                              config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            print ("Starting chief queue runner and running init_tokens_op")
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)
'''
    ## Perform training ##
    print('\nSTART LEARNING')
    print('==================== '+str(datetime.datetime.now())+' ====================')
    time_begin = time.time()
    print("Training begins @ %s" % time.ctime(time_begin))

    # Training-loop
    lossbox = []
    local_step = 0
    for e in range(EPOCH):
        epoch_time_begin = time.time()
        for b in range(int(DATASET_NUM/BATCH)):
            batch, actuals = get_next_batch(train_images, train_labels, len(train_labels))
            _, step = sess.run([train_step, global_step], feed_dict={x_input: batch, ans: actuals})
            local_step += 1
            if b == int(DATASET_NUM/BATCH) - 1:
                print(str(sess.run(loss, feed_dict={x_input: batch, ans: actuals})).ljust(7)[:7] + '  ', end="")
            #if b%loss_loop == 0:
                #print('Batch: %4d' % int(b+1)+', \tLoss: '+str(sess.run(loss, feed_dict={x_input: batch, ans: actuals})))

        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_begin
        print("%8d     %3d        %5f        " % (step, e + 1, epoch_time), end="")
        test(sess, test_images, test_labels, predict, x_input)
        lossbox.append(sess.run(loss, feed_dict={x_input: batch, ans: actuals})) 

    print('==================== '+str(datetime.datetime.now())+' ====================')
    print('\nEND LEARNING')
    time_end = time.time()
    print("Training ends @ %s" % time.ctime(time_end))
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

if __name__ == "__main__":
  tf.app.run()
