"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import t_ring.tensorflow as tr

import vgg19_trainable as vgg19
import utils
import numpy as np
img1 = utils.load_image("./test_data/tiger.jpeg")
img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

learning_rate = 0.05


tf.logging.set_verbosity(tf.logging.INFO)

batch1 = img1.reshape((1, 224, 224, 3))
'''
def get_next_batch(train_images, train_labels, data_len,batch_size = 1):
    actual = [img1_true_result]
    return batch1,actual
'''
def get_next_batch(batch_size=1):

        batch_images = []
        batch_labels = []

        for i in range(batch_size):
            # random class choice 
            # (randomly choose a folder of image of the same class from a list of previously sorted wnids)
            class_index = random.randint(0, 999)

            folder = wnid_labels[class_index]
            batch_images.append(img1)
            batch_labels.append(img1_true_result)
            print("%d" %i)

        np.vstack(batch_images)
        np.vstack(batch_labels)
        return batch_images, batch_labels
def main(_):
    tr.init()
    global_step = tf.Variable(0, name="global_step", trainable=False)
    with tf.name_scope('input'):
        sess = tf.Session()

        images = tf.placeholder(tf.float32, [1, 224, 224, 3])
        true_out = tf.placeholder(tf.float32, [1, 1000])
        train_mode = tf.placeholder(tf.bool)

        #vgg = vgg19.Vgg19('./vgg19.npy')
        vgg = vgg19.Vgg19()
        vgg.build(images, train_mode)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print(vgg.get_var_count())

        sess.run(tf.global_variables_initializer())

        # test classification
        prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        #utils.print_prob(prob[0], './synset.txt')


    print("ok5")
    with tf.name_scope('train'):
        loss = tf.reduce_sum((vgg.prob - true_out) ** 2)
        optimizer = tf.train.GradientDescentOptimizer(0.0001)
        optimizer =  tr.DistributedOptimizer(optimizer)
        train_step = optimizer.minimize(loss,  global_step = global_step)

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
    train_images = []
    train_labels = [] 
    print("ok 9")
    cnt = 0
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
        #for i in range(10000):
            # Run a training step synchronously.
            #print("start")
            #print(cnt)
            batch, actuals = get_next_batch(train_images, train_labels, len(train_labels))

            mon_sess.run(train_step, feed_dict={images: batch1, true_out: actuals, train_mode: True})
            cnt = cnt +1
            #print("FIN")
            #print(cnt)
            # test classification again, should have a higher probability about tiger
            #prob = mon_sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        print("DONE")
        print(cnt)

if __name__ == "__main__":
    print("OK1")
    tf.app.run()