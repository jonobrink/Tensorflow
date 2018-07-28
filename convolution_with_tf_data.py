import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

data = fetch_mldata('Mnist original')
target = data['target']
images = data['data']
images = MinMaxScaler().fit_transform(images)
images = images.reshape(-1,28,28,1)
target = target.reshape(-1,1)
target = OneHotEncoder(sparse = False).fit_transform(target)
target[1]

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)


###
#create generator functions here
###

tf.reset_default_graph()

#hyperparameters
with tf.name_scope('Hyperparameters'):
    Display_Step = 100
    learning_rate = 0.003
    minibatch_size = 1000
    n_epochs = 50
    train_test_ratio = 10000
    pkeep = 0.5
    l1_input_channels = 1
    l1_filter_channels = 16
    l2_input_channels = 16
    l2_filter_channels = 32
    stride = 1


x_train, x_test, y_train, y_test =  train_test_split(images, target,
test_size = 10000, random_state = 42, stratify = target)



with tf.name_scope('Placeholders'):
    x = tf.placeholder(shape = (None, 28,28,1), dtype = tf.float64, name = 'x')
    y = tf.placeholder(shape = (None,10), dtype = tf.float64, name = 'y')
    dropout = tf.placeholder(dtype = tf.float64, name = 'keep_prob')


with tf.name_scope('Variables'):
    conv_weights_1 = tf.Variable(tf.truncated_normal((5,5,1,10), dtype = tf.float64), name = 'Convolutional_weights_1')
    conv_bias_1 = tf.Variable(tf.zeros((10), dtype = tf.float64), name = 'Convolutional_Bias_1')

    conv_weights_2 = tf.Variable(tf.truncated_normal((5,5,10,10), dtype = tf.float64), name = 'Convolutional_weights_2')
    conv_bias_2 = tf.Variable(tf.zeros((10), dtype = tf.float64), name = 'Convolutional_Bias_2')

    connected_layer_weights1 = tf.Variable(tf.truncated_normal(shape = [(7*7*10),200], dtype = tf.float64), name = 'Fully_connected_Weights_1')
    connected_layer_biases1 = tf.Variable(tf.zeros((200), dtype = tf.float64), name = 'Fully_connected_Biases_1')

    connected_layer_weights2 = tf.Variable(tf.truncated_normal(shape = [200,10], dtype = tf.float64), name = 'Fully_connected_Weights_2')
    connected_layer_biases2 = tf.Variable(tf.zeros((10), dtype = tf.float64), name = 'Fully_connected_Biases_2')


with tf.name_scope('Model'):

    conv_layer1 = tf.nn.conv2d(x, filter = conv_weights_1, strides = [1,1,1,1],
    padding = 'SAME', name = 'conv_layer1')
    conv_layer1 += conv_bias_1
    # check ordering of the max_pool and relu layers
    conv_layer1 = tf.nn.max_pool(conv_layer1, ksize = [1,2,2,1],
    strides = [1,2,2,1], padding = 'SAME')
    conv_layer1 = tf.nn.relu(conv_layer1)
    #con_layer1 = tf.nn.dropout(conv_layer1, dropout)

    conv_layer2 = tf.nn.conv2d(conv_layer1, filter = conv_weights_2, strides = [1,1,1,1],
    padding = 'SAME', name = 'conv_layer2')
    conv_layer2 += conv_bias_2
    # check ordering of the max_pool and relu layers
    conv_layer2 = tf.nn.max_pool(conv_layer2, ksize = [1,2,2,1],
    strides = [1,2,2,1], padding = 'SAME' )
    conv_layer2 = tf.nn.relu(conv_layer2)
    #con_layer2 = tf.nn.dropout(conv_layer2, dropout)

    conv_layer2 = tf.reshape(conv_layer2, [-1,7*7*10])

    fully_connected_layer1 = tf.matmul(conv_layer2,connected_layer_weights1, name = 'fully_connected_layer1') + connected_layer_biases1
    fully_connected_layer1 = tf.nn.dropout(fully_connected_layer1, dropout)

    y_logits = tf.matmul(fully_connected_layer1, connected_layer_weights2, name = 'y_logits') + connected_layer_biases2

    y_prob = tf.nn.softmax(y_logits, axis = 1, name = 'y_prob')

    pred = tf.argmax(y_logits, axis = 1)

with tf.name_scope('Loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = y_logits, labels = y, name = 'loss')
    mean_loss = tf.reduce_mean(loss, name = 'mean_loss')

with tf.name_scope('Optimizer'):
    trainer = tf.train.AdamOptimizer(learning_rate)
    train = trainer.minimize(mean_loss)

with tf.name_scope('Metrics'):
    true_y = tf.argmax(y, axis = 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true_y), tf.float64))

with tf.name_scope('Summary'):
    writer_1 = tf.summary.FileWriter('/Users/jonathanbrink/Desktop/tensorboard/Conv_nn')
    #tf.summary.scalar('accuracy',accuracy)
    tf.summary.scalar('loss', mean_loss)
    tf.summary.histogram('conv_weights_1', conv_weights_1)
    tf.summary.histogram('conv_weights_2', conv_weights_2)
    tf.summary.histogram('conv_bias_1', conv_bias_1)
    tf.summary.histogram('conv_bias_2', conv_bias_2)
    tf.summary.histogram('connected_layer_weights1', connected_layer_weights1)
    tf.summary.histogram('connected_layer_weights2', connected_layer_weights2)
    tf.summary.histogram('connected_layer_bias1', connected_layer_biases1)
    tf.summary.histogram('connected_layer_bias2', connected_layer_biases2)

    merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    writer_1.add_graph(sess.graph)

    batch_pos = 0

    batch_X, batch_Y = mnist.train.next_batch(minibatch_size)

    feed_dict_train = {x: batch_X, y: batch_Y, dropout: pkeep}

    feed_dict_train_acc = {x: batch_X, y: batch_Y, dropout: 1}

    feed_dict_test_acc = {x: mnist.test.images , y:mnist.test.labels, dropout: 1}

    batch_iteration = 0

    for epoch in range(n_epochs):
        batch_pos = 0
        while batch_pos <= len(x_train):
            train_outcome, loss1, summary1 = sess.run([train, mean_loss, merged_summary], feed_dict = feed_dict_train)
            print(loss1)
            if batch_iteration % Display_Step == 0:
                test_accuracy = sess.run(accuracy, feed_dict = feed_dict_test_acc)
                train_accuracy = sess.run(accuracy, feed_dict = feed_dict_train_acc)
                print('Batch Iteration: %s \nTraining Accuracy: %s \nTest Accuracy: %s' %(batch_iteration, train_accuracy, test_accuracy))
            writer_1.add_summary(summary1, batch_iteration)
            batch_pos += minibatch_size
            batch_iteration += 1
