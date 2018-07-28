import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import importlib.util
import os
from importlib import reload
import ml_graph

help(tf.data.Dataset)





data = fetch_mldata('Mnist original')
target = data['target']
images = data['data']
#images = (images - (255/2.0))/255
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
images = min_max_scaler.fit_transform(images)
images = images.reshape(-1,28,28,1)
target = target.reshape(-1,1)
target = OneHotEncoder(sparse = False).fit_transform(target)
target[1]

images[1]

###
#create generator functions here
###

tf.reset_default_graph()

#hyperparameters
with tf.name_scope('Hyperparameters'):
    Display_Step = 10
    learning_rate = 0.003
    minibatch_size = 1000
    target_accuracy = 0.99
    n_epochs = 10
    train_test_ratio = 10000
    pkeep_full = 0.75
    pkeep_conv = 1
    l1_input_channels = 1
    l1_filter_channels = 12
    l2_filter_channels = 24
    l3_filter_channels =36
    stride = 1


x_train, x_test, y_train, y_test =  train_test_split(images, target,
test_size = 10000, random_state = 42, stratify = target)



with tf.name_scope('Placeholders'):
    x = tf.placeholder(shape = (None, 28,28,1), dtype = tf.float64, name = 'x')
    y = tf.placeholder(shape = (None,10), dtype = tf.float64, name = 'y')
    dropout_full = tf.placeholder(dtype = tf.float64, name = 'keep_prob_full')
    dropout_conv = tf.placeholder(dtype = tf.float64, name = 'keep_prob_conv')

with tf.name_scope('Variables'):
    conv_weights_1 = tf.Variable(tf.truncated_normal((6,6,l1_input_channels,l1_filter_channels), dtype = tf.float64, stddev= 0.1), name = 'Convolutional_weights_1')
    conv_bias_1 = tf.Variable(tf.ones((l1_filter_channels), dtype = tf.float64)/100, name = 'Convolutional_Bias_1')

    conv_weights_2 = tf.Variable(tf.truncated_normal((5,4,l1_filter_channels,l2_filter_channels), dtype = tf.float64, stddev= 0.1), name = 'Convolutional_weights_2')
    conv_bias_2 = tf.Variable(tf.ones((l2_filter_channels), dtype = tf.float64)/100, name = 'Convolutional_Bias_2')

    conv_weights_3 = tf.Variable(tf.truncated_normal((4,4,l2_filter_channels,l3_filter_channels), dtype = tf.float64, stddev= 0.1), name = 'Convolutional_weights_3')
    conv_bias_3 = tf.Variable(tf.ones((l3_filter_channels), dtype = tf.float64)/100, name = 'Convolutional_Bias_3')

    connected_layer_weights1 = tf.Variable(tf.truncated_normal(shape = [(7*7*l3_filter_channels),600], dtype = tf.float64, stddev= 0.1), name = 'Fully_connected_Weights_1')
    connected_layer_biases1 = tf.Variable(tf.ones((600), dtype = tf.float64)/100, name = 'Fully_connected_Biases_1')

    connected_layer_weights2 = tf.Variable(tf.truncated_normal(shape = [600,300], dtype = tf.float64, stddev= 0.1), name = 'Fully_connected_Weights_2')
    connected_layer_biases2 = tf.Variable(tf.ones((300), dtype = tf.float64)/100, name = 'Fully_connected_Biases_2')

    connected_layer_weights3 = tf.Variable(tf.truncated_normal(shape = [300,100], dtype = tf.float64, stddev= 0.1), name = 'Fully_connected_Weights_3')
    connected_layer_biases3 = tf.Variable(tf.ones((100), dtype = tf.float64)/100, name = 'Fully_connected_Biases_3')

    connected_layer_weights4 = tf.Variable(tf.truncated_normal(shape = [100,50], dtype = tf.float64, stddev= 0.1), name = 'Fully_connected_Weights_4')
    connected_layer_biases4 = tf.Variable(tf.ones((50), dtype = tf.float64)/100, name = 'Fully_connected_Biases_4')

    connected_layer_weights5 = tf.Variable(tf.truncated_normal(shape = [50,10], dtype = tf.float64, stddev= 0.1), name = 'Fully_connected_Weights_5')
    connected_layer_biases5 = tf.Variable(tf.ones((10), dtype = tf.float64)/100, name = 'Fully_connected_Biases_5')


with tf.name_scope('Model'):

    conv_layer1 = tf.nn.conv2d(x, filter = conv_weights_1, strides = [1,1,1,1],
    padding = 'SAME', name = 'conv_layer1')
    conv_layer1 += conv_bias_1
    # check ordering of the max_pool and relu layers
    con_layer1 = tf.nn.dropout(conv_layer1, dropout_conv)
    conv_layer1 = tf.nn.relu(conv_layer1)

    conv_layer2 = tf.nn.conv2d(conv_layer1, filter = conv_weights_2, strides = [1,1,1,1],
    padding = 'SAME', name = 'conv_layer2')
    conv_layer2 += conv_bias_2
    # check ordering of the max_pool and relu layers
    conv_layer2 = tf.nn.max_pool(conv_layer2, ksize = [1,2,2,1],
    strides = [1,2,2,1], padding = 'SAME' )
    conv_layer2 = tf.nn.relu(conv_layer2)
    con_layer2 = tf.nn.dropout(conv_layer2, dropout_conv)

    conv_layer3 = tf.nn.conv2d(conv_layer2, filter = conv_weights_3, strides = [1,1,1,1],
    padding = 'SAME', name = 'conv_layer3')
    conv_layer3 += conv_bias_3
    # check ordering of the max_pool and relu layers
    conv_layer3 = tf.nn.max_pool(conv_layer3, ksize = [1,2,2,1],
    strides = [1,2,2,1], padding = 'SAME')
    conv_layer3 = tf.nn.relu(conv_layer3)
    con_layer3 = tf.nn.dropout(conv_layer3, dropout_conv)


    conv_layer3 = tf.reshape(conv_layer3, [-1,7*7*l3_filter_channels])

    fully_connected_layer1 = tf.matmul(conv_layer3,connected_layer_weights1, name = 'fully_connected_layer1') + connected_layer_biases1
    fully_connected_layer1 = tf.nn.relu(fully_connected_layer1)
    fully_connected_layer1 = tf.nn.dropout(fully_connected_layer1, dropout_full)

    fully_connected_layer2 = tf.matmul(fully_connected_layer1,connected_layer_weights2, name = 'fully_connected_layer2') + connected_layer_biases2
    fully_connected_layer2 = tf.nn.relu(fully_connected_layer2)
    fully_connected_layer2 = tf.nn.dropout(fully_connected_layer2, dropout_full)

    fully_connected_layer3 = tf.matmul(fully_connected_layer2,connected_layer_weights3, name = 'fully_connected_layer3') + connected_layer_biases3
    fully_connected_layer3 = tf.nn.relu(fully_connected_layer3)
    fully_connected_layer3 = tf.nn.dropout(fully_connected_layer3, dropout_full)

    fully_connected_layer4 = tf.matmul(fully_connected_layer3,connected_layer_weights4, name = 'fully_connected_layer4') + connected_layer_biases4
    fully_connected_layer4 = tf.nn.relu(fully_connected_layer4)
    fully_connected_layer4 = tf.nn.dropout(fully_connected_layer4, dropout_full)

    y_logits = tf.matmul(fully_connected_layer4, connected_layer_weights5, name = 'y_logits') + connected_layer_biases5

    y_prob = tf.nn.softmax(y_logits, axis = 1, name = 'y_prob')

    pred = tf.argmax(y_logits, axis = 1)

with tf.name_scope('Loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = y_logits, labels = y, name = 'loss')
    sum_loss = tf.reduce_sum(loss, name = 'sum_of_loss')
    mean_loss = tf.reduce_mean(loss, name = 'mean_loss')

with tf.name_scope('Optimizer'):
    trainer = tf.train.AdamOptimizer(learning_rate)
    train = trainer.minimize(sum_loss)

with tf.name_scope('Metrics'):
    true_y = tf.argmax(y, axis = 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true_y), tf.float64))

with tf.name_scope('Summary'):
    writer_1 = tf.summary.FileWriter('/Users/jonathanbrink/Desktop/tensorboard/Conv_nn')
    tf.summary.scalar('accuracy',accuracy)
    tf.summary.scalar('loss', mean_loss)
    merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    iteration_list = []


    writer_1.add_graph(sess.graph)

    batch_pos = 0

    feed_dict_train_acc = {x: x_train[:1000], y: y_train[:1000], dropout_conv: 1, dropout_full:1}

    feed_dict_test_acc = {x: x_test[1000:2000], y:y_test[1000:2000], dropout_conv: 1, dropout_full: 1}

    batch_iteration = 0


    for epoch in range(n_epochs):
        batch_pos = 0
        while batch_pos <= len(x_train):
            feed_dict_train = {x: x_train[batch_pos: batch_pos + minibatch_size], y: y_train[batch_pos: batch_pos + minibatch_size], dropout_conv: pkeep_conv, dropout_full: pkeep_full}
            train_outcome, training_loss1 = sess.run([train, mean_loss], feed_dict = feed_dict_train)
            batch_pos += minibatch_size
            batch_iteration += 1
            if batch_iteration % Display_Step == 0:
                LR = sess.run(trainer._lr_t)
                test_loss, test_accuracy = sess.run([mean_loss, accuracy], feed_dict = feed_dict_test_acc)
                train_accuracy = sess.run(accuracy, feed_dict = feed_dict_train_acc)
                print('Batch Iteration: %s \nTraining Accuracy: %s \nTest Accuracy: %s \nTrain Loss: %s \nTest Loss: %s \nLearning Rate: %s' %(batch_iteration, train_accuracy, test_accuracy, training_loss1, test_loss, LR))
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
                train_loss_list.append(training_loss1)
                test_loss_list.append(test_loss)
                iteration_list.append(batch_iteration)
                reload(ml_graph)
                ml_graph.training_graphs(test_accuracy_list, train_accuracy_list, test_loss_list, train_loss_list, iteration_list, target_accuracy=0.99, zoom = True)

                #plt.plot([train_loss_list,test_loss_list], batch_iteration)
            #writer_1.add_summary(summary1, batch_iteration)
