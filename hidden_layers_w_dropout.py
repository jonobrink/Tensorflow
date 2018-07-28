import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

original_data = fetch_mldata('MNIST original')
original_data
data = original_data['data']
data.shape
data[1]

#preprocessing
scaler = preprocessing.MaxAbsScaler()
data=scaler.fit_transform(data)
data[1]


labels = original_data['target'].reshape(-1,1)
labels.shape
rand_choice = np.random.randint(0,70000)
plt.imshow(data[rand_choice].reshape(28,28), cmap='gray_r')
labels[rand_choice]

test_split = 0.01

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= test_split)
one_hot = OneHotEncoder(sparse = False)
y_train = one_hot.fit_transform(y_train)
y_test = one_hot.fit_transform(y_test)

tf.reset_default_graph()

with tf.name_scope('hyperparameters'):
    Display_Step = 100
    learning_rate = 0.003
    minibatch_size = 100
    input_length = len(x_train[1])
    output_length = len(y_train[1])
    n_epochs = 5000
    dropout_prob = 0.75
    # Hidden layer sizes
    L1 = 200
    L2 = 100
    L3 = 60
    L4 = 30
    L5 = 10

with tf.name_scope('Placeholders'):
    x = tf.placeholder(shape =(None,input_length), dtype = tf.float64, name = 'x')
    y = tf.placeholder(shape =(None,output_length), dtype = tf.float64, name = 'y')
    pkeep = tf.placeholder(tf.float64, name = 'dropout_prob')

with tf.name_scope('Variables'):
    b_layer1 = tf.Variable(tf.zeros(shape = (L1), dtype = tf.float64), name = 'bias_layer1')
    W_layer1 = tf.Variable(tf.truncated_normal(shape = (input_length,L1), dtype = tf.float64, stddev = 0.1) , name = 'weights_layer1')

    b_layer2 = tf.Variable(tf.zeros(shape = (L2), dtype = tf.float64), name = 'bias_layer2')
    W_layer2 = tf.Variable(tf.truncated_normal(shape = (L1,L2), dtype = tf.float64, stddev = 0.1) , name = 'weights_layer2')

    b_layer3 = tf.Variable(tf.zeros(shape = (L3), dtype = tf.float64), name = 'bias_layer3')
    W_layer3 = tf.Variable(tf.truncated_normal(shape = (L2,L3), dtype = tf.float64, stddev = 0.1) , name = 'weights_layer3')

    b_layer4 = tf.Variable(tf.zeros(shape = (L4), dtype = tf.float64), name = 'bias_layer3')
    W_layer4 = tf.Variable(tf.truncated_normal(shape = (L3,L4), dtype = tf.float64, stddev = 0.1) , name = 'weights_layer3')

    b_layer5 = tf.Variable(tf.zeros(shape = (L5), dtype = tf.float64), name = 'bias_layer3')
    W_layer5 = tf.Variable(tf.truncated_normal(shape = (L4,L5), dtype = tf.float64, stddev = 0.1) , name = 'weights_layer3')


with tf.name_scope('Activations_Predictions'):
    Y1 = tf.nn.relu(tf.matmul(x,W_layer1) + b_layer1)
    Y1 = tf.nn.dropout(Y1, pkeep)
    Y2 = tf.nn.relu(tf.matmul(Y1,W_layer2) + b_layer2)
    Y2 = tf.nn.dropout(Y2, pkeep)
    Y3 = tf.nn.relu(tf.matmul(Y2,W_layer3) + b_layer3)
    Y3 = tf.nn.dropout(Y3, pkeep)
    Y4 = tf.nn.relu(tf.matmul(Y3,W_layer4) + b_layer4)
    Y4 = tf.nn.dropout(Y4, pkeep)
    Y5 = tf.matmul(Y4,W_layer5) + b_layer5
    y_prob = tf.nn.softmax(Y5)


with tf.name_scope('Metrics'):
        actual_labels = tf.argmax(y, axis = 1)
        pred = tf.argmax(y_prob, axis = 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, actual_labels), tf.float64))

with tf.name_scope('Loss_Function'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = Y5, labels = y)
    l =tf.reduce_mean(loss)


with tf.name_scope('Optimizer'):
    trainer = tf.train.AdamOptimizer(learning_rate)
    train = trainer.minimize(l)

with tf.name_scope('Summary'):
    writer1 = tf.summary.FileWriter('/Users/jonathanbrink/Desktop/tensorboard/DNN')
    tf.summary.scalar('loss', l)
    tf.summary.scalar('training_accuracy', correct)
    #tf.summary.scalar('test_accuracy', correct_test)
    tf.summary.histogram('Layer 1 Weights', W_layer1)
    tf.summary.histogram('Layer 2 Weights', W_layer2)
    tf.summary.histogram('Layer 3 Weights', W_layer3)
    tf.summary.histogram('Layer 4 Weights', W_layer4)
    tf.summary.histogram('Layer 5 Weights', W_layer5)
    tf.summary.histogram('Layer 1 Biases', b_layer1)
    tf.summary.histogram('Layer 2 Biases', b_layer2)
    tf.summary.histogram('Layer 3 Biases', b_layer3)
    tf.summary.histogram('Layer 4 Biases', b_layer4)
    tf.summary.histogram('Layer 5 Biases', b_layer5)
    merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer1.add_graph(sess.graph)
    for i in range(n_epochs):
        batch_pos, batch_number = 0,0
        while batch_pos <= (len(x_train)/minibatch_size):
            iteration = i*(len(x_train)/minibatch_size)+batch_number
            feed_dict_train = {x: x_train[batch_pos:batch_pos+minibatch_size], y: y_train[batch_pos:batch_pos+minibatch_size], pkeep : dropout_prob}
            feed_dict_train_no_drop = {x: x_train, y: y_train, pkeep : 1}
            feed_dict_test = {x: x_test, y: y_test, pkeep : 1}
            trained, loss_calc, summary = sess.run([train, loss, merged_summary], feed_dict = feed_dict_train)
            if iteration%Display_Step == 0:
                correct_train1 = sess.run(accuracy, feed_dict = feed_dict_train_no_drop)
                correct_test1 = sess.run(accuracy, feed_dict = feed_dict_test)
                print('minibatch iteration: %s \ntraining_accuracy: %s \ntest_accuracy: %s \n' %(iteration,correct_train1*100, correct_test1*100))
            batch_pos += minibatch_size
            batch_number += 1
        writer1.add_summary(summary, iteration)
