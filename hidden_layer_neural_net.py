import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

original_data = fetch_mldata('MNIST original')
original_data
data = original_data['data']
data.shape
labels = original_data['target'].reshape(-1,1)
labels.shape
rand_choice = np.random.randint(0,70000)
plt.imshow(data[rand_choice].reshape(28,28), cmap='gray_r')
labels[rand_choice]



test_split = 0.1

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= test_split)
one_hot = OneHotEncoder(sparse = False)
y_train = one_hot.fit_transform(y_train)
y_test = one_hot.fit_transform(y_test)

tf.reset_default_graph()

with tf.name_scope('hyperparameters'):
    learning_rate = 0.01
    minibatch_size = 1000
    input_length = len(x_train[1])
    output_length = len(y_test[1])
    n_epochs = 500
    # Hidden layer sizes
    L1 = 300
    L2 = 50
    L3 = 10

with tf.name_scope('Placeholders'):
    x = tf.placeholder(shape =(None,inputs), dtype = tf.float64, name = 'x')
    y = tf.placeholder(shape =(None,outputs), dtype = tf.float64, name = 'y')

with tf.name_scope('Variables'):
    b_layer1 = tf.Variable(tf.random_normal(shape = (L1,), dtype = tf.float64), name = 'bias_layer1')
    W_layer1 = tf.Variable(tf.random_normal(shape = (input_length,L1), dtype = tf.float64), name = 'weights_layer1')

    b_layer2 = tf.Variable(tf.random_normal(shape = (L2,), dtype = tf.float64), name = 'bias_layer2')
    W_layer2 = tf.Variable(tf.random_normal(shape = (L1,L2), dtype = tf.float64), name = 'weights_layer2')

    b_layer3 = tf.Variable(tf.random_normal(shape = (L3,), dtype = tf.float64), name = 'bias_layer3')
    W_layer3 = tf.Variable(tf.random_normal(shape = (L2,L3), dtype = tf.float64), name = 'weights_layer3')


with tf.name_scope('Activations_Predictions'):
    layer1_activations = tf.nn.relu(tf.matmul(x,W_layer1) + b_layer1)
    layer2_activations = tf.nn.relu(tf.matmul(layer1_activations,W_layer2) + b_layer2)
    output_activations = tf.matmul(layer2_activations,W_layer3) + b_layer3
    y_prob = tf.nn.softmax(output_activations, axis = 1)
    pred_train = tf.argmax(y_prob, axis = 1)
    actual_train = tf.argmax(y, axis = 1)


with tf.name_scope('Metrics'):
        # test set
        layer1_activations = tf.nn.relu(tf.matmul(x,W_layer1) + b_layer1)
        layer2_activations = tf.nn.relu(tf.matmul(layer1_activations,W_layer2) + b_layer2)
        output_activations = tf.matmul(layer2_activations,W_layer3) + b_layer3
        y_prob_test = tf.nn.softmax(output_activations, axis = 1)
        pred_test = tf.cast(tf.argmax(y_prob_test, axis = 1), tf.float64)

        actual_test = tf.cast(tf.argmax(y, axis = 1), tf.float64)
        correct_train = tf.reduce_mean(tf.cast(tf.equal(pred_train, actual_train), tf.float64))
        correct_test = tf.reduce_mean(tf.cast(tf.equal(pred_test, actual_test), tf.float64))


with tf.name_scope('Loss_Function'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = output_activations, labels =y )
    l =tf.reduce_mean(loss)


with tf.name_scope('Optimizer'):
    trainer = tf.train.AdamOptimizer(learning_rate)
    train = trainer.minimize(l)

with tf.name_scope('Summary'):
    writer1 = tf.summary.FileWriter('/Users/jonathanbrink/Desktop/tensorboard/DNN')
    tf.summary.scalar('loss', l)
    tf.summary.scalar('training_accuracy', correct_train)
    #tf.summary.scalar('test_accuracy', correct_test)
    tf.summary.histogram('Layer 1 Weights', W_layer1)
    tf.summary.histogram('Layer 2 Weights', W_layer2)
    tf.summary.histogram('Layer 3 Weights', W_layer3)
    tf.summary.histogram('Layer 1 Biases', b_layer1)
    tf.summary.histogram('Layer 2 Biases', b_layer2)
    tf.summary.histogram('Layer 3 Biases', b_layer3)
    merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer1.add_graph(sess.graph)
    for i in range(n_epochs):
        batch_pos, batch_number = 0,0
        while batch_pos <= (len(x_train)/minibatch_size):
            feed_dict_train = {x: x_train[batch_pos:batch_pos+minibatch_size], y: y_train[batch_pos:batch_pos+minibatch_size]}
            feed_dict_test = {x: x_test, y: y_test}
            trained, loss_calc, correct_train1, summary = sess.run([train, loss, correct_train, merged_summary], feed_dict = feed_dict_train)
            correct_test1 = sess.run(correct_test, feed_dict = feed_dict_test)
            iteration = i*(len(x_train)/minibatch_size)+batch_number
            print('minibatch iteration: %s \ntraining_accuracy: %s \ntest_accuracy: %s \n' %(iteration,correct_train1*100, correct_test1*100))
            batch_pos += minibatch_size
            batch_number += 1
        writer1.add_summary(summary, iteration)
