from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np




mnist = fetch_mldata('MNIST original')
mnist.keys()
data = mnist['data']
target = mnist['target'].reshape(len(mnist['target']), 1)

plt.imshow(data[np.random.randint(1,len(data))].reshape(28,28), cmap='gray_r')

encoder = OneHotEncoder()
new_y = encoder.fit_transform(target).toarray()


#constants
im_height = 28
im_width = 28
pixels = im_height*im_width
channels = 1

categories = 10


# hyperparameters
learning_rate = 0.01
n_epochs = 80
minibatch_size = 1000
train_test_split_ratio =0.1
n_hidden = 1
dropout_p = 0.5


X_train, X_test, y_train, y_test = train_test_split(data, new_y, test_size = train_test_split_ratio)


tf.reset_default_graph()

with tf.name_scope('placeholders'):
    x_training = tf.placeholder(shape = (minibatch_size, pixels), dtype = tf.float64)
    y_training = tf.placeholder(shape = (minibatch_size, 10), dtype = tf.float64)
    x_testing = tf.placeholder(shape = (len(X_test), pixels), dtype = tf.float64)
    y_testing = tf.placeholder(shape = (len(y_test), 10), dtype = tf.float64)

with tf.name_scope('weights_and_biases'):
    W = tf.Variable(tf.random_normal((pixels,10), dtype = tf.float64), name = 'Weight_Vector')
    b = tf.Variable(tf.random_normal((10,), dtype = tf.float64), name = 'Bias_Vector')

with tf.name_scope('prediction'):
    y_logit_train = tf.matmul(x_training, W) + b
    pred_train = tf.argmax(y_logit_train, axis=1, output_type = tf.int32)
    y_true_train = tf.argmax(y_training, axis = 1, output_type = tf.int32)
    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_train, pred_train), tf.float64))

    y_logit_test = tf.matmul(x_testing, W) + b
    pred_test = tf.argmax(y_logit_test, axis=1, output_type = tf.int32)
    y_true_test = tf.argmax(y_testing, axis = 1, output_type = tf.int32)
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_test, pred_test), tf.float64))

with tf.name_scope('loss'):
    l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_training, logits = y_logit_train, name = 'loss'))

with tf.name_scope('optimizer'):
    train_ml=tf.train.AdamOptimizer(learning_rate).minimize(l)

with tf.name_scope('summaries'):
    writer = tf.summary.FileWriter('/Users/jonathanbrink/Desktop/tensorboard/neural_network')

    with tf.Session() as sess:
        writer.add_graph(sess.graph)
    tf.summary.histogram('weights', W)
    tf.summary.histogram('biases', b)
    tf.summary.scalar('loss', l)
    tf.summary.scalar('train_accuracy', train_accuracy)
    tf.summary.scalar('test_accuracy', test_accuracy)
    merged_sum = tf.summary.merge_all()



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        step = 0
        while step < len(X_train):
            x_batch, y_batch = X_train[step:step+minibatch_size], y_train[step:step+minibatch_size]
            feed_dict={x_training: x_batch, y_training: y_batch, x_testing:X_test , y_testing:y_test}
            training, summary, loss21, train_acc, test_acc = sess.run([train_ml, merged_sum, l, train_accuracy, test_accuracy], feed_dict = feed_dict)
            writer.add_summary(summary, (epoch*len(X_train) + step))
            step+=minibatch_size
            print('step: %s, epoch: %s, loss: %s, train_accuracy: %s, test_accuracy: %s' %(step, epoch, loss21, train_acc, test_acc))


help(tf.argmax)
help(tf.equal)
help(tf.to_int32)
help(tf.summary.scalar)
help(tf.cast)
