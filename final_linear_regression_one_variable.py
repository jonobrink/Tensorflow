import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sample_size = 100000

x_np = np.random.normal(0,1,(sample_size,1))
y_np = 2 + 5*x_np + np.random.normal(0,1,(sample_size,1))
y_np=np.reshape(y_np,(sample_size,1))



#hyperparameters
learning_rate = 0.01
n_epochs = 30
minibatch_size = 1000


tf.reset_default_graph()

with tf.name_scope('Placeholders'):
    x = tf.placeholder(shape=(minibatch_size,1), dtype = tf.float64, name = 'x')
    y = tf.placeholder(shape=(minibatch_size,1), dtype = tf.float64, name = 'y')

with tf.name_scope('Variables'):
    b = tf.Variable(tf.random_normal(shape=(1,), dtype = tf.float64), name = 'b')
    w = tf.Variable(tf.random_normal(shape=(1,1), dtype = tf.float64), name = 'w')

with tf.name_scope('prediction'):
    y_pred = tf.matmul(x,w) + b

with tf.name_scope('Loss'):
    loss = tf.reduce_sum((y - y_pred)**2)

with tf.name_scope('Optimizer'):
    optimizer=tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

with tf.name_scope('Summary'):
    writer = tf.summary.FileWriter('/Users/jonathanbrink/Desktop/tensorboard/lin_reg')
    with tf.Session() as sess:
        writer.add_graph(sess.graph)
    tf.summary.histogram('weight', w)
    tf.summary.histogram('bias', b)
    tf.summary.scalar('loss', loss)
    merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    #filewrite = tf.summary.FileWriter('/Users/jonathanbrink/Desktop/tensorboard/test', sess.graph)
    #sess.run(c)
    sess.run(tf.global_variables_initializer())
    for n in range(n_epochs):
        run = 0
        while run < sample_size:
            feed = {x : x_np[run:run+minibatch_size], y : y_np[run:run+minibatch_size]}
            train_result, loss1, summary = sess.run([train, loss, merged_summary], feed_dict = feed)
            weight = sess.run(w)
            bias = sess.run(b)
            #loss1 = sess.run(loss)
            writer.add_summary(summary)
            print('run: %d, loss: %s , weight: %s, bias: %s' %(run+(n*sample_size), loss1 ,weight,bias))
            run+=minibatch_size

sess.close()

help(tf.summary.FileWriter)
help(tf.Variable)
help(writer.add_summary)
