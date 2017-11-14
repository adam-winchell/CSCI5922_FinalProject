'''
Histogram Loss Deep Embedding in Tensorflow
Python 3.6
'''
import sys
sys.path.append('./data/mnist')

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from read_mnist import MNIST

NUM_INPUT = 784
NUM_CLASSES = 10
H1 = 1024
H2 = 1024
H3 = 2048

LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 128

# Store layers weight & bias
weights = {
    'wd1': tf.Variable(tf.random_normal([NUM_INPUT, H1])),
    'wd2': tf.Variable(tf.random_normal([H1, H2])),
    'wd3': tf.Variable(tf.random_normal([H2, H3])),
    'wd4': tf.Variable(tf.random_normal([H3, NUM_CLASSES]))
}

biases = {
    'bd1': tf.Variable(tf.random_normal([H1])),
    'bd2': tf.Variable(tf.random_normal([H2])),
    'bd3': tf.Variable(tf.random_normal([H3])),
    'bd4': tf.Variable(tf.random_normal([NUM_CLASSES]))
}

# tf Graph input
x = tf.placeholder(tf.float32, [None, NUM_INPUT])
y = tf.placeholder(tf.int64, [None, NUM_CLASSES])

# Construct model
fc1 = tf.add(tf.matmul(x, weights['wd1']), biases['bd1'], name='fully_connected1')
fc1_nl = tf.nn.relu(fc1, name='fully_connected1_nl')

fc2 = tf.add(tf.matmul(fc1_nl, weights['wd2']), biases['bd2'], name='fully_connected2')
fc2_nl = tf.nn.relu(fc2, name='fully_connected2_nl')

fc3 = tf.add(tf.matmul(fc2_nl, weights['wd3']), biases['bd3'], name='fully_connected3')
fc3_nl = tf.nn.relu(fc3, name='fully_connected3_nl')

fc4 = tf.add(tf.matmul(fc3_nl, weights['wd4']), biases['bd4'], name='fully_connected4')
#fc4_nl = tf.nn.relu(fc4, name='fully_connected4_nl')

pred = fc4

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

train_images, train_labels, one_hot_train_labels = MNIST(sys.argv[1]).load_training()
test_images, test_labels, one_hot_test_labels = MNIST(sys.argv[1]).load_testing()

# Launch the graph
with tf.Session() as sess:

    x_train, x_valid, y_train, y_valid = train_test_split(train_images, one_hot_train_labels, test_size=0.16667)

    sess.run(init)

    for epoch in range(1, EPOCHS + 1):

        shuffle = np.random.permutation(len(y_train))
        x_train, y_train = x_train[shuffle], y_train[shuffle]

        for i in range(0, len(y_train), BATCH_SIZE):
            x_train_mb, y_train_mb = x_train[i:i + BATCH_SIZE], y_train[i:i + BATCH_SIZE]

            sess.run(optimizer, feed_dict={x: x_train_mb, y: y_train_mb})

            loss, acc = sess.run([cost, accuracy], feed_dict={x: x_train, y: y_train})
            print("Epoch " + str(epoch) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))

        # if epoch % 1 == 0 or epoch == 1:
        #     loss, acc = sess.run([cost, accuracy], feed_dict={x: x_train, y: y_train})
        #     print("Epoch " + str(epoch) + ", Minibatch Loss= " +
        #           "{:.6f}".format(loss) + ", Training Accuracy= " +
        #           "{:.5f}".format(acc))

    print('Optimization Finished')
