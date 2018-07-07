import tensorflow as tf
import numpy as np

# need data-03-diabetes.csv
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
data_x = xy[:, 0:-1]
data_y = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1.+ tf.exp(tf.matmul(X, W) + b)
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/Loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# do not need tf.cast. So, I simplize this line
accuracy = tf.reduce_mean(tf.equal(predicted, Y))
                                  
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: data_x, Y: data_y})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: data_x, Y: data_y})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
