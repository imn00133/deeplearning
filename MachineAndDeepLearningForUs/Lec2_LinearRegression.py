import tensorflow as tf

# X and Y data
train_x = [1, 2, 3]
train_y = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis WX+b
hypothesis = train_x * W + b

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - train_y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 세션에서 그래프 실행
sess = tf.Session()

# 그래프 내 글로벌 변수 초기화
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
