import tensorflow as tf

# graph를 만든다.
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
print(node1)
print(node2)
print(node3)

# sess.run()을 통해 그래프를 실행한다.
# 그래프의 변수를 업데이트하거나 값을 되돌려 받는다.
sess = tf.Session()
print(sess.run([node1, node2]))
print(sess.run(node3))
