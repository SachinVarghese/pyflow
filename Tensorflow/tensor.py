import tensorflow as tf

x1 = tf.constant(5)

x2 = tf.constant(6)

x3 = x1 * x2

# x3 = tf.multiply(x1, x2)

with tf.Session() as sess:
    output = sess.run(x3)
    print(output)
    # sess.close()

print(output)
