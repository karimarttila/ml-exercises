import os
# Uncomment this if you want to supress tensorflow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print("You should see: Hello, TensorFlow!...")
print(sess.run(hello))

# Let's test some TensorFlow arithmetics:
# From: https://www.tensorflow.org/get_started/get_started
node1 = tf.constant(3)
node2 = tf.constant(4)
print("Printing nodes constant 3 and 4...")
print("You should see Tensor(... and node1 + node2 = 7")
print(node1, node2)
print("node1 + node2 = {0}".format(sess.run(node1 + node2)))

