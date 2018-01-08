import os
# Uncomment this if you want to supress tensorflow warnings.
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

