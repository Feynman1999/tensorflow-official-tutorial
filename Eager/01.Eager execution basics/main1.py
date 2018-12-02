import tensorflow as tf 
import numpy as np
import timeit

tf.enable_eager_execution()

# Tensor objects have a data type and a shape
# TensorFlow offers a rich library of operations
# (tf.add, tf.matmul, tf.linalg.inv  etc.) that consume and produce Tensors.

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.encode_base64("hello world"))
print(tf.square(2)+tf.square(3))

x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

#The most obvious differences between NumPy arrays and TensorFlow Tensors are
# 1.Tensors can be backed by accelerator memory (like GPU, TPU).
# 2.Tensors are immutable.

'''
Conversion between TensorFlow Tensors and NumPy ndarrays is quite simple as:
TensorFlow operations automatically convert NumPy ndarrays to Tensors.
NumPy operations automatically convert Tensors to NumPy ndarrays.
'''
ndarray = np.ones((3,3))
print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray,42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

x=tf.random_uniform([3,3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

test="""
x = tf.random_uniform([1000,1000])
tf.matmul(x,x)
"""

setup="""import tensorflow as tf 
from __main__ import x
"""

#Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"): #指定cpu
    assert x.device.endswith("CPU:0")
    print(timeit.timeit(stmt=test, setup=setup,number=100))