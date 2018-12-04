import tensorflow as tf

tf.enable_eager_execution()

x = tf.zeros([10,10])
x += 2
print(x.numpy())

v = tf.Variable(1.0)
assert v.numpy() == 1.0

#Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0

#use 'v' in a TF opration like tf.square() and reassign
v.assign(tf.square(v))
print(v.numpy())

v = v * 2

print(v.numpy())