'''
At times it may be inconvenient to encapsulate(封装) computation of 
interest into a function. For example, if you want the gradient
of the output with respect to intermediate(中间的) values computed in
the function. In such cases, the slightly more verbose but 
explicit(明确的)  **tf.GradientTape**  context is useful. All computation 
inside the context(上下文) of a **tf.GradientTape** is "recorded".
'''
import tensorflow as tf 

tf.enable_eager_execution()
tfe = tf.contrib.eager # shorthand for some symbols

x= tf.ones((2, 2))

with tf.GradientTape(persistent = True) as t: # persistent 持久的
    t.watch(x)
    y = tf.reduce_sum(x) # 就是求和的意思(降维)
    z = tf.multiply(y,y)


#use the same tape to compute the derivative of z with 
# respect to the intermediate value y
dz_dy = t.gradient(z, y) # 对y求导
print(dz_dy.numpy())


# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x) # 对x求导
print(dz_dx.numpy())


'''
higher-order gradients

Operations inside of the GradientTape context manager
are recorded for automatic differentiation. 

If gradients are computed in that context, 
then the gradient computation is recorded as well. 

As a result, the exact same API works for higher-order 
gradients as well. For example:
'''

x = tf.Variable(1.0)  # Convert the Python 1.0 to a Tensor object
with tf.GradientTape() as t:
    with tf.GradientTape() as t2: 
        y = x*x*x
        # Compute the gradient inside the 't' context manager
        # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y,x)
d2y_dx2 = t.gradient(dy_dx, x)
print(dy_dx.numpy(), d2y_dx2.numpy())










