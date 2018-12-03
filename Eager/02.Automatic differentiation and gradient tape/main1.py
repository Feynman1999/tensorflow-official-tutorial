import tensorflow as tf 
import matplotlib.pyplot as plt
from math import pi

tf.enable_eager_execution()
tfe = tf.contrib.eager

def f(x):
    return tf.square(tf.sin(x))

print( f(pi/2).numpy() == 1.0 )
grad_f = tfe.gradients_function(f) # f的梯度函数
print(tf.abs(grad_f((pi/2,pi))).numpy())

# Higher-order gradients
def grad(f):  #相当于直接返回一个一阶梯度函数
    return lambda x : tfe.gradients_function(f)(x)[0]

x = tf.lin_space(-2*pi,2*pi,100) # 100 points between -2pi ~ 2pi

plt.plot(x, f(x), label = 'f')
plt.plot(x, grad(f)(x), label = 'first derivative')
plt.plot(x, grad(grad(f))(x), label = 'second derivative')
plt.plot(x, grad(grad(grad(f)))(x), label = 'third derivative')
plt.legend()
plt.show()


