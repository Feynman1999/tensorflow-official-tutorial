'''
Gradient tapes
'''
import tensorflow as tf 

tf.enable_eager_execution()
tfe = tf.contrib.eager # shorthand for some symbols


# x^y
def f(x,y): 
    output = 1
    # Must use range(int(y)) instead of range(y) in Python 3 when
    # using TensorFlow 1.10 and earlier. Can use range(y) in 1.11+
    for i in range(int(y)): # you can use for loop (#^.^#)
        output = tf.multiply(output, x)
    return output         


# d x^y / d x
def g(x,y):
    # Return the gradient of 'f' with respect to it's first parameter  default?
    return tfe.gradients_function(f)(x,y)[0]


print( f(3,2).numpy() )
print( g(3.0,2).numpy() )
print( f(3,3).numpy() )
print( g(3.0,3).numpy() )

