import tensorflow as tf 
tfe = tf.contrib.eager 

tf.enable_eager_execution()


layer1 = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to 
# specify it manually, which is useful in some complex models.
layer1 = tf.keras.layers.Dense(10, input_shape=(None, 5))


# simplely call it __all__ ?
layer1(tf.zeros([10,5]))# 第一维是样本个数 第二维是input_shape

# layers have many useful methods.
# for example ,you can inspect all variables in a layer 
# by calling ** layer.variables **   In this case a fully-connected 
# layer will have variables for weights and biases. 
print(layer1.variables)


# The variables are also accessible through nice accessors
print(layer1.kernel,'\n\n',layer1.bias)

