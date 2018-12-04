import tensorflow as tf 
tf.enable_eager_execution()

tfe = tf.contrib.eager

# the best way to implement your own layer is extending
# the tf.keras.Layer class and implementing: 
# *__init__, where you can do all input-independent initialization

# *build , where you know the shapes of the input tensors
# can do the rest of the initialization 

# *call , where you do the forward computation

'''
notice that you don't have to wati until build is called to
create your variables,you can also create them in __init__
However, the advantage of creating them in build is that it 
enables late variable creation based on the shape of the inputs
the layer will operate on.
On the other hand, creating variables in __init__ would mean that
shapes required to create the variables will need to be explicitly specified
'''

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
            shape=[input_shape[-1].value,self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


layer1 = MyDenseLayer(10) 
# 给过输入后自动确定layer的variables的shape
print(layer1(tf.zeros([10,5])))
print(layer1.variables)
# 当然，不是必须这样 ，你可以直接在init里确定variables的shape

'''
Overall code is easier to read and maintain if it uses
standard layers whenever possible, as other readers 
will be familiar with the behavior of standard layers.
If you want to use a layer which is not present in 
tf.keras.layers or tf.contrib.layers, consider filing
a github issue or, even better, sending us a pull 
request!
'''

