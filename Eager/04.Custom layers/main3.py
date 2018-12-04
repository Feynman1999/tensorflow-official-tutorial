'''
Many interesting layer-like things in machine learning models
are implemented by composing existing layers. For example, each
residual block in a resnet is a composition of convolutions, 
batch normalizations, and a shortcut.

The main class used when creating a layer-like thing which contains
other layers is ** tf.keras.Model ** Implementing one is done by 
inheriting from tf.keras.Model.
'''
import tensorflow as tf 
tf.enable_eager_execution()

tfe = tf.contrib.eager

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):   
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1 , filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1,1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1,1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor , training = False):
        x = self.conv2a(input_tensor)
        print(x.numpy(),end='\n\n')
        x = self.bn2a(x, training = training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        print(x.numpy(),end='\n\n')
        x = self.bn2b(x, training = training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        print(x.numpy(),end='\n\n')
        x = self.bn2c(x, training = training)

        x += input_tensor # 维度相同
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1,[2,2,1])
print(block(tf.zeros(
    [1,2,3,3]
)))

print(x.name for x in block.variables)


'''
Much of the time, however, models which compose many 
layers simply call one layer after the other. This can 
be done in very little code using tf.keras.Sequential
'''
my_seq = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1,(1,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(2,1,padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(3,(1,1)),
    tf.keras.layers.BatchNormalization()
])

my_seq(tf.zeros([1,2,3,3]))