'''
Example:Fitting a linear model

Let's now put the few concepts we have so far 

Tensor, Gradient Tape , Variable

to build and train a simple model 

this typically involves a few steps
1.define the model
2.define a loss function
3.Obtain training data
4.run through the training data and use an 'optimizer' to 
adjust the variables to fit the data
'''
import tensorflow as tf
import matplotlib.pyplot as plt 
tf.enable_eager_execution() # dont forget it


# create model
class Model(object):
    def __init__(self):
        # Initialize variable to(5.0 , 0.0)
        # In practice ,these should be initialized to random values
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()
# print(model(3.0).numpy())


# define loss
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


# make training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise 

# visualize 
plt.scatter(inputs,outputs,c='b',label='training data')
plt.scatter(inputs,model(inputs),c='r',label='model')
plt.legend()

print('current loss: ')
print(loss(model(inputs), outputs).numpy())


# define a training loop
'''
There are many variants of the gradient descent scheme
that are captured in tf.train.Optimizer implementations. 

in the spirit of building from first principles,
in this particular example we will implement the 
basic math ourselves.
'''
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t: 
        current_loss = loss(model(inputs), outputs)
    dW,db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)



'''
finally ,let's repeatly run through the training data and see how W and b evolve
'''
model = Model()

# Collect the history of W-values and b-values to plot later
Ws,bs=[],[]
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs),outputs)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % 
        (epoch,Ws[-1],bs[-1],current_loss))
    train(model, inputs, outputs, learning_rate = 0.1)


#plot it
plt.figure()
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W]*len(epochs), 'r--',
         [TRUE_b]*len(epochs), 'b--')
plt.legend(['W','b','true W','true b'])
plt.show()
