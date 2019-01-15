from __future__ import absolute_import , division , print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np

tf.enable_eager_execution()
tfe = tf.contrib.eager
print("Tensorflow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))


# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']

feature_names = column_names[:-1] 
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

'''
每个标签都分别与一个字符串名称（例如“setosa”）相关联，但机器学习通常依赖于数字值。标签编号会映射到一个指定的表示法，例如：
0：山鸢尾
1：变色鸢尾
2：维吉尼亚鸢尾
'''
class_names = ['Iris setosa','Iris versicolor', 'Iris virginica']

'''
创建一个 tf.data.Dataset

TensorFlow 的 Dataset API 可处理在向模型加载数据时遇到的许多常见情况。
这是一种高阶 API，用于读取数据并将其转换为可供训练使用的格式。
'''
batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp, # 前面得到过的s
    batch_size, # 很关键
    column_names = column_names,
    label_name = label_name,
    num_epochs = 1
)
# print(type(train_dataset))
# print(type(next(iter(train_dataset))))
features , labels = next(iter(train_dataset))

# print(features)
print(labels[0:5].numpy())

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap = 'viridis')# cmap控制颜色的系列

plt.xlabel("Petal length")
plt.ylabel("Sepal length")



'''
要简化模型构建步骤，请创建一个函数以将特征字典重新打包为形状为 
(batch_size, num_features) 的单个数组。
'''
#print(list(features.values()))
def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()),axis=1) #按照列stack 很自然
    return features , labels

'''
然后使用 tf.data.Dataset.map 方法将每个 
(features,label) 对的 features 打包到训练数据集中：
'''
train_dataset = train_dataset.map(pack_features_vector)

'''
Dataset 的 features 元素现在是形状为 
(batch_size, num_features) 的数组。
我们来看看前5个样本：
'''
features, labels = next(iter(train_dataset)) #next取一个batch?
print(features[:5])



'''
select model
'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = tf.nn.relu, input_shape=(4,)),
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dense(3)
])

# predictions = model(features)

# print("Prediction: {}".format(tf.argmax(predictions, axis=1))) #对 每一行 取最大的值的下标作为预测的类别
# print("    Labels: {}".format(labels))

def loss(model, x, y):
    y_ = model(x)
    # 注意使用softmax
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

# l = loss(model, features, labels)
# print("Loss test: {}".format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape: 
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
global_step = tf.train.get_or_create_global_step()

loss_value , grads = grad(model,features,labels)
print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))
optimizer.apply_gradients(zip(grads,model.variables), global_step)
print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))


# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 256

for epoch in range(num_epochs):  # [0,1,...,199,200]
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    # Training loop - using batches of 32  一个
    for x,y in train_dataset: # train_dataset的元素是一个batch
        # Optimize the model
        loss_value , grads = grad(model,x,y)
        # 对于每一个batch 做一次梯度下降
        optimizer.apply_gradients(zip(grads, model.variables),global_step)
        # Track progress
        epoch_loss_avg(loss_value) # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end of epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 20 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                            epoch_loss_avg.result(),
                                                            epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()