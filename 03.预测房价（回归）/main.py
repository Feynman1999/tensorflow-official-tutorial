import tensorflow as tf 
from tensorflow import keras
import numpy as np
import pandas as pd
from plot import plot_history
from matplotlib import pyplot as plt

'''
数据预处理
'''
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# shuffle the training set
# 生成train_labels.shape个0~1间的随机浮点数 
# 然后使用argsort 获得排序后对应原List中的id 那么由于之前是random的 
# 就相当于产生了一个随机排列
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training Set Size: {}".format(train_data.shape))
print("Testing Set Size: {}".format(test_data.shape))
print("第一个数据：\n",train_data[0])


# 使用 Pandas 库在格式规范的表格中显示数据集的前几行：
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']


#　DataFrame 类型类似于数据库表结构的数据结构，其含有行索引和列索引，
# 可以将DataFrame 想成是由相同索引的Series组成的Dict类型。
df = pd.DataFrame(train_data, columns=column_names)  
print(df.head())
# 下面查看标签（以千美元为单位）
print(train_labels[0:10])


'''
标准化特征 

虽然在未进行特征标准化的情况下，模型可能会收敛，但这样做会增
加训练难度，而且使生成的模型更加依赖于在输入中选择使用的单位。
'''
# 按照列求平均 （很自然）和 标准差
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print(train_data[0]) # First training sample , normalized


'''
构建模型
'''
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu,
                     input_shape = (train_data.shape[1],)))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer = optimizer,
                  metrics=['mae'])# mse:Mean squared error   mae:Mean Abs Error
    return model


model = build_model()
model.summary()
# 可见第一层有896: (13+1)*64 个参数   
# 第二层有4160: (64+1)*64个参数   
# 第三层有65: (64+1)*1个参数


'''
训练模型
'''
EPOCHS = 500

# Display training progress by 
# printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 ==0 : print('')  # 每一百个换行一次
        print('.',end='')

# The patience parameter is the amount of epochs to check for improvement
# 防止过拟合或者做无用功
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=EPOCHS/20)


# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, 
                    callbacks=[early_stop, PrintDot()])   #verbose 表示是否显示详细信息


'''
作LOSS图
'''
plot_history(history)


'''
在测试集上评估
'''
[loss, mae] = model.evaluate(test_data, test_labels, verbose=1)
print("\nTesting set Mean Abs Error: ${:7.2f}".format(mae * 1000))


'''
预测
'''
test_predictions = model.predict(test_data)
test_predictions = test_predictions.flatten(order='C') #将二维矩阵转为一维
# C means to flatten in row-major order   (C-style)  default
# F means to flatten in column-major order   (Fortran- style) 
# ‘A’ means to flatten in column-major order if a is Fortran contiguous
#  in memory, row-major order otherwise. ‘K’ means to flatten a in the 
# order the elements occur in memory. 

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100]) #参考线
plt.savefig('预测结果与真实值对比.png')

plt.figure()
error = test_predictions - test_labels
n,bins,patches = plt.hist(error, bins = 50) # 分成50块 查看每个error区间内对应的数量
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.savefig('预测误差.png')

print(type(n),type(bins),type(patches))
print(n,bins)

plt.show()

