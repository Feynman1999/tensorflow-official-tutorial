import os,pathlib
import tensorflow as tf 
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# -1表示我不想计算具体值 会自动填补 
print(train_images.shape)
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(28*28,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

times=1
checkpoint_path = "training_{}/cp.ckpt".format(times) #全路径
checkpoint_dir = os.path.dirname(checkpoint_path) #所在文件路径
# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

# model.fit(train_images,train_labels, epochs =10 ,
#             validation_data = (test_images, test_labels),
#             callbacks = [cp_callback])









#上述代码将创建一个 TensorFlow 检查点文件集合，这些文件在每个周期结束时更新：

#下面，重新构建一个未经训练的全新模型，并用测试集对其进行评估。
# 未训练模型的表现有很大的偶然性（准确率期望为 10%）：
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# 然后从检查点加载权重，并重新评估：
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))







# 更多的选项 控制每多少epoch设置一次检查点 并为生成的检查点提供独一无二的名称
# 训练一个新模型，每隔 5 个周期保存一次检查点并设置唯一名称：

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"  #全路径
checkpoint_dir = os.path.dirname(checkpoint_path) #所在文件路径
# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1,
                                              period=5)# Save weights, every 5-epochs.
model = create_model()
# model.fit(train_images,train_labels, epochs =50 ,
#             validation_data = (test_images, test_labels),
#             callbacks = [cp_callback],verbose=0)


# Sort the checkpoints by modification time.
checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
print(checkpoints,'\n',latest)
# 默认的 TensorFlow 格式仅保存最近的 5 个检查点。