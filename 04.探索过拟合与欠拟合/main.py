import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


'''
数据预处理
'''
NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
print(train_data.shape)

def multi_hot_sequences(sequences, dimension):
    # create an all_zero matrix of shape(len(sequences), dimension)
    results = np.zeros(  (len(sequences), dimension)  ) # 参数应该提供一个元组
    for i, word_indices in enumerate(sequences): # 可同时获得索引和值
        results[i,word_indices] = 1.0
    return results

train_data = multi_hot_sequences(train_data, NUM_WORDS)
test_data  = multi_hot_sequences(test_data, NUM_WORDS)
# plt.plot(train_data[0])
# plt.show()


'''
建立模型
'''
baseline_model = keras.Sequential([
    # 'input_shape' is only required here so that '.summary' works
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model.compile( optimizer = 'adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy','binary_crossentropy'])
baseline_model.summary()



smaller_model = keras.Sequential([
    keras.layers.Dense(4,activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy','binary_crossentropy'])
smaller_model.summary()



bigger_model =  keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1,   activation=tf.nn.sigmoid)
])
bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])
bigger_model.summary()



baseline_model_l2 = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001),
                        input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model_l2.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy','binary_crossentropy'])
baseline_model_l2.summary()



baseline_model_dropout = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model_dropout.compile(optimizer='adam',
                                loss='binary_crossentropy',
                                metrics=['accuracy','binary_crossentropy'])
baseline_model_dropout.summary()                                



'''
训练模型
'''
def train_model(model):
    history = model.fit(train_data,
                        train_labels,
                        epochs=20,
                        batch_size=512,
                        validation_data=(test_data, test_labels),
                        verbose=2) 
    return history

a= train_model(baseline_model)
# b= train_model(smaller_model)
# c= train_model(bigger_model)
# d = train_model(baseline_model_l2)
e = train_model(baseline_model_dropout)


'''
作图 查看模型效果
'''
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    for name,history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],'--',label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])


plot_history([('baseline',a),
              #('smaller_model',b),
              #('bigger_model',c),
              #('baseline_l2',d),
              ('baseline_dropout',e)])
plt.show()

