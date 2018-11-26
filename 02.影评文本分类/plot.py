import matplotlib.pyplot as plt

def plot(history_dict):
    # 一共有 4 个条目：每个条目对应训练和验证期间的一个受监控指标。
    # 我们可以使用这些指标绘制训练损失与验证损失图表以进行对比，并绘制训练准确率与验证准确率图表：
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1,len(acc)+1)

    plt.figure(1)
    # "bo" is for blue dot
    plt.plot(epochs,loss,'bo',label='Training Loss')
    # "b" is for "solid blue line"
    plt.plot(epochs,val_loss,'b',label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.figure(2)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()