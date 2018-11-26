import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
    Dict = history.history
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(Dict['mean_absolute_error']),
                label='Train_Loss')
    plt.plot(history.epoch, np.array(Dict['val_mean_absolute_error']),
                label='Val_Loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.savefig('训练过程.png')
