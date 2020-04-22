import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def summarize_history(history, model_name):
    figure, axes = plt.subplots(2, 1)

    axes[0].set_title('Cross Entropy Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].plot(history.history['loss'], color='black', label='train')
    axes[0].plot(history.history['val_loss'], color='red', label='val')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].set_title('Classification Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].plot(history.history['accuracy'], color='black', label='train')
    axes[1].plot(history.history['val_accuracy'], color='blue', label='val')
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    figure.tight_layout()

    plt.savefig('{}_history_summary.png'.format(model_name))
