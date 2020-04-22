import keras
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime
from keras.models import load_model
from models.pretrained import vgg16_based_model
from models.summary import summarize_history
from preprocessing import create_testing_data, create_training_data

CHANNELS = 3
IMAGE_SIZE = 224
LEARNING_RATE = 1e-3
TEST_DIR = './datasets/test'
TEST_FILE = 'testing_data.npy'
TRAIN_DIR = './datasets/train'
TRAIN_FILE = 'training_data.npy'

if __name__ == '__main__':
    '''Training'''
    log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = keras.callbacks.TensorBoard(log_dir=log_dir)

    training_data_file = os.path.join(TRAIN_DIR, TRAIN_FILE)

    if os.path.exists(training_data_file):
        training_data = np.load(training_data_file, allow_pickle=True)
        print('File {} loaded'.format(TRAIN_FILE))
    else:
        training_data = create_training_data(TRAIN_DIR, TRAIN_FILE, IMAGE_SIZE)

    vgg16_name = vgg16_based_model.__name__
    model_name = '{}.h5'.format(vgg16_based_model.__name__)

    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        model = vgg16_based_model(IMAGE_SIZE, CHANNELS)

        training = training_data[:22500]
        validation = training_data[22500:]

        X_train = np.array([i[0] for i in training])
        y_train = np.array([i[1] for i in training])

        X_validation = np.array([i[0] for i in validation])
        y_validation = np.array([i[1] for i in validation])

        history = model.fit(X_train, y_train,
                            batch_size=64,
                            epochs=3,
                            validation_data=(X_validation, y_validation),
                            shuffle=True,
                            verbose=1,
                            callbacks=[callbacks])

        model.save(model_name)

        summarize_history(history, vgg16_name)

    '''Testing'''
    testing_data_file = os.path.join(TEST_DIR, TEST_FILE)

    if os.path.exists(testing_data_file):
        testing_data = np.load(testing_data_file, allow_pickle=True)
        print('File {} loaded'.format(TEST_FILE))
    else:
        testing_data = create_testing_data(TEST_DIR, TEST_FILE, IMAGE_SIZE)

    figure = plt.figure()

    for number, data in enumerate(testing_data[:12]):
        image_number = data[1]
        image_data = data[0]

        y = figure.add_subplot(3, 4, (number + 1))
        data = np.array(image_data)
        image = image_data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
        prediction = model.predict([image])[0]

        if np.argmax(prediction) == 1:
            label = 'Dog'
        else:
            label = 'Cat'

        plt.title(label)

        y.imshow(data)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.savefig('{}_predictions.png'.format(vgg16_name))
    plt.show()
