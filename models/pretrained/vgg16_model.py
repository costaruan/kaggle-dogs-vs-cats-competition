from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD


def vgg16_based_model(size, channels):
    model = VGG16(include_top=False,
                  input_shape=(size, size, channels))

    for layer in model.layers:
        layer.trainable = False

    flatten = Flatten()(model.layers[-1].output)
    fully_connected = Dense(units=128,
                            activation='relu',
                            kernel_initializer='he_uniform')(flatten)
    output = Dense(units=2,
                   activation='sigmoid')(fully_connected)

    model = Model(inputs=model.inputs,
                  outputs=output)

    optimizer = SGD(lr=0.001,
                    momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
