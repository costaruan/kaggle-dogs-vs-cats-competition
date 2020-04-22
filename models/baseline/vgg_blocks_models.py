from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD


def vgg_one_block_model(size, channels):
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same',
                     input_shape=(size, size, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128,
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(units=2,
                    activation='sigmoid'))

    optimizer = SGD(lr=0.001,
                    momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def vgg_two_blocks_model(size, channels):
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same',
                     input_shape=(size, size, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128,
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(units=2,
                    activation='sigmoid'))

    optimizer = SGD(lr=0.001,
                    momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def vgg_three_blocks_model(size, channels):
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same',
                     input_shape=(size, size, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128,
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=2,
                    activation='sigmoid'))

    optimizer = SGD(lr=0.001,
                    momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
