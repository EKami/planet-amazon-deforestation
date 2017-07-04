import numpy as np
import os
import random

import data_helper
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D


class Gans:
    def __init__(self, preprocessor):
        self.discriminator = Sequential()
        self.generator = Sequential()
        self.discriminator_model = Sequential()
        self.adversarial_model = Sequential()
        self.preprocessor = preprocessor

    def is_images_already_generated(self):
        return False

    def add_discriminator(self):
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.discriminator.add(Conv2D(depth * 1, 5, strides=2, input_shape=input_shape,
                                      padding='same', activation=LeakyReLU(alpha=0.2)))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth * 2, 5, strides=2,
                                      padding='same', activation=LeakyReLU(alpha=0.2)))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth * 4, 5, strides=2,
                                      padding='same', activation=LeakyReLU(alpha=0.2)))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth * 8, 5, strides=1, padding='same',
                                      activation=LeakyReLU(alpha=0.2)))
        self.discriminator.add(Dropout(dropout))

        # Out: 1-dim probability
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(1))
        self.discriminator.add(Activation('sigmoid'))
        self.discriminator.summary()

    def add_generator(self):
        dropout = 0.4
        depth = 64 + 64 + 64 + 64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.generator.add(Dense(dim * dim * depth, input_dim=100))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        self.generator.add(Reshape((dim, dim, depth)))
        self.generator.add(Dropout(dropout))
        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        self.generator.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.generator.add(Conv2DTranspose(1, 5, padding='same'))
        self.generator.add(Activation('sigmoid'))
        self.generator.summary()
        return self.G

    def build_models(self):
        optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        self.discriminator_model.add(self.discriminator())
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.adversarial_model.add(self.generator())
        self.adversarial_model.add(self.discriminator())
        self.adversarial_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train(self):
        images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :, :, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        images_fake = self.generator.predict(noise)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        d_loss = self.discriminator.train_on_batch(x, y)
        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
        a_loss = self.adversarial.train_on_batch(noise, y)
