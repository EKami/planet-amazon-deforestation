import numpy as np
import os
import random

import data_helper
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.layers.core import Reshape


class Gans:
    def __init__(self, preprocessor):
        self.discriminator = Sequential()
        self.generator = Sequential()
        self.discriminator_model = Sequential()
        self.adversarial_model = Sequential()
        self.preprocessor = preprocessor
        self.img_channels = 3

    def is_images_already_generated(self):
        return False

    def add_discriminator(self, output_size):
        dropout = 0.4
        self.discriminator.add(Conv2D(64, 5, strides=2,
                                      input_shape=(*self.preprocessor.img_resize, self.img_channels),
                                      padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(128, 5, strides=2, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(256, 5, strides=2, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(512, 5, strides=1, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        # Out: 1-dim probability
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(output_size, activation='sigmoid'))
        self.generator.summary()

    def add_generator(self, z_vector=100):
        dropout = 0.4
        depth = 64 * 4
        initial_width = int(self.preprocessor.img_resize[0] / 4)
        initial_height = int(self.preprocessor.img_resize[1] / 4)

        # In: 100
        # Out: dim x dim x depth
        self.generator.add(Dense(initial_width * initial_height * depth, input_dim=z_vector))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        self.generator.add(Reshape((initial_width, initial_height, depth)))
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

        self.generator.add(Conv2DTranspose(self.img_channels, 5, padding='same'))
        self.generator.add(Activation('sigmoid'))
        self.generator.summary()

    def add_discriminator_model(self):
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.discriminator_model.add(self.discriminator)
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def add_adversarial_model(self):
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.adversarial_model = Sequential()
        self.adversarial_model.add(self.generator)
        self.adversarial_model.add(self.discriminator)
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
