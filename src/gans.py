import numpy as np
import os
import random

import data_helper
from keras_helper import LossHistory

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
    def __init__(self, preprocessor, z_vector_size=100):
        self.discriminator = Sequential()
        self.generator = Sequential()
        self.discriminator_model = Sequential()
        self.adversarial_model = Sequential()
        self.preprocessor = preprocessor
        self.img_channels = 3
        self.z_vector_size = z_vector_size

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

    def add_generator(self):
        dropout = 0.4
        depth = 64 * 4
        initial_width = int(self.preprocessor.img_resize[0] / 4)
        initial_height = int(self.preprocessor.img_resize[1] / 4)

        # In: 100
        # Out: dim x dim x depth
        self.generator.add(Dense(initial_width * initial_height * depth, input_dim=self.z_vector_size))
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

    def train(self, epoch=1, batch_size=128, train_callbacks=()):
        # TODO remove the validation set

        discriminator_history = LossHistory()
        adversarial_history = LossHistory()
        train_generator = self.preprocessor.get_train_generator(batch_size)

        for i in range(epoch):
            images_train = next(train_generator)
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.z_vector_size])

            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))

            # Stop here
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval > 0:
                if (i + 1) % save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i + 1))

        self.discriminator_model.fit_generator(train_generator,
                                               len(self.preprocessor.X_train) / batch_size,
                                               epochs=1, verbose=1,
                                               validation_data=(self.preprocessor.X_val, self.preprocessor.y_val),
                                               callbacks=[discriminator_history, *train_callbacks])
        # y = np.ones([batch_size, 1])
        # noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.z_vector_size])
        # a_loss = self.adversarial_model.train_on_batch(noise, y)
