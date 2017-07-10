import os
import numpy as np
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.core import Reshape
import tensorflow as tf

from src.data_helper import AmazonPreprocessor


class Gans:
    """
    -   How to train a GAN?
        https://github.com/soumith/ganhacks

    -   Gans by example with Keras:
        https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

    - DCGans by Ian GoodFellow:
        https://github.com/EKami/deep_learning_foundation_nanodegree/blob/master/semi-supervised-gans/semi-supervised_learning_2_solution.ipynb

    """
    def __init__(self, preprocessor: AmazonPreprocessor, z_vector_size=100):
        self.discriminator = Sequential()
        self.generator = Sequential()
        self.discriminator_model = Sequential()
        self.adversarial_model = Sequential()
        self.preprocessor = preprocessor
        self.img_channels = 3
        self.z_vector_size = z_vector_size
        current_path = os.path.dirname(__file__)
        self.generated_images_path = os.path.join(current_path, "../input/generated")
        self.generated_labels_file = os.path.join(current_path, "../input/generated.csv")

    def is_images_already_generated(self):
        return os.path.exists(self.generated_images_path) and os.path.exists(self.generated_labels_file)

    def add_discriminator(self, output_size):
        dropout = 0.5
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
        # +1 because the last column is to classify fake data
        self.discriminator.add(Dense(output_size + 1, activation='sigmoid'))

    def add_generator(self):
        dropout = 0.5
        depth = 64 * 4
        initial_width = int(self.preprocessor.img_resize[0] / 4)
        initial_height = int(self.preprocessor.img_resize[1] / 4)

        # In: 100
        # Out: dim x dim x depth
        self.generator.add(Dense(initial_width * initial_height * depth, input_dim=self.z_vector_size))
        self.generator.add(BatchNormalization())
        self.generator.add(Activation('relu'))

        self.generator.add(Reshape((initial_width, initial_height, depth)))
        self.generator.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(Activation('relu'))

        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(Activation('relu'))

        self.generator.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        self.generator.add(BatchNormalization())
        self.generator.add(Activation('relu'))

        self.generator.add(Conv2DTranspose(self.img_channels, 5, padding='same'))
        self.generator.add(Activation('sigmoid'))

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

    def train(self, batch_size=16):
        d_acc = []
        a_acc = []

        writer = tf.summary.FileWriter(logdir='./logs')
        train_generator = self.preprocessor.get_train_generator(batch_size, 1)
        for i in tqdm(range(int(len(self.preprocessor.X_train) / batch_size)), desc="Training GANs"):
            images_train, labels_real = next(train_generator)
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.z_vector_size])

            images_fake = self.generator.predict(noise)
            labels_fake = np.zeros([labels_real.shape[0], labels_real.shape[1] + 1])

            # Add 1 class to the fake labels as fake
            labels_fake[:, len(labels_fake[0]) - 1] = 1
            # Add 0 at the end of each onehot vector for real images
            labels_real = np.column_stack((labels_real, [0] * batch_size))

            x = np.concatenate((images_train, images_fake))
            y = np.concatenate((labels_real, labels_fake))

            d_acc_curr = self.discriminator_model.train_on_batch(x, y)[1]
            d_acc.append(d_acc_curr)

            summary = tf.Summary(value=[tf.Summary.Value(tag="discriminator_accuracy", simple_value=d_acc_curr)])
            writer.add_summary(summary)

            y = np.ones([batch_size, labels_fake.shape[1]])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, self.z_vector_size])
            a_acc_curr = self.adversarial_model.train_on_batch(noise, y)[1]
            a_acc.append(a_acc_curr)

            summary = tf.Summary(value=[tf.Summary.Value(tag="adversarial_accuracy", simple_value=a_acc_curr)])
            writer.add_summary(summary)

        return [d_acc, a_acc]

    def generate(self, n):
        noise = np.random.uniform(-1.0, 1.0, size=[n, 100])
        images_generated = self.generator.predict(noise)
        labels_generated = self.discriminator_model.predict(images_generated)
        return [images_generated, labels_generated]

