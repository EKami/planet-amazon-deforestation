import numpy as np
import os
import random

import data_helper
from sklearn.metrics import fbeta_score
from PIL import Image

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class AmazonKerasClassifier:
    def __init__(self):
        self.losses = []
        self.classifier = Sequential()

    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))

        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(128, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

    def add_flatten_layer(self):
        self.classifier.add(Flatten())

    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def _train_generator(self, train_files, y_labels, img_resize, batch_size):

        for i in range(len(train_files)):
            batch_features = np.zeros((batch_size, *img_resize, 3))
            batch_labels = np.zeros((batch_size, len(y_labels[0])))

            offset = batch_size * i
            for j in range(min(batch_size, len(train_files) - offset)):
                img = Image.open(train_files[offset + j])
                img.thumbnail(img_resize)  # Resize the image

                # Augment the image `img` here

                # Convert to RGB and normalize
                img_array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
                batch_features[j] = img_array
                batch_labels[j] = y_labels[offset + j]
            yield batch_features, batch_labels

    def train_model(self, train_files_list, train_labels, val_files_list, val_labels, img_resize,
                    learn_rate=0.001, epoch=5, batch_size=128, train_callbacks=()):
        history = LossHistory()
        X_valid, y_valid = data_helper.preprocess_val_files(val_files_list, val_labels, img_resize)
        generator = self._train_generator(train_files_list, train_labels, img_resize, batch_size)

        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.classifier.fit_generator(generator, len(train_files_list) / batch_size,
                                      epochs=epoch,
                                      verbose=1,
                                      validation_data=(X_valid, y_valid),
                                      callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions

    def map_predictions(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        backend.clear_session()
