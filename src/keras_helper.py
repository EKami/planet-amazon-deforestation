import numpy as np
import os

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback
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
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))
        self.classifier.add(Conv2D(16, (2, 2), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))

    def add_flatten_layer(self):
        self.classifier.add(Flatten())

    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(256, activation='relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, x_train, y_train, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)
        self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, *train_callbacks])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]

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

