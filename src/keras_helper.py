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
    def __init__(self, preprocessor):
        self.losses = []
        self.classifier = Sequential()
        self.preprocessor = preprocessor

    def add_conv_layer(self):
        img_channels = 3
        self.classifier.add(BatchNormalization(input_shape=(*self.preprocessor.img_resize, img_channels)))

        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
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

    def train_model(self, learn_rate=0.001, epoch=5, batch_size=128, train_callbacks=()):
        history = LossHistory()
        train_generator = self.preprocessor.get_train_generator(batch_size)
        X_val, y_val = self.preprocessor.X_val, self.preprocessor.y_val
        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # early stopping will auto-stop training process if model stops learning after 3 epochs
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        self.classifier.fit_generator(train_generator, len(self.preprocessor.X_train) / batch_size,
                                      epochs=epoch,
                                      verbose=1,
                                      validation_data=(X_val, y_val),
                                      callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_val, y_val)
        return [history.train_losses, history.val_losses, fbeta_score]

    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)

    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)

    def predict(self, batch_size=128):
        """
        Launch the predictions on the test dataset as well as the additional test dataset
        :return:
            predictions_labels: list
                An array containing 17 length long arrays
            filenames: list
                File names associated to each prediction
        """
        generator = self.preprocessor.get_prediction_generator(batch_size)
        predictions_labels = self.classifier.predict_generator(generator, len(self.preprocessor.X_test_filename) / batch_size)
        assert len(predictions_labels) == len(self.preprocessor.X_test), \
            "len(predictions_labels) = {}, len(self.preprocessor.X_test) = {}".format(
                len(predictions_labels), len(self.preprocessor.X_test))
        return predictions_labels, np.array(self.preprocessor.X_test)

    def map_predictions(self, predictions, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [self.preprocessor.y_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        backend.clear_session()
