import numpy as np
import os
import gc

from sklearn.metrics import fbeta_score

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.callbacks import Callback


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''

    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1

    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'


class AmazonKerasClassifier:
    def __init__(self, ):
        self.losses = []

    def train_model(self, epoch, batch_size, ):
        history = LossHistory()
        training_set, test_set = _get_datasets_directory_flows()

        classifier.fit(training_set,
                                 steps_per_epoch=8000 / batch_size,
                                 epochs=epoch,
                                 validation_data=test_set,
                                 validation_steps=2000 / batch_size,
                                 workers=12,
                                 max_q_size=100,
                                 callbacks=[history])
