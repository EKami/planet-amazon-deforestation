import numpy as np
import pandas as pd
import os
import gc
from itertools import chain

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
    def __init__(self, train_set_folder, test_set_folder, train_csv_file):
        self.losses = []
        self.train_set_folder = train_set_folder
        self.test_set_folder = test_set_folder
        self.train_csv_file = train_csv_file

    def _preprocess_data(self):
        labels_df = pd.read_csv(self.train_csv_file)
        labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
        label_map = {l: i for i, l in enumerate(labels)}

        print(label_map)
        # for f, tags in tqdm(df_train.values, miniters=1000):
        #     img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
        #     targets = np.zeros(17)
        #     for t in tags.split(' '):
        #         targets[label_map[t]] = 1
        #     x_train.append(cv2.resize(img, (32, 32)))
        #     y_train.append(targets)
        # train_datagen = ImageDataGenerator(rescale=1. / 255).flow()
        # test_datagen = ImageDataGenerator(rescale=1. / 255).flow()
        # return [training_set, test_set]

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
