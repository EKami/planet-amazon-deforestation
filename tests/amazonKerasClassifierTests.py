import sys
import os
import pandas as pd
import numpy as np
from itertools import chain


sys.path.append('../src')
sys.path.append('../tests')
import data_helper


class TestAmazonKerasClassifier:
    """
    Use with pytest -q -s amazonKerasClassifierTests.py
    Checks that the preprocessed data have the right shape
    """
    def test_data_preprocess(self):
        img_resize = (16, 16)
        color_channels = 3  # RGB
        train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()

        assert os.path.exists(train_jpeg_dir), "The {} folder does not exist".format(train_jpeg_dir)
        assert os.path.exists(test_jpeg_dir), "The {} folder does not exist".format(test_jpeg_dir)
        assert os.path.exists(train_csv_file), "The {} file does not exist".format(test_jpeg_additional)
        assert os.path.exists(train_csv_file), "The {} file does not exist".format(train_csv_file)

        x_train, x_test, y_train, y_map, x_test_filename = data_helper.preprocess_data(train_jpeg_dir, test_jpeg_dir,
                                                                                       test_jpeg_additional,
                                                                                       train_csv_file, img_resize)
        labels_df = pd.read_csv(train_csv_file)
        labels_count = len(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
        train_files_count = len(os.listdir(train_jpeg_dir))
        test_files_count = len(os.listdir(test_jpeg_dir)) + len(os.listdir(test_jpeg_additional))
        assert x_train.shape == (train_files_count, *img_resize, color_channels)
        assert x_test.shape == (test_files_count, *img_resize, color_channels)
        assert y_train.shape == (train_files_count, labels_count)
