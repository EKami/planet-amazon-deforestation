import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from multiprocessing import Pool, cpu_count


def get_jpeg_data_files_paths():
    """
    Returns the input file folders path
    
    :return: The input file paths as list [train_jpeg_dir, test_jpeg_dir, train_csv_file]
    """

    data_root_folder = os.path.abspath("../input/")
    train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
    train_csv_file = os.path.join(data_root_folder, 'train.csv')

    assert os.path.exists(data_root_folder), "The {} folder does not exist".format(data_root_folder)
    assert os.path.exists(train_jpeg_dir), "The {} folder does not exist".format(train_jpeg_dir)
    assert os.path.exists(test_jpeg_dir), "The {} folder does not exist".format(test_jpeg_dir)
    assert os.path.exists(train_csv_file), "The {} file does not exist".format(train_csv_file)
    return [train_jpeg_dir, test_jpeg_dir, train_csv_file]


def _get_train_matrices(train_set_folder, train_csv_file, scale_fct, img_resize):
    labels_df = pd.read_csv(train_csv_file)
    labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
    labels_map = {l: i for i, l in enumerate(labels)}

    x_train = []
    y_train = []
    print("Transforming train data to matrices...")
    sys.stdout.flush()

    # TODO finish
    p = Pool(cpu_count())
    #ret = p.map(get_features, paths)

    for file_name, tags in tqdm(labels_df.values):
        img = cv2.imread('{}/{}.jpg'.format(train_set_folder, file_name))
        targets = np.zeros(len(labels))
        for t in tags.split(' '):
            targets[labels_map[t]] = 1
        x_train.append(cv2.resize(img, img_resize))
        y_train.append(targets)
    x_train = scale_fct(np.array(x_train, np.float32))
    return [x_train, np.array(y_train, np.uint8), {v: k for k, v in labels_map.items()}]


def _get_test_matrices(test_set_folder, img_resize):
    x_test = []
    x_test_filename = []
    files = os.listdir(test_set_folder)
    print("Transforming test data to matrices...")
    sys.stdout.flush()
    for file_name in tqdm(files):
        img = cv2.imread('{}/{}'.format(test_set_folder, file_name))
        x_test.append(cv2.resize(img, img_resize))
        x_test_filename.append(file_name)
    return [np.array(x_test, np.float32), np.array(x_test_filename)]


def preprocess_data(train_set_folder, test_set_folder, train_csv_file, img_resize=(32, 32)):
    """
    Transform the images to ready to use data for CNN
    :param train_set_folder: the folder containing the images for training
    :param test_set_folder: the folder containing the images for testing
    :param train_csv_file: the file containing the labels of the training images
    :param img_resize: the standard size you want to have on images when transformed to matrices
    :return: The images matrices and labels as [x_train, x_test, y_train, labels_map, x_test_filename]
        x_train: The X train values as a numpy array
        x_test: The X test values as a numpy array
        y_train: The Y train values as a numpy array
        labels_map: The mapping between the tags labels and their indices
        x_test_filename: The files name of each test images in the same order as the x_test arrays
    """
    x_train, y_train, labels_map = _get_train_matrices(train_set_folder, train_csv_file, lambda x: x / 255, img_resize)
    x_test, x_test_filename = _get_test_matrices(test_set_folder, img_resize)
    print("Done.")
    return [x_train, x_test, y_train, labels_map, x_test_filename]
