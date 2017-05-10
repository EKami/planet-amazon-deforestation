import os
import sys
import cv2
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

def get_jpeg_data_files_paths():
    """
    Returns the input file folders path
    
    :return: The input file paths as list [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]
    """

    data_root_folder = os.path.abspath("../input/")
    train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
    test_jpeg_additional = os.path.join(data_root_folder, 'test-jpg-additional')
    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')

    assert os.path.exists(data_root_folder), "The {} folder does not exist".format(data_root_folder)
    assert os.path.exists(train_jpeg_dir), "The {} folder does not exist".format(train_jpeg_dir)
    assert os.path.exists(test_jpeg_dir), "The {} folder does not exist".format(test_jpeg_dir)
    assert os.path.exists(train_csv_file), "The {} file does not exist".format(test_jpeg_additional)
    assert os.path.exists(train_csv_file), "The {} file does not exist".format(train_csv_file)
    return [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]

def _train_transform_to_matrices(*args):
    file_path, tags, labels_map, img_resize = args[0][0], args[0][1], args[0][2], args[0][3]
    img_array = np.array(cv2.resize(cv2.imread(file_path), img_resize), dtype=np.float32)
    cv2.normalize(img_array, img_array, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    targets = np.zeros(len(labels_map))
    for t in tags.split(' '):
        targets[labels_map[t]] = 1
    return img_array, targets


def _test_transform_to_matrices(*args):
    test_set_folder, file_name, img_resize = args[0][0], args[0][1], args[0][2]
    img_array = np.array(cv2.resize(cv2.imread('{}/{}'.format(test_set_folder, file_name)), img_resize),
                         dtype=np.float32)
    cv2.normalize(img_array, img_array, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return [img_array, file_name]


def _get_train_matrices(train_set_folder, train_csv_file, img_resize, process_count):
    labels_df = pd.read_csv(train_csv_file)
    labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
    labels_map = {l: i for i, l in enumerate(labels)}

    files_path = []
    tags_list = []
    for file_name, tags in labels_df.values:
        files_path.append('{}/{}.jpg'.format(train_set_folder, file_name))
        tags_list.append(tags)

    # Multiprocess transformation
    x_train = []
    y_train = []
    print("Transforming train data to matrices. Using {} threads...".format(process_count))
    sys.stdout.flush()
    with ThreadPoolExecutor(process_count) as pool:
        for img_array, targets in tqdm(pool.map(_train_transform_to_matrices,
                                                [[file_path, tag, labels_map, img_resize]
                                                 for file_path, tag in zip(files_path, tags_list)]),
                                       total=len(files_path)):
            x_train.append(img_array)
            y_train.append(targets)
    return [x_train, y_train, {v: k for k, v in labels_map.items()}]


def _get_test_matrices(test_set_folder, img_resize, process_count):
    x_test = []
    x_test_filename = []
    files_name = os.listdir(test_set_folder)

    with ThreadPoolExecutor(process_count) as pool:
        for img_array, file_name in tqdm(pool.map(_test_transform_to_matrices,
                                             [[test_set_folder, file_name, img_resize]
                                              for file_name in files_name]),
                                         total=len(files_name)):
            x_test.append(img_array)
            x_test_filename.append(file_name)
    return [x_test, x_test_filename]


def preprocess_train_data(train_set_folder, train_csv_file, img_resize=(32, 32), process_count=cpu_count()):
    """
    Transform the train images to ready to use data for the CNN 
    :param train_set_folder: the folder containing the images for training
    :param train_csv_file: the file containing the labels of the training images
    :param img_resize: the standard size you want to have on images when transformed to matrices
    :param process_count: the number of process you want to use to preprocess the data.
        If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
    :return: The images matrices and labels as [x_train, y_train, labels_map]
        x_train: The X train values as a numpy array
        y_train: The Y train values as a numpy array
        labels_map: The mapping between the tags labels and their indices
    """
    x_train, y_train, labels_map = _get_train_matrices(train_set_folder, train_csv_file, img_resize, process_count)
    ret = [np.array(x_train), np.array(y_train, dtype=np.uint8), labels_map]
    print("Done. Size consumed by arrays {} mb".format((ret[0].nbytes + ret[1].nbytes) / 1024 / 1024))
    return ret

def preprocess_test_data(test_set_folder, img_resize=(32, 32), process_count=cpu_count()):
    """
    Transform the images to ready to use data for the CNN
    :param test_set_folder: the folder containing the images for testing
    :param img_resize: the standard size you want to have on images when transformed to matrices
    :param process_count: the number of process you want to use to preprocess the data.
        If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
    :return: The images matrices and labels as [x_test, x_test_filename]
        x_test: The X test values as a numpy array
        x_test_filename: The files name of each test images in the same order as the x_test arrays
    """
    x_test, x_test_filename = _get_test_matrices(test_set_folder, img_resize, process_count)
    ret = [np.array(x_test), x_test_filename]
    print("Done. Size consumed by arrays {} mb".format(ret[0].nbytes / 1024 / 1024))
    return ret

def preprocess_data(train_set_folder, test_set_folder,
                    test_set_additional, train_csv_file, img_resize=(32, 32), process_count=cpu_count()):
    """
    Transform the all the images to ready to use data for the CNN
    :param train_set_folder: the folder containing the images for training
    :param test_set_folder: the folder containing the images for testing
    :param test_set_additional: the folder containing the images for additional testing (updated on 05/05/2017) 
            https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32157
    :param train_csv_file: the file containing the labels of the training images
    :param img_resize: the standard size you want to have on images when transformed to matrices
    :param process_count: the number of process you want to use to preprocess the data.
        If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
    :return: The images matrices and labels as [x_train, x_test, y_train, labels_map, x_test_filename]
        x_train: The X train values as a numpy array
        x_test: The X test values as a numpy array
        y_train: The Y train values as a numpy array
        labels_map: The mapping between the tags labels and their indices
        x_test_filename: The files name of each test images in the same order as the x_test arrays
    """
    x_train, y_train, labels_map = _get_train_matrices(train_set_folder, train_csv_file, img_resize, process_count)
    print("Transforming test data to matrices. Using {} threads...".format(process_count))
    sys.stdout.flush()
    x_test, x_test_filename = _get_test_matrices(test_set_folder, img_resize, process_count)
    print("Transforming additional test data to matrices. Using {} threads...".format(process_count))
    sys.stdout.flush()
    x_test_add, x_test_filename_add = _get_test_matrices(test_set_additional, img_resize, process_count)
    x_test = np.vstack((x_test, x_test_add))
    x_test_filename = np.hstack((x_test_filename, x_test_filename_add))
    ret = [np.array(x_train), np.array(x_test), np.array(y_train, dtype=np.uint8), labels_map, x_test_filename]
    gc.collect()
    print("Done. Size consumed by arrays {} mb".format((ret[0].nbytes + ret[1].nbytes + ret[2].nbytes) /1024/1024))
    return ret
