import os
import sys
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from itertools import chain
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator


class AmazonPreprocessor:
    def __init__(self, train_jpeg_dir, train_csv_file, test_jpeg_dir, test_additional_jpeg_dir,
                 img_resize=(32, 32), validation_split=0.2, process_count=cpu_count()):
        """
        This class is used by the classifier to preprocess certains data, don't forget to call the init() method
        after an object from this class gets created
        :param validation_split: float
            Value between 0 and 1 used to split training set from validation set
        :param train_jpeg_dir: string
            The directory of the train files
        :param train_csv_file: string
            The path of the file containing the training labels
        :param test_jpeg_dir: string
            The directory of the all the test images
        :param test_additional_jpeg_dir: string
            The directory of the all the additional test images
        :param img_resize: tuple (int, int)
            The resize size of the original image given by the file_path argument
        :param process_count: int
            The number of process you want to use to preprocess the data.
            If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
        """
        self.process_count = process_count
        self.validation_split = validation_split
        self.img_resize = img_resize
        self.test_additional_jpeg_dir = test_additional_jpeg_dir
        self.test_jpeg_dir = test_jpeg_dir
        self.train_csv_file = train_csv_file
        self.train_jpeg_dir = train_jpeg_dir
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.y_map = None
        self.X_test = None
        self.X_test_filename = None

    def init(self):
        """
        Initialize the preprocessor and preprocess required data for the classifier to use
        """
        self.X_train, self.y_train, self.X_val, self.y_val, self.y_map = self._get_train_data_files()
        # Contains all the test files including the additional ones
        self.X_test, self.X_test_filename = self._get_test_data_files()

        if not self.img_resize:
            self.img_resize = Image.open(self.X_test_filename[0]).size
            print("Default image size is", self.img_resize)

        # The validation data cannot be preprocessed in batches as we also need them to compute the f2 score
        self.X_val, self.y_val = self._preprocess_val_files()

    def get_train_generator(self, batch_size):
        """
        Returns a batch generator which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier. Doing so allow to greatly optimize
        memory usage as the images are processed then deleted by chunks (defined by batch_size)
        instead of preprocessing them all at once and feeding them to the classifier.
        :param batch_size: int
            The batch size
        :return: generator
            The batch generator
        """
        # Image Augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)  # randomly flip images horizontally
        loop_range = len(self.X_train)
        while True:
            for i in range(loop_range):
                start_offset = batch_size * i

                # The last remaining files could be smaller than the batch_size
                range_offset = min(batch_size, loop_range - start_offset)

                # If we reached the end of the list then we break the loop
                if range_offset <= 0:
                    break

                batch_features = np.zeros((range_offset, *self.img_resize, 3))
                batch_labels = np.zeros((range_offset, len(self.y_train[0])))

                for j in range(range_offset):
                    # Maybe shuffle the index?
                    img = Image.open(self.X_train[start_offset + j])
                    img.thumbnail(self.img_resize)

                    # Augment the image `img` here

                    # Convert to RGB and normalize
                    img_array = np.asarray(img.convert("RGB"), dtype=np.float32)

                    img_array = img_array[:, :, ::-1]
                    # Zero-center by mean pixel
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68

                    batch_features[j] = img_array
                    batch_labels[j] = self.y_train[start_offset + j]

                # Augment the images (using Keras allow us to add randomization/shuffle to augmented images)
                # Here the next batch of the data generator (and only one for this iteration)
                # is taken and returned in the yield statement
                yield next(datagen.flow(batch_features, batch_labels, range_offset))

    def get_prediction_generator(self, batch_size):
        """
        Returns a batch generator which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier. Doing so allow to greatly optimize
        memory usage as the images are processed then deleted by chunks (defined by batch_size)
        instead of preprocessing them all at once and feeding them to the classifier.
        :param batch_size: int
            The batch size
        :return: generator
            The batch generator
        """

        # NO SHUFFLE HERE as we need our predictions to be in the same order as the inputs
        loop_range = len(self.X_test_filename)
        while True:
            for i in range(loop_range):
                start_offset = batch_size * i

                # The last remaining files could be smaller than the batch_size
                range_offset = min(batch_size, loop_range - start_offset)

                # If we reached the end of the list then we break the loop
                if range_offset <= 0:
                    break

                img_arrays = np.zeros((range_offset, *self.img_resize, 3))

                for j in range(range_offset):
                    img = Image.open(self.X_test_filename[start_offset + j])
                    img.thumbnail(self.img_resize)

                    # Convert to RGB and normalize
                    img_array = np.asarray(img.convert("RGB"), dtype=np.float32)

                    img_array = img_array[:, :, ::-1]
                    # Zero-center by mean pixel
                    img_array[:, :, 0] -= 103.939
                    img_array[:, :, 1] -= 116.779
                    img_array[:, :, 2] -= 123.68
                    img_array = img_array / 255

                    img_arrays[j] = img_array
                yield img_arrays

    def _get_class_mapping(self, *args):
        """

        :param args: list of arguments
            file_path: string
                The path of the image
            tags_str: string
                The associated tags as 1 string
            labels_map: dict {int: string}
                The map between the image label and their id
        :return: img_array, targets
            file_path: string
                The path to the file
            targets: Numpy array
                A 17 length vector
        """
        # Unpack the *args
        file_path, tags_str, labels_map = list(args[0])
        targets = np.zeros(len(labels_map))

        for t in tags_str.split(' '):
            targets[labels_map[t]] = 1
        return file_path, targets

    def _get_validation_split(self):
        train = pd.read_csv(self.train_csv_file)
        # mapping labels to integer classes
        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))
        label_map = {l: i for i, l in enumerate(labels)}

        y_train = []
        for f,tags in (train.values):
            targets = np.zeros(len(label_map))
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            y_train.append(targets)

        y_train = np.array(y_train, np.uint8)
        trn_index = []
        val_index = []
        index = np.arange(len(train))
        for i in (range(len(label_map))):
            sss = StratifiedShuffleSplit(n_splits=2, test_size=self.validation_split, random_state=i)
            for train_index, test_index in sss.split(index,y_train[:,i]):
                X_train, X_test = index[train_index], index[test_index]
            # to ensure there is no repetetion within each split and between the splits
            trn_index = trn_index + list(set(X_train) - set(trn_index) - set(val_index))
            val_index = val_index + list(set(X_test) - set(val_index) - set(trn_index))
        return np.array(trn_index), np.array(val_index)

    def _get_train_data_files(self):
        labels_df = pd.read_csv(self.train_csv_file)
        x_train_files, y_train_files = [], []
        x_val_files, y_val_files = [], []
        train_files, train_tags = [], []
        val_files, val_tags = [], []

        files_path = []
        tags_list = []
        for file_name, tags in labels_df.values:
            files_path.append('{}/{}.jpg'.format(self.train_jpeg_dir, file_name))
            tags_list.append(tags)


        if self.validation_split != 0:

            trn_index, val_index = self._get_validation_split()
            for index in trn_index:
                train_files.append(files_path[index])
                train_tags.append(tags_list[index])
            for index in val_index:
                val_files.append(files_path[index])
                val_tags.append(tags_list[index])

        else:
            train_files = files_path
            train_tags = tags_list


        labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
        y_map = {l: i for i, l in enumerate(labels)}

        with ThreadPoolExecutor(self.process_count) as pool:
            for file_name, targets in tqdm(pool.map(self._get_class_mapping,
                                                    [(file_name, tags, y_map)
                                                     for file_name, tags in zip(train_files, train_tags)]),
                                           total=len(train_files)):
                x_train_files.append(file_name)
                y_train_files.append(targets)

        if self.validation_split != 0:
            with ThreadPoolExecutor(self.process_count) as pool:
                for file_name, targets in tqdm(pool.map(self._get_class_mapping,
                                                        [(file_name, tags, y_map)
                                                         for file_name, tags in zip(val_files, val_tags)]),
                                               total=len(val_files)):
                    x_val_files.append(file_name)
                    y_val_files.append(targets)

        return [x_train_files, y_train_files, x_val_files, y_val_files, {v: k for k, v in y_map.items()}]

    def _val_transform_to_matrices(self, *args):
        """
        :param args: list of arguments
            file_name: string
                The name of the image
            :return: img_array, file_name
                img_array: Numpy array
                    The image from the file_path as a numpy array resized with img_resize
                file_name: string
                    The name of the test image
            """
        file_path, val_labels = list(args[0])
        img = Image.open(file_path)
        img.thumbnail(self.img_resize)

        # Augment the image `img` here

        # Convert to RGB and normalize
        img_array = np.array(img.convert("RGB"), dtype=np.float32)
        img_array = img_array[:, :, ::-1]
        # Zero-center by mean pixel
        img_array[:, :, 0] -= 103.939
        img_array[:, :, 1] -= 116.779
        img_array[:, :, 2] -= 123.68
        img_array = img_array / 255

        return img_array, val_labels

    def _preprocess_val_files(self):
        """
        Transform the images to ready to use data for the CNN
        :param val_labels: list
            List of file labels
        :param val_files: list
            List of file path
        :param process_count: int
            The number of process you want to use to preprocess the data.
            If you run into issues, lower this number. Its default value is equal to the number of core of your CPU
        :return: The images matrices and labels as [x_test, x_test_filename]
            x_test: The X test values as a numpy array
            x_test_filename: The files name of each test images in the same order as the x_test arrays
        """
        x = []
        final_val_labels = []

        # Multiprocess transformation, the map() function take a function as a 1st argument
        # and the argument to pass to it as the 2nd argument. These arguments are processed
        # asynchronously on threads defined by process_count and their results are stored in
        # the x_test and x_test_filename lists
        print("Transforming val dataset...")
        sys.stdout.flush()
        with ThreadPoolExecutor(self.process_count) as pool:
            for img_array, targets in tqdm(pool.map(self._val_transform_to_matrices,
                                                    [(file_path, labels)
                                                     for file_path, labels in zip(self.X_val, self.y_val)]),
                                           total=len(self.X_val)):
                x.append(img_array)
                final_val_labels.append(targets)
        ret = [np.array(x), np.array(final_val_labels)]
        print("Done. Size consumed by validation matrices {} mb".format(ret[0].nbytes / 1024 / 1024))
        sys.stdout.flush()
        return ret

    def _get_test_data_files(self):
        files_name = os.listdir(self.test_jpeg_dir)
        files_name_add = os.listdir(self.test_additional_jpeg_dir)
        # ! hstack is deprecated
        X_test_filename = np.hstack(([name.split(".")[0] for name in files_name],
                                     [name.split(".")[0] for name in files_name_add]))
        X_test_file_path = np.hstack(([self.test_jpeg_dir + "/" + name for name in files_name],
                                      [self.test_additional_jpeg_dir + "/" + name for name in files_name_add]))
        return X_test_filename, X_test_file_path


def get_jpeg_data_files_paths():
    """
    Returns the input file folders path

    :return: list of strings
        The input file paths as list [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]
    """

    data_root_folder = os.path.abspath("../input/")
    train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
    test_jpeg_additional = os.path.join(data_root_folder, 'test-jpg-additional')
    train_csv_file = os.path.join(data_root_folder, 'train_v2.csv')
    return [train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file]
