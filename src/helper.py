import os


def get_jpeg_data_files_paths():
    """
    Returns the data file folders path
    
    :return: The data file paths as list [train_jpeg_dir, test_jpeg_dir, train_csv_file]
    """

    data_root_folder = os.path.abspath("../data/")
    train_jpeg_dir = os.path.join(data_root_folder, 'train-jpg')
    test_jpeg_dir = os.path.join(data_root_folder, 'test-jpg')
    train_csv_file = os.path.join(data_root_folder, 'train.csv')

    assert os.path.exists(data_root_folder), "The {} folder does not exist".format(data_root_folder)
    assert os.path.exists(train_jpeg_dir), "The {} folder does not exist".format(train_jpeg_dir)
    assert os.path.exists(test_jpeg_dir), "The {} folder does not exist".format(test_jpeg_dir)
    assert os.path.exists(train_csv_file), "The {} file does not exist".format(train_csv_file)
    return [train_jpeg_dir, test_jpeg_dir, train_csv_file]
