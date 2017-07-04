import sys
import pandas as pd

sys.path.append('../src')
sys.path.append('../tests')
import data_helper
from gans import Gans
from data_helper import AmazonPreprocessor


class TestGans:
    """
    Use with pytest -q -s gans_tests.py
    Checks that the preprocessed data have the right shape
    """

    def test_gans(self):
        img_resize = None
        train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
        preprocessor = AmazonPreprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_jpeg_additional,
                                          img_resize, validation_split=0.1)
        preprocessor.init()
        gans = Gans(preprocessor)
        gans.add_discriminator(17)
        gans.add_generator()
        gans.add_discriminator_model()
        gans.add_adversarial_model()
        #gans.train()
        print("Dir", train_jpeg_dir)
