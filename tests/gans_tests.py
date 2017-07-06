from src.data_helper import AmazonPreprocessor
from src.data_helper import get_jpeg_data_files_paths
from src.gans import Gans

import matplotlib.pyplot as plt


class TestGans:
    """
    Use with pytest -q -s gans_tests.py
    Checks that the preprocessed data have the right shape
    """

    def test_gans(self):
        img_resize = None
        train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = get_jpeg_data_files_paths()
        preprocessor = AmazonPreprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_jpeg_additional,
                                          img_resize, validation_split=0)
        preprocessor.init()
        gans = Gans(preprocessor)
        gans.add_discriminator(17)
        gans.add_generator()
        gans.add_discriminator_model()
        gans.add_adversarial_model()
        gans.train()
        images_generated, labels_generated = gans.generate(30)

        print("Labels fake result 1:", labels_generated[0])

        plt.rc('axes', grid=False)
        _, axs = plt.subplots(3, 4, sharex='col', sharey='row', figsize=(64, 64))
        axs = axs.ravel()

        for i in range(12):
            axs[i].imshow(images_generated[i])

        plt.show()

t = TestGans()
t.test_gans()
