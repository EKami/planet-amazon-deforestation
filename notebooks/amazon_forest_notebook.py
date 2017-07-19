# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Planet: Understanding the Amazon deforestation from Space challenge

# <markdowncell>

# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **

# <markdowncell>

# Start by adding the helper files to the python path

# <codecell>

import sys

sys.path.append('../src')
sys.path.append('../tests')

# <markdowncell>

# ## Import required modules

# <codecell>

import os
import gc
import bcolz
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint, CSVLogger

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN 
from imblearn.under_sampling import ClusterCentroids
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

import data_helper
from data_helper import AmazonPreprocessor
from keras_helper import AmazonKerasClassifier, Fbeta
from kaggle_data.downloader import KaggleDataDownloader

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# <markdowncell>

# Print tensorflow version for reuse (the Keras module is used directly from the tensorflow framework)

# <codecell>

tf.__version__

# <markdowncell>

# ## Download the competition files
# Download the dataset files and extract them automatically with the help of [Kaggle data downloader](https://github.com/EKami/kaggle-data-downloader)

# <codecell>

competition_name = "planet-understanding-the-amazon-from-space"

train, train_u = "train-jpg.tar.7z", "train-jpg.tar"
test, test_u = "test-jpg.tar.7z", "test-jpg.tar"
test_additional, test_additional_u = "test-jpg-additional.tar.7z", "test-jpg-additional.tar"
test_labels = "train_v2.csv.zip"
destination_path = "../input/"
is_datasets_present = False

# If the folders already exists then the files may already be extracted
# This is a bit hacky but it's sufficient for our needs
datasets_path = data_helper.get_jpeg_data_files_paths()
for dir_path in datasets_path:
    if os.path.exists(dir_path):
        is_datasets_present = True

if not is_datasets_present:
    # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
    downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)
    
    train_output_path = downloader.download_dataset(train, destination_path)
    downloader.decompress(train_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + train_u, destination_path) # Extract the content of the previous tar file
    os.remove(train_output_path) # Removes the 7z file
    os.remove(destination_path + train_u) # Removes the tar file
    
    test_output_path = downloader.download_dataset(test, destination_path)
    downloader.decompress(test_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + test_u, destination_path) # Extract the content of the previous tar file
    os.remove(test_output_path) # Removes the 7z file
    os.remove(destination_path + test_u) # Removes the tar file
    
    test_add_output_path = downloader.download_dataset(test_additional, destination_path)
    downloader.decompress(test_add_output_path, destination_path) # Outputs a tar file
    downloader.decompress(destination_path + test_additional_u, destination_path) # Extract the content of the previous tar file
    os.remove(test_add_output_path) # Removes the 7z file
    os.remove(destination_path + test_additional_u) # Removes the tar file
    
    test_labels_output_path = downloader.download_dataset(test_labels, destination_path)
    downloader.decompress(test_labels_output_path, destination_path) # Outputs a csv file
    os.remove(test_labels_output_path) # Removes the zip file
else:
    print("All datasets are present.")

# <markdowncell>

# ## Inspect image labels
# Visualize what the training set looks like

# <codecell>

train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
labels_df.head()

# <markdowncell>

# Each image can be tagged with multiple tags, lets list all uniques tags

# <codecell>

# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))

# <markdowncell>

# ### Repartition of each labels

# <codecell>

# Histogram of label instances
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')

# <markdowncell>

# ## Images
# Visualize some chip images to know what we are dealing with.
# Lets vizualise 1 chip for the 17 images to get a sense of their differences.

# <codecell>

images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir + '/' + image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))

# <markdowncell>

# # Image resize & validation split
# Define the dimensions of the image data trained by the network. Recommended resized images could be 32x32, 64x64, or 128x128 to speedup the training. 
# 
# You could also use `None` to use full sized images.
# 
# Be careful, the higher the `validation_split_size` the more RAM you will consume.

# <codecell>

img_resize = (128, 128) # The resize size of each image ex: (64, 64) or None to use the default image size
validation_split_size = 0.2

# <markdowncell>

# # Data preprocessing
# Due to the hudge amount of memory the preprocessed images can take, we will create a dedicated `AmazonPreprocessor` class which job is to preprocess the data right in time at specific steps (training/inference) so that our RAM don't get completely filled by the preprocessed images. 
# 
# The only exception to this being the validation dataset as we need to use it as-is for f2 score calculation as well as when we calculate the validation accuracy of each batch.

# <codecell>

preprocessor = AmazonPreprocessor(train_jpeg_dir, train_csv_file, test_jpeg_dir, test_jpeg_additional, 
                                  img_resize, validation_split_size)
preprocessor.init()

# <codecell>

print("X_train/y_train lenght: {}/{}".format(len(preprocessor.X_train), len(preprocessor.y_train)))
print("X_val/y_val lenght: {}/{}".format(len(preprocessor.X_val), len(preprocessor.y_val)))
print("X_test/X_test_filename lenght: {}/{}".format(len(preprocessor.X_test), len(preprocessor.X_test_filename)))
preprocessor.y_map

# <markdowncell>

# ## Choose Hyperparameters
# 
# Choose your hyperparameters below for training. 
# 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined in the array `learn_rates` and corresponding number of epochs for each `epochs_arr`. Feel free to change these values if you like or just use the defaults.
# 
# If you opted for a high `img_resize` then you may want to lower the `batch_size` to fit your images matrices into the GPU memory. With an `img_resize` of `(256, 256)` (the default size) and a batch_size of `64` you should at least have a GPU with 8GB or VRAM.

# <codecell>

batch_size = 64
epochs_arr = [35, 15, 5]
learn_rates = [0.001, 0.0001, 0.00001]

# <markdowncell>

# # Define and Train model
# 
# Here we define the model and begin training. 

# <codecell>

classifier = AmazonKerasClassifier(preprocessor)
classifier.add_conv_layer()
classifier.add_flatten_layer()
classifier.add_ann_layer(len(preprocessor.y_map))

loss_rdir = 'saved_loss'
or_weights_path = "best_original_weights.hdf5"
train_losses, val_losses, fbeta_score = [], [], None

if os.path.exists(loss_rdir) and os.path.exists(or_weights_path):
    train_losses, val_losses, fbeta_score = bcolz.open(loss_rdir)
    print("Weights will be loaded from disk")
else:
    # Saves the best model weights across all epochs in the training process.
    checkpoint = ModelCheckpoint(filepath=or_weights_path, monitor='val_acc', verbose=1, save_best_only=True)
    # Creates a file with a log of loss and accuracy per epoch
    csv = CSVLogger(filename='training.log', append=True)
    # Calculates fbeta_score after each epoch during training
    fbeta = Fbeta()
    
    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(preprocessor.X_train, 
                                                                               preprocessor.y_train, 
                                                                               learn_rate, epochs, batch_size,
                                                                               augment_data=True,
                                                                               train_callbacks=[checkpoint, fbeta, csv])
        train_losses += tmp_train_losses
        val_losses += tmp_val_losses
    c = bcolz.carray([train_losses, val_losses, fbeta_score], rootdir=loss_rdir, mode='w')
    c.flush()
    # Here we check if our validation set loss is consistent. Otherwise it would mean that it is too small.
    classifier.show_validation_set_consistency()
    # Show the fbeta score evolution for each epochs
    print("Fbeta score list:\n" + str(fbeta.fbeta))

# <markdowncell>

# ## Load Best Weights
# 
# Loads the best weights of the training (will directly jump to this step if the model was already trained)

# <codecell>

classifier.load_weights(or_weights_path)
print("Weights loaded")

# <markdowncell>

# ## Monitor the results

# <markdowncell>

# Check that we do not overfit by plotting the losses of the train and validation sets

# <codecell>

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();

# <markdowncell>

# Look at our overall fbeta_score

# <codecell>

fbeta_score
#old 0.92240015515682305

# <markdowncell>

# # Pseudo labeling

# <markdowncell>

# To do pseudo labeling we'll start by merging our training data with a subset of the test set.
# Below we merge the **paths** where the files are located, not the actual data. Using directly the data would eat up too much memory.

# <codecell>

# Take a subset of the test set
x_subset = preprocessor.X_test_filename[:20000]

# We predict the labels of our preprocessor.X_test_filename set
pseudo_predictions, pseudo_paths = classifier.predict(x_subset, batch_size)

# We use the 0.2 threshold to make predictions one-hot encoded
# TODO Use https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html
pseudo_predictions = [[1 if y > 0.2 else 0 for y in x] for x in pseudo_predictions]

labels_list = list(preprocessor.y_map.values())
pred_df = pd.DataFrame([[x, *y] for x, y in zip(pseudo_paths, pseudo_predictions)], columns=["file", *labels_list])
nb_classes = [label + ": " +str(pred_df[label].value_counts()[1]) for label in labels_list]
for count in nb_classes: print(count)

# <markdowncell>

# ### Undersample the most used classes
# 
# We need to undersample the predictions which has the most seen set of labels

# <codecell>

# Apply some artisanal made undersampling (LOL)
undersampled_df = pred_df[(pred_df['conventional_mine'] == 1) | 
                         (pred_df['blow_down'] == 1) |
                         (pred_df['slash_burn'] == 1) |
                         (pred_df['blooming'] == 1) |
                         (pred_df['artisinal_mine'] == 1) |
                         (pred_df['selective_logging'] == 1) | 
                         (pred_df['bare_ground'] == 1) |
                         (pred_df['cloudy'] == 1) |
                         (pred_df['haze'] == 1) |
                         (pred_df['habitation'] == 1) |
                         (pred_df['cultivation'] == 1)]
nb_classes = [label + ": " +str(undersampled_df[label].value_counts()[1]) for label in undersampled_df.columns]
print("Undersampled classes count:")
for count in nb_classes: print(count)

# We can now concatenate our pseudo features with the training features
combo_x_pseudo = np.concatenate([preprocessor.X_train[:5000], [v[0] for v in undersampled_df.iloc[:, 0:1].values.tolist()]])

# Now we concatenate our predictions to our training labels
combo_y_pseudo = np.concatenate([preprocessor.y_train[:5000], undersampled_df.iloc[:, 1:].values.tolist()])

# <codecell>

assert combo_x_pseudo.shape[0] == combo_y_pseudo.shape[0]
print("X pseudo shape: {}\nY pseudo shape: {}".format(combo_x_pseudo.shape, combo_y_pseudo.shape))

# <markdowncell>

# ## Re-train with Pseudo labels
# We take the same model from before and train it against pseudo labeled images

# <codecell>

# Don't create a new checkpoint. Use our existing one to update it's weights
# checkpoint = ModelCheckpoint("pseudo_labeling_with_augment_weights.hdf5", monitor='val_acc', 
#                              verbose=1, save_best_only=True, mode='max')

# <codecell>

epochs_arr = [5, 3, 2]
learn_rates = [0.001, 0.0001, 0.00001]

# We don't use `checkpoint` here as the weights could have been loaded directly without creating the variable
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(combo_x_pseudo, combo_y_pseudo, 
                                                                           learn_rate, epochs, batch_size=64, 
                                                                           augment_data=True)
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses

# <markdowncell>

# ## Load Best Weights

# <codecell>

# Keep the weights of the original classifier merged with the pseudo labeling
# classifier.load_weights(or_weights_path)
# print("Weights loaded")

# <markdowncell>

# ## Plot the loss change

# <codecell>

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();

# <markdowncell>

# And finally get the final f2 score

# <codecell>

fbeta_score

# <markdowncell>

# ## Make final predictions

# <codecell>

predictions, x_test_filename = classifier.predict(preprocessor.X_test_filename, batch_size)
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions ({}) entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              x_test_filename[0], predictions[0]))

# <markdowncell>

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class

# <codecell>

# For now we'll just put all thresholds to 0.2 but we need to calculate the real ones in the future
#thresholds = [0.2] * len(labels_set)
thresholds = [0.24, 0.2, 0.2, 0.2, 0.2, 0.14, 0.05, 0.2, 0.2, 0.25, 0.25, 0.24, 0.2, 0.25, 0.2, 0.2, 0.25]

# <markdowncell>

# Now lets map our predictions to their tags by using the thresholds

# <codecell>

predicted_labels = classifier.map_predictions(predictions, thresholds)

# <markdowncell>

# Finally lets assemble and visualize our predictions for the test dataset

# <codecell>

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split("/")[-1].split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

# <codecell>

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
print("Predictions rows:", final_df.size)
final_df.head()

# <codecell>

tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');

# <markdowncell>

# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...

# <markdowncell>

# And save it to a submission file

# <codecell>

final_df.to_csv('../submission_file.csv', index=False)
classifier.close()

# <markdowncell>

# #### That's it, we're done!
