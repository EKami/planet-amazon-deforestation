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
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import data_helper
from keras_helper import AmazonKerasClassifier
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

# # Image Resize
# Define the dimensions of the image data trained by the network. Due to memory constraints we can't load in the full size 256x256 jpg images. Recommended resized images could be 32x32, 64x64, or 128x128.

# <codecell>

img_resize = (64, 64) # The resize size of each image

# <markdowncell>

# # Data preprocessing
# Preprocess the data in order to fit it into the Keras model.
# 
# Due to the hudge amount of memory the resulting matrices will take, the preprocessing will be splitted into several steps:
#     - Preprocess training data (images and labels) and train the neural net with it
#     - Delete the training data and call the gc to free up memory
#     - Preprocess the first testing set
#     - Predict the first testing set labels
#     - Delete the first testing set
#     - Preprocess the second testing set
#     - Predict the second testing set labels and append them to the first testing set
#     - Delete the second testing set

# <codecell>

import bcolz

# Create functions to save and load tensor arrays.
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

# <codecell>

x_train, y_train, y_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)

# Save preprocessed training tensors
save_array('x_train.bc', x_train)
save_array('y_train.bc', y_train)

# Load in preprocessed training tensors
x_train = load_array('x_train.bc')
y_train = load_array('y_train.bc')

# Free up all available memory space after this heavy operation
gc.collect();

# <codecell>

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
y_map

# <markdowncell>

# ## Obtain bottleneck features for Transfer Learning
# Here we define a pre-trained model (i.e. VGG16, ResNet50, etc.) which we use to extract bottleneck features from our training data

# <markdowncell>

# ## VGG16
# For now we are just choosing VGG16 but this could technically be applied to any pretrained model

# <codecell>

# VGG16 - if using a different image resize make sure to change input_shape

from keras.applications.vgg16 import VGG16
model = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
model.summary()

# <markdowncell>

# First we extract bottlneck features from the training images

# <codecell>

# Extract bottleneck features from the VGG16 model
VGG16_train = model.predict(x_train)

#save the bottleneck features to disk for later access
save_array('VGG16_train.bc', VGG16_train)

# assign the loaded bottleneck features
VGG16_train = load_array('VGG16_train.bc')

# For now we just need training bottlenecks, we can load in the test data later when we make predictions
del x_train
gc.collect()

#check to make sure the shape for bottleneck features is as expected
VGG16_train.shape

# <markdowncell>

# Now we also need to extract bottleneck features from the test images

# <codecell>

# We should load in the test tensors now since we need to extract bottleneck features from them
x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)

# Save test preprocessed tensors
save_array('x_test.bc', x_test)
save_array('x_test_filename.bc', x_test_filename)

# Extract bottleneck features from the VGG16 model
VGG16_test = model.predict(x_test)

#save the bottleneck features to disk for later access
save_array('VGG16_test.bc', VGG16_test)

# assign the loaded bottleneck features
VGG16_test = load_array('VGG16_test.bc')

# For now we just need training bottlenecks, we can load in the test data later when we make predictions
del x_test, x_test_filename, VGG16_test
gc.collect()

#check to make sure the shape for bottleneck features is as expected
VGG16_test.shape

# <codecell>

# Repeat the process for the additional test images
x_test_additional, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)

# Save test preprocessed tensors
save_array('x_test_additional.bc', x_test_additional)
save_array('x_test_filename_additional.bc', x_test_filename_additional)

# Extract bottleneck features from the VGG16 model
VGG16_test_additional = model.predict(x_test_additional)

#save the bottleneck features to disk for later access
save_array('VGG16_test_additional.bc', VGG16_test_additional)

# assign the loaded bottleneck features
VGG16_test_additional = load_array('VGG16_test_additional.bc')

del x_test_additional, x_test_filename_additional, VGG16_test_additional
gc.collect()

#check to make sure the shape for bottleneck features is as expected
VGG16_test_additional.shape

# <markdowncell>

# ## Load already extracted bottleneck features
# Run this code block if already have bottleneck features that we want to load in

# <codecell>

VGG16_train = load_array('VGG16_train.bc')

VGG16_train.shape

# <markdowncell>

# ## Create a checkpoint
# 
# Creating a checkpoint saves the best model weights across all epochs in the training process. This ensures that we will always use only the best weights when making our predictions on the test set rather than using the default which takes the final score from the last epoch. 

# <codecell>

from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

# <markdowncell>

# ## Choose Hyperparameters
# 
# Choose your hyperparameters below for training. 

# <codecell>

validation_split_size = 0.2
batch_size = 128

# <markdowncell>

# ## Define and Train model - Transfer Learning
# 
# Here we define the top model for our transfer learning model and begin training. 
# 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined `lr_1, lr_2, lr_3` and corresponding number of epochs for each `epochs_1, epochs_2, epochs_3`. Feel free to change these values if you like or just use the defaults. 

# <codecell>

# For Dense layers, choose to have one or two layers with configurable num of filters. 

classifier = AmazonKerasClassifier()
classifier.add_flatten_layerTL(input_shape=VGG16_train.shape[1:])
classifier.add_ann_layer(512, len(y_map))
#classifier.add_ann_layer_double(512, len(y_map))

train_losses, val_losses = [], []
epochs_arr = [10, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(VGG16_train, y_train, learn_rate, epochs, 
                                                                           batch_size, validation_split_size=validation_split_size, 
                                                                           train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses

# <markdowncell>

# ## Define and Train model - CNN from scratch
# 
# Here we define the model for our CNN built from scratch and begin training. 
# 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined `lr_1, lr_2, lr_3` and corresponding number of epochs for each `epochs_1, epochs_2, epochs_3`. Feel free to change these values if you like or just use the defaults. 

# <codecell>

classifier = AmazonKerasClassifier()
classifier.add_conv_layer(img_resize)
classifier.add_flatten_layer()
classifier.add_ann_layer(len(y_map))

train_losses, val_losses = [], []
epochs_arr = [10, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(x_train, y_train, learn_rate, epochs, 
                                                                           batch_size, validation_split_size=validation_split_size, 
                                                                           train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses

# <markdowncell>

# ## Load Best Weights

# <markdowncell>

# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training

# <codecell>

classifier.load_weights("weights.best.hdf5")
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

# Look at our fbeta_score

# <codecell>

fbeta_score

# <markdowncell>

# Before launching our predictions lets preprocess the test data and delete the old training data matrices

# <markdowncell>

# ## Make Predictions - Transfer Learning
# If using transfer learning, run these two code blocks to make our predictions then skip to `Determine Threshold`

# <codecell>

del VGG16_train, y_train
gc.collect()

# Load test bottleneck tensors
VGG16_test = load_array('VGG16_test.bc')

# Predict the labels of our x_test images
predictions = classifier.predict(VGG16_test)

# <codecell>

del VGG16_test
gc.collect()

# Load test bottleneck tensors
VGG16_test_additional = load_array('VGG16_test_additional.bc')

# Predict the labels of our x_test additional images
new_predictions = classifier.predict(VGG16_test_additional)

del VGG16_test_additional
gc.collect()

# Combine our predictions
predictions = np.vstack((predictions, new_predictions))

# Load back in the test filenames
x_test_filename = load_array('x_test_filename1.bc')
x_test_filename_additional = load_array('x_test_filename2.bc')

x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              predictions[0]))

# <markdowncell>

# ## Make Predictions - CNN from scratch
# Run these two code blocks if using the CNN from scratch model

# <codecell>

del x_train, y_train
gc.collect()

x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)

# Save test preprocessed tensors so we can just load them in next time directly
save_array('x_test.bc', x_test)
save_array('x_test_filename.bc', x_test_filename)
x_test = load_array('x_test.bc')
x_test_filename = load_array('x_test_filename.bc')

# Predict the labels of our x_test images
predictions = classifier.predict(x_test)

# <markdowncell>

# Now lets launch the predictions on the additionnal dataset (updated on 05/05/2017 on Kaggle)

# <codecell>

del x_test
gc.collect()

x_test, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)
new_predictions = classifier.predict(x_test)

del x_test
gc.collect()
predictions = np.vstack((predictions, new_predictions))
x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              predictions[0]))

# <markdowncell>

# ## Determine Thresholds - Both Transfer Learning/CNN from Scratch
# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class.
# 
# To do so we will take the median value of each classes.

# <codecell>

# For now we'll just put all thresholds to 0.17 since its optimal
thresholds = [0.17] * len(labels_set)

# TODO complete
tags_pred = np.array(predictions).T
_, axs = plt.subplots(5, 4, figsize=(15, 20))
axs = axs.ravel()

for i, tag_vals in enumerate(tags_pred):
    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])

# <markdowncell>

# Now lets map our predictions to their tags and use the thresholds we just retrieved

# <codecell>

predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)

# <markdowncell>

# Finally lets assemble and visualize our prediction for the test dataset

# <codecell>

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

# <codecell>

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
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

# That's it, we're done!
