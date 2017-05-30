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

# ### Import required modules

# <codecell>

import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint

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

# ### Download the competition files
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

# ### Inspect image labels
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

# ### Images
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

# ### Image Resize
# Define the dimensions of the image data trained by the network. Due to memory constraints we can't load in the full size 256x256 jpg images. Recommended resized images could be 32x32, 64x64, or 128x128.

# <codecell>

img_resize = (64, 64) # The resize size of each image

# <markdowncell>

# ## Part 1: Weather labels

# <markdowncell>

# ### Create a checkpoint
# 
# Creating a checkpoint saves the best model weights across all epochs in the training process. This ensures that we will always use only the best weights when making our predictions on the test set rather than using the default which takes the final score from the last epoch. 

# <codecell>

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

# <markdowncell>

# ### Data preprocessing
# Preprocess the weather data in order to fit it into the Keras model.

# <codecell>

x_weather_train, y_weather_train, y_weather_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, 
                                                                                    label_filter='weather', 
                                                                                    img_resize=img_resize)
x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)
x_test_add, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)
# Free up all available memory space after this heavy operation
gc.collect();

# <codecell>

print("x_weather_train shape: {}".format(x_weather_train.shape))
print("y_weather_train shape: {}".format(y_weather_train.shape))
y_weather_map

# <markdowncell>

# ### Choose Hyperparameters
# 
# Choose your hyperparameters below for training.
# 
# Note that we have created a learning rate annealing schedule with a series of learning rates as defined in the array `learn_rates` and corresponding number of epochs for each `epochs_arr`. Feel free to change these values if you like or just use the defaults.

# <codecell>

validation_split_size = 0.2
batch_size = 128
epochs_arr = [5, 2, 2]
learn_rates = [0.001, 0.0001, 0.00001]

# <markdowncell>

# ### Define and Train model for the weather labels
# 
# Here we'll create a model with a softmax classifier as output on the weather labels (**only one weather** label can be present) for one prediction.

# <codecell>

weather_classifier = AmazonKerasClassifier('softmax')
weather_classifier.add_conv_layer(img_resize)
weather_classifier.add_flatten_layer()
weather_classifier.add_ann_layer(len(y_weather_map))

train_losses, val_losses = [], []
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = weather_classifier.train_model(x_weather_train, y_weather_train, learn_rate, epochs, 
                                                                           batch_size, validation_split_size=validation_split_size, 
                                                                           train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses

# <markdowncell>

# ### Monitor the results
# Check that we do not overfit by plotting the losses of the train and validation sets

# <codecell>

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend();

# <markdowncell>

# ### Load Best Weights
# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training

# <codecell>

weather_classifier.load_weights("weights.best.hdf5")
print("Weights loaded")

# <markdowncell>

# ### Predict and save the predictions
# Here we will store our predictions in the `weather_predictions` variable so that we can retrieve it at the end of this notebook.
# 
# /!\ Don't forget the predictions on the additionnal dataset (updated on 05/05/2017 on Kaggle)

# <codecell>

# Predict the labels of our x_test images
weather_predictions = weather_classifier.predict(x_test)
weather_predictions_add = weather_classifier.predict(x_test_add)

weather_predictions = np.vstack((weather_predictions, weather_predictions_add))

weather_classifier.close()

# <markdowncell>

# ### Free used resources
# Free the used resources to not use too much RAM

# <codecell>

del x_weather_train, y_weather_train
gc.collect();

# <markdowncell>

# ## Part 2: Lands labels

# <markdowncell>

# ### Create a checkpoint
# 
# Creating a checkpoint saves the best model weights across all epochs in the training process. This ensures that we will always use only the best weights when making our predictions on the test set rather than using the default which takes the final score from the last epoch. 

# <codecell>

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

# <markdowncell>

# ### Data preprocessing
# Preprocess the lands data in order to fit it into the Keras model.

# <codecell>

x_land_train, y_land_train, y_land_map = data_helper.preprocess_train_data(train_jpeg_dir, 
                                                                           train_csv_file, 
                                                                           label_filter='land', 
                                                                           img_resize=img_resize)
# Free up all available memory space after this heavy operation
gc.collect();

# <codecell>

print("x_land_train shape: {}".format(x_land_train.shape))
print("y_land_train shape: {}".format(y_land_train.shape))
y_land_map

# <markdowncell>

# ### Choose Hyperparameters
# 
# Choose your hyperparameters below for training.

# <codecell>

validation_split_size = 0.2
batch_size = 128
epochs_arr = [10, 5, 5]
learn_rates = [0.001, 0.0001, 0.00001]

# <markdowncell>

# ### Define and Train model
# 
# Here we define another model with a sigmoid output (**one or more** land labels can be present) and begin training. 

# <codecell>

lands_classifier = AmazonKerasClassifier('sigmoid')
lands_classifier.add_conv_layer(img_resize)
lands_classifier.add_flatten_layer()
lands_classifier.add_ann_layer(len(y_land_map))

train_losses, val_losses = [], []
for learn_rate, epochs in zip(learn_rates, epochs_arr):
    tmp_train_losses, tmp_val_losses, fbeta_score = lands_classifier.train_model(x_land_train, y_land_train, 
                                                                                 learn_rate, epochs, batch_size, 
                                                                                 validation_split_size=validation_split_size, 
                                                                                 train_callbacks=[checkpoint])
    train_losses += tmp_train_losses
    val_losses += tmp_val_losses

# <markdowncell>

# ### Load Best Weights
# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training

# <codecell>

lands_classifier.load_weights("weights.best.hdf5")
print("Weights loaded")

# <markdowncell>

# ### Monitor the results
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

# ### Predict and save the predictions
# Here we will store our predictions in the `lands_predictions` variable so that we can retrieve it at the end of this notebook.
# 
# /!\ Don't forget the predictions on the additionnal dataset (updated on 05/05/2017 on Kaggle)

# <codecell>

lands_predictions = lands_classifier.predict(x_test)
add_lands_predictions = lands_classifier.predict(x_test_add)

lands_predictions = np.vstack((lands_predictions, add_lands_predictions))
x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
lands_classifier.close()

print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(lands_predictions.shape, 
                                                                                       x_test_filename.shape, 
                                                                                       lands_predictions[0]))

# <markdowncell>

# ### Free used resources
# Free the used resources to not use too much RAM

# <codecell>

del x_land_train, y_land_train
gc.collect()

# <markdowncell>

# ### Find prediction thresholds
# Before mapping our predictions to their appropriate labels we need to figure out what sigmoid threshold to take for each class.

# <codecell>

# For now we'll just put all thresholds to 0.2 
thresholds = [0.2] * len(labels_set)

# TODO complete
tags_pred = np.array(lands_predictions).T
_, axs = plt.subplots(4, 4, figsize=(15, 20))
axs = axs.ravel()

for i, tag_vals in enumerate(tags_pred):
    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_land_map[i])

# <markdowncell>

# ## Part 3: Finalize predictions

# <markdowncell>

# ### Get labels out of prediction vectors

# <codecell>

land_labels = lands_classifier.map_predictions(lands_predictions, y_land_map, thresholds)
weather_labels = weather_classifier.map_predictions(weather_predictions, y_weather_map)

# <markdowncell>

# ### Combine weather and land predictions

# <codecell>

# TODO check the self.classifier.compile(loss='binary_crossentropy') in classifier.train_model()

land_tags_list = [None] * len(land_labels)
weather_tags_list = [None] * len(weather_labels)

for i, tags in enumerate(land_labels):
    land_tags_list[i] = ' '.join(map(str, tags))
    
for i, tags in enumerate(weather_labels):
    weather_tags_list[i] = ' '.join(map(str, tags))

# Create the dataframe entries with the land label
final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, land_tags_list)]

for i, entry in enumerate(final_data):
    entry[1] = entry[1] + " " + weather_tags_list[i]
    
print(final_data[:5])

# <markdowncell>

# Finally lets assemble and visualize our prediction for the test dataset

# <codecell>

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
final_df.head()

# <codecell>

tags_s = pd.Series(list(chain.from_iterable(land_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');

# <markdowncell>

# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...

# <markdowncell>

# And save it to a submission file

# <codecell>

final_df.to_csv('../submission_file.csv', index=False)

# <markdowncell>

# That's it, we're done!
