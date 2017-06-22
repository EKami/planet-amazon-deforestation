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
import keras

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

img_resize = (128, 128) # The resize size of each image

# <markdowncell>

# ## Define hyperparameters

# <codecell>

validation_split_size = 0.2
batch_size = 128

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

x_train, y_train, y_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
# Free up all available memory space after this heavy operation
gc.collect();

# <codecell>

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
y_map

# <markdowncell>

# ## Create a checkpoint

# <markdowncell>

# Creating a checkpoint saves the best model weights across all epochs in the training process. This ensures that we will always use only the best weights when making our predictions on the test set rather than using the default which takes the final score from the last epoch. 

# <codecell>

from keras.callbacks import ModelCheckpoint

filepath="weights.92.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# <markdowncell>

# ## Create the neural network definition

# <codecell>

learn_rate = 0.002
epochs = 10
classifier = AmazonKerasClassifier()
classifier.add_conv_layer(img_resize)
classifier.add_flatten_layer()
classifier.add_ann_layer(len(y_map))
train_losses, val_losses, fbeta_score = classifier.train_model(x_train, y_train, learn_rate, epochs, batch_size, validation_split_size=validation_split_size, train_callbacks=[checkpoint])

# <codecell>

learn_rate = 0.0002
epochs = 5
train_losses, val_losses, fbeta_score = classifier.train_model(x_train, y_train, learn_rate, epochs, batch_size, validation_split_size=validation_split_size, train_callbacks=[checkpoint])

# <codecell>

learn_rate = 0.00002
epochs = 3
train_losses, val_losses, fbeta_score = classifier.train_model(x_train, y_train, learn_rate, epochs, batch_size, validation_split_size=validation_split_size, train_callbacks=[checkpoint])

# <markdowncell>

# ## Load Best Weights

# <codecell>

classifier.load_weights("weights.best128.hdf5")
print("Weights loaded")

# <codecell>

# by saving the model we can access it later (almost like a timewarp to get back to the exact same trained model)
classifier.save_model('128_model.h5')

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

# # Pseudo Labeling

# <codecell>

# we need to free up all memory we can for pseudo labeling
del classifier
gc.collect()

# <codecell>

# we need to use X_train instead of x_train for pseudo labeling, otherwise we will potentially have duplicate images in validation set
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

# <codecell>

# first we load in the test data
x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)

# we don't want to use all of the test data, only a subset of it
x_test_trunc = x_test[:20000]

del x_test, x_train
gc.collect()

#We can now concatenate our pseudo features with the training features
x_pseudo = np.concatenate([X_train, x_test_trunc])
del X_train
gc.collect
x_pseudo.shape

# <codecell>

# It will be better for memory allocation and more convenient for us to work from our loaded model file here on out. 
from keras.models import load_model

model = load_model('128_model.h5')

# <codecell>

# Now we predict the labels of our x_test images
predictions = model.predict(x_test_trunc)

del x_test_trunc
gc.collect()

predictions_list = [[1 if y > 0.2 else 0 for y in x] for x in predictions]

#Next we concatenate our predictions to our training labels
y_pseudo = np.concatenate([y_train, predictions_list])

#Now we can safely delete the old training data matrices
del y_train, predictions, predictions_list
gc.collect()

y_pseudo.shape

# <markdowncell>

# # Re-train with Pseudo Labeling

# <codecell>

from keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

filepath="128_model_pseudo.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# <codecell>

from keras.optimizers import Adamax

opt = Adamax(lr=0.002)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy'])

# <codecell>

model.fit(x_pseudo, y_pseudo, epochs=5, verbose=1, callbacks=[checkpoint, earlyStopping],
              validation_data=(X_valid, y_valid))

# <codecell>

model.load_weights("128_model_pseudo.hdf5")
print("Weights loaded")

# <codecell>

# continue training at lower learning rate
opt = Adamax(lr=0.0002)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy'])

# <codecell>

model.fit(x_pseudo, y_pseudo, epochs=3, verbose=1, callbacks=[checkpoint, earlyStopping],
              validation_data=(X_valid, y_valid))

# <codecell>

model.load_weights("128_model_pseudo.hdf5")
print("Weights loaded")

# <codecell>

# save our model so we can always go back to it later
model.save('128_model_pseudo.h5')

# <codecell>

from sklearn.metrics import fbeta_score

def get_fbeta_score(model, X_valid, y_valid):
    p_valid = model.predict(X_valid)
    return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

fbeta_score = get_fbeta_score(model, X_valid, y_valid)

fbeta_score

# <markdowncell>

# # Re-train with Augmentation

# <codecell>

filepath="128_model_pseudo_augmented.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# <codecell>

from keras.preprocessing.image import ImageDataGenerator

#Image Augmentation
datagen = ImageDataGenerator(
                        width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
                        height_shift_range=0.1,  # randomly shift images vertically (10% of total height)   
                        horizontal_flip=True,
                        vertical_flip=True) # randomly flip images horizontally
datagen.fit(x_pseudo)
batches = datagen.flow(x_pseudo, y_pseudo, batch_size=64)

# <codecell>

model.fit_generator(batches, epochs=5, verbose=1, callbacks=[checkpoint, earlyStopping], 
                    validation_data=(X_valid, y_valid), steps_per_epoch=x_pseudo.shape[0] // 64)

# <codecell>

model.load_weights("128_model_pseudo_augmented.hdf5")
print("Weights loaded")

# <codecell>

# Continue training with lower learning rate
opt = Adamax(lr=0.0002)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy'])

# <codecell>

model.fit_generator(batches, epochs=3, verbose=1, callbacks=[checkpoint, earlyStopping], 
                    validation_data=(X_valid, y_valid), steps_per_epoch=x_pseudo.shape[0] // 64)

# <codecell>

model.load_weights("128_model_pseudo_augmented.hdf5")
print("Weights loaded")

# <codecell>

# save our model so we can always go back to it later
model.save('128_model_pseudo_augmented.h5')

# <markdowncell>

# # Resume Training without pseudo labels

# <codecell>

del x_pseudo, y_pseudo
gc.collect()

x_train, y_train, y_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
# Free up all available memory space after this heavy operation
gc.collect();

X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

del x_train
gc.collect()

# <codecell>

filepath="128_model_pseudo_augmented_resume.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# <codecell>

datagen = ImageDataGenerator(
                        width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
                        height_shift_range=0.1,  # randomly shift images vertically (10% of total height)   
                        horizontal_flip=True,
                        vertical_flip=True) # randomly flip images horizontally
datagen.fit(X_train)
batches = datagen.flow(X_train, y_train, batch_size=64)

# <codecell>

model.fit_generator(batches, epochs=5, verbose=1, callbacks=[checkpoint, earlyStopping], 
                    validation_data=(X_valid, y_valid), steps_per_epoch=X_train.shape[0] // 64)

# <codecell>

model.load_weights("128_model_pseudo_augmented_resume.hdf5")
print("Weights loaded")

# <codecell>

model.save('128_model_pseudo_augmented_resume.h5')

# <markdowncell>

# Before launching our predictions lets preprocess the test data and delete the old training data matrices

# <codecell>

del X_train, y_train
gc.collect()

x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)
# Predict the labels of our x_test images
predictions = model.predict(x_test)

# <markdowncell>

# Now lets launch the predictions on the additionnal dataset (updated on 05/05/2017 on Kaggle)

# <codecell>

del x_test
gc.collect()

x_test, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)
new_predictions = model.predict(x_test)

del x_test
gc.collect()
predictions = np.vstack((predictions, new_predictions))
x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              predictions[0]))

# <markdowncell>

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class.
# 
# To do so we will take the median value of each classes.

# <codecell>

# For now we'll just put all thresholds to 0.2 
#thresholds = [0.17] * len(labels_set)

#optimal thresholds
thresholds = [0.2, 0.2, 0.2, 0.2, 0.2, 0.14, 0.08, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


# TODO complete
tags_pred = np.array(predictions).T
_, axs = plt.subplots(5, 4, figsize=(15, 20))
axs = axs.ravel()

for i, tag_vals in enumerate(tags_pred):
    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])

# <markdowncell>

# Now lets map our predictions to their tags and use the thresholds we just retrieved

# <codecell>

def map_predictions(predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

# <codecell>

predicted_labels = map_predictions(predictions, y_map, thresholds)

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

# <markdowncell>

# That's it, we're done!
