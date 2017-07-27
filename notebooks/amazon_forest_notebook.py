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

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import vgg16
import data_helper
from data_helper import AmazonPreprocessor
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

print("X_train/y_train length: {}/{}".format(len(preprocessor.X_train), len(preprocessor.y_train)))
print("X_val/y_val length: {}/{}".format(len(preprocessor.X_val), len(preprocessor.y_val)))
print("X_test/X_test_filename length: {}/{}".format(len(preprocessor.X_test), len(preprocessor.X_test_filename)))
preprocessor.y_map

# <markdowncell>

# # Funetuning
# 
# Here we define the model for finetuning

# <codecell>

model = vgg16.create_model(img_dim=(128, 128, 3))
model.summary()

# <markdowncell>

# ## Fine-tune conv layers
# We will now finetune all layers in the VGG16 model. 

# <codecell>

loss_rdir = 'weights/saved_loss'
or_weights_path = "weights/best_original_weights.hdf5"
batch_size = 128
epochs = 25
train_losses, val_losses = [], []
X_train, y_train = preprocessor.X_train, preprocessor.y_train
X_val, y_val = preprocessor.X_val, preprocessor.y_val

if os.path.exists(loss_rdir) and os.path.exists(or_weights_path):
    train_losses, val_losses = bcolz.open(loss_rdir)
    print("Weights will be loaded from disk")
else:
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
                 ModelCheckpoint(filepath=or_weights_path, verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='auto')]
    
    train_generator = preprocessor.get_train_generator(X_train, y_train, batch_size, augment_data=True)
    steps = len(X_train) / batch_size

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])
    history = model.fit_generator(train_generator, steps, epochs=epochs, verbose=1, 
                        validation_data=(X_val, y_val), callbacks=callbacks)

    train_losses, val_losses = history.history['loss'], history.history['val_loss']
    c = bcolz.carray([train_losses, val_losses], rootdir=loss_rdir, mode='w')
    c.flush()
    # Here we check if our validation set loss is consistent. Otherwise it would mean that it is too small.
    vgg16.show_validation_set_consistency(model, X_val, y_val)

# <markdowncell>

# ## Visualize Loss Curve

# <codecell>

plt.plot(train_losses)
plt.plot(val_losses)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left');

# <markdowncell>

# ## Load Best Weights

# <codecell>

model.load_weights(or_weights_path)
print("Weights loaded")

# <markdowncell>

# ## Check Fbeta Score

# <codecell>

fbeta_score = vgg16.fbeta(model, X_val, y_val)

fbeta_score

# <markdowncell>

# # Pseudo labeling

# <markdowncell>

# To do pseudo labeling we'll start by merging our training data with a subset of the test set.
# Below we merge the **paths** where the files are located, not the actual data. Using directly the data would eat up too much memory.

# <codecell>

#
# TODO: remove dropout
#

# Take a subset of the test set
x_subset = preprocessor.X_test_filename[:20000]

# We predict the labels of our preprocessor.X_test_filename set
pseudo_predictions, pseudo_paths = vgg16.predict(model, preprocessor, x_subset, batch_size=batch_size)

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
combo_x_pseudo = np.concatenate([preprocessor.X_train[:50000], [v[0] for v in undersampled_df.iloc[:, 0:1].values.tolist()]])

# Now we concatenate our predictions to our training labels
combo_y_pseudo = np.concatenate([preprocessor.y_train[:50000], undersampled_df.iloc[:, 1:].values.tolist()])

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
train_generator = preprocessor.get_train_generator(combo_x_pseudo, combo_y_pseudo, batch_size, augment_data=True)
steps = len(X_train) / batch_size
    
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(train_generator, steps, epochs=epochs, verbose=1, 
                              validation_data=(X_val, y_val), callbacks=callbacks)
train_losses, val_losses = history.history['loss'], history.history['val_loss']

# <markdowncell>

# ## Load Best Weights

# <codecell>

# Keep the weights of the original classifier merged with the pseudo labeling
# classifier.load_weights(or_weights_path)
# print("Weights loaded")

# <markdowncell>

# ## Plot the loss change

# <codecell>

plt.plot(train_losses)
plt.plot(val_losses)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left');

# <markdowncell>

# And finally get the final f2 score

# <codecell>

fbeta_score = vgg16.fbeta(model, X_val, y_val)

fbeta_score

# <markdowncell>

# ## Make final predictions

# <codecell>

predictions, x_test_filename = vgg16.predict(model, preprocessor, preprocessor.X_test_filename, batch_size)
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions ({}) entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              x_test_filename[0], predictions[0]))

# <markdowncell>

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class

# <codecell>

thresholds = [0.2] * len(labels_set)

# <markdowncell>

# Now lets map our predictions to their tags by using the thresholds

# <codecell>

predicted_labels = vgg16.map_predictions(preprocessor, predictions, thresholds)

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

# <markdowncell>

# #### That's it, we're done!
