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
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.callbacks import ModelCheckpoint, CSVLogger, History

import data_helper
from data_helper import AmazonPreprocessor
from keras_helper import AmazonKerasClassifier, Fbeta
from kaggle_data.downloader import KaggleDataDownloader

# Finetuning Imports
import densenet169
from densenet169 import *
from keras import optimizers
from keras.models import Model
from sklearn.metrics import fbeta_score
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adamax, SGD


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

img_resize = (256, 256) # The resize size of each image ex: (64, 64) or None to use the default image size
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

# ## Callbacks
# 
# 
# 
# __Checkpoint__
# 
# Saves the best model weights across all epochs in the training process.
# 
# __CSVLogger__
# 
# Creates a file with a log of loss and accuracy per epoch
# 
# __FBeta__
# 
# Calculates fbeta_score after each epoch during training

# <codecell>

checkpoint = ModelCheckpoint(filepath="weights.best.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

# creates a file with a log of loss and accuracy per epoch
csv = CSVLogger(filename='training.log', append=True)

# Calculates fbeta_score after each epoch during training
fbeta = Fbeta()

# Tracks training history for later visualization
history = History()

# <markdowncell>

# # Funetuning
# 
# Here we define the model for finetuning

# <codecell>

import densenet169
from densenet169 import *
base_model = densenet169.create_model()

# Create new classifier layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.5)(x)

predictions = Dense(17, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all bottom layers, only train classifier first
for layer in base_model.layers:
    layer.trainable = False

# <markdowncell>

# ## Train Classifier
# We first start by training only the classifier.

# <codecell>

#Choose Hyperparameters & Compile
batch_size = 64
learn_rate = 0.002
opt = Adamax(lr=learn_rate)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics = ['accuracy'])

# Define generator and validation set
train_generator = preprocessor.get_train_generator(batch_size)
X_val, y_val = preprocessor.X_val, preprocessor.y_val


# Training
hist = model.fit_generator(train_generator, len(preprocessor.X_train) / batch_size,
                    epochs=20, verbose=1, validation_data=(X_val, y_val),
                    callbacks=[history, checkpoint, csv, fbeta])

print(fbeta.fbeta)

# <markdowncell>

# ## Load Best Weights

# <markdowncell>

# Here you should load back in the best weights that were automatically saved by ModelCheckpoint during training

# <codecell>

model.load_weights("weights.best.hdf5")
print("Weights loaded")

# <markdowncell>

# ## Monitor the results

# <markdowncell>

# Check that we do not overfit by plotting the losses of the train and validation sets

# <codecell>

plt.plot(range(epochs), hist.history[
         'val_loss'], 'b-', label='Val Loss')
plt.plot(range(epochs), hist.history[
         'loss'], 'g--', label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# <markdowncell>

# Look at our overall fbeta_score

# <codecell>

def get_fbeta_score(model, X_valid, y_valid):
    p_valid = model.predict(X_valid)
    return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

get_fbeta_score(model, X_val, y_val)

# <markdowncell>

# ## Fine-tune conv layers
# Now that we have a trained classifier, we can unfreeze all other layers in the model (or keep certain layers still frozen) and retrain. 

# <codecell>

# We can either freeze the first X number of layers and unfreeze only certain layers or we can unfreeze all layers:

#for layer in model.layers[:69]:
#    layer.trainable = False
#for layer in model.layers[69:]:
#    layer.trainable = True

for layer in model.layers:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics = ['accuracy'])

# Let's create a new log file for this one
csv = CSVLogger('training_2.log')


# We may need to use a reduced batch size since using the whole model here
batch_size = 32
train_generator = preprocessor.get_train_generator(batch_size)


hist_bottom = model.fit_generator(train_generator, len(preprocessor.X_train) / batch_size,
                    epochs=100, verbose=1, validation_data=(X_val, y_val),
                    callbacks=[history, checkpoint, csv, fbeta])

print(fbeta.fbeta)

# <codecell>

model.load_weights("weights.best.hdf5")
print("Weights loaded")

# <codecell>

get_fbeta_score(model, X_val, y_val)

# <codecell>

p_valid = model.predict(X_val)

def optimise_f2_thresholds(y, p, verbose=True, resolution=75):
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score
    x = [0.2]*17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)
        
    return x

# <codecell>

optimise_f2_thresholds(y_val, np.array(p_valid), verbose=True, resolution=75)

# <markdowncell>

# ## Make predictions

# <codecell>

def predict(batch_size=128):
        """
        Launch the predictions on the test dataset as well as the additional test dataset
        :return:
            predictions_labels: list
                An array containing 17 length long arrays
            filenames: list
                File names associated to each prediction
        """
        generator = preprocessor.get_prediction_generator(batch_size)
        predictions_labels = model.predict_generator(generator=generator, verbose=1,
                                                     steps=len(preprocessor.X_test_filename) / batch_size)
        assert len(predictions_labels) == len(preprocessor.X_test), \
            "len(predictions_labels) = {}, len(preprocessor.X_test) = {}".format(
                len(predictions_labels), len(preprocessor.X_test))
        return predictions_labels, np.array(preprocessor.X_test)

# <codecell>

predictions, x_test_filename = predict(batch_size)
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions ({}) entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              x_test_filename[0], predictions[0]))

# <codecell>

# Create functions to save and load tensor arrays.
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

save_array('dn169_predictions.bc', predictions)

# <markdowncell>

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class

# <codecell>

# Use thresholds generated by `optimise_f2_thresholds`
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

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

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
