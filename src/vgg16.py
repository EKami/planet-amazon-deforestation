import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score

def create_model(img_dim=(128, 128, 3)):
    input_tensor = Input(shape=img_dim)
    base_model = VGG16(include_top=False,
                       weights='imagenet',
                       input_shape=img_dim)
    
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(17, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def predict(model, preprocessor, batch_size=128):
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

def map_predictions(preprocessor, predictions, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [preprocessor.y_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

def fbeta(model, X_valid, y_valid):
    p_valid = model.predict(X_valid)
    return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
