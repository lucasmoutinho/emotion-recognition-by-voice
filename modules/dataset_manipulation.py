import math

import random

from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from modules.dataset_loader import DatasetLoader
import numpy as np

class DatasetManipulation:
    EMOTION_NUMBERS = [0, 1, 2, 3, 4, 5, 6]

    def add_padding(self, X, y, max_len):
        X = np.asarray(X)
        y = np.asarray(y)

        X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float64')
        return X, y

    def wanted_emotion_indexes(self, wanted_emotion, y_all):
        # selection of indexes of desired emotion
        indexes = []
        for i in range(0, len(y_all)):
            if y_all[i][0] == wanted_emotion:
                indexes.append(i)
        return indexes

    def apply_smote(self, X, y):
        shape_0 = X.shape[0]
        shape_1 = X.shape[1]
        shape_2 = X.shape[2]
        X = X.reshape(shape_0, shape_1 * shape_2)

        # Apply SMOTE
        smt = SMOTE()
        try:
            X, y = smt.fit_sample(X, y)
        except:
            pass

        # Reshaping back to original shape dimensions 1 and 2
        X = X.reshape(X.shape[0], shape_1, shape_2)

        return X, y

    def binary_emotion_label(self, wanted_emotion, y_all):
        for i in range(0, len(y_all)):
            if y_all[i][0] == wanted_emotion:
                y_all[i][0] = 1
            else:
                y_all[i][0] = 0
        return y_all

    def get_train_test_split_for_emotion(self, wanted_emotion):
        X_all, y_all = DatasetLoader().compose_complete_dataset()
        y_list = []

        # Create list of indexes for each emotion
        for emotion in self.EMOTION_NUMBERS:
            current_emotion_indexes = self.wanted_emotion_indexes(emotion, y_all)
            y_list.append(current_emotion_indexes)

        # Create a list of wanted indexes containing all indexes for the desired emotion.
        # That number will be the first half of the list, the other half will contain
        # A slice of random indexes from the other emotions. This will result in a wanted list
        # with half of indexes of the desired emotion and other half with other emotions
        indexes_wanted = y_list[wanted_emotion]
        half_dataset_number = len(y_list[wanted_emotion])
        other_emotions_number = math.ceil(
            half_dataset_number / (len(self.EMOTION_NUMBERS) - 1))
        for emotion in self.EMOTION_NUMBERS:
            if emotion != wanted_emotion:
                indexes_wanted = indexes_wanted + random.sample(
                    y_list[emotion], other_emotions_number)

        # Adjust the labels as being the desired emotion or not
        y_all = self.binary_emotion_label(wanted_emotion, y_all)

        max_len = len(np.asarray(X_all)[0])
        for row in np.asarray(X_all):
            if len(row) > max_len:
                max_len = len(row)

        # Create X and y
        X = np.take(X_all, indexes_wanted)
        y = [inst_y[0] for inst_y in y_all]
        y = np.take(y, indexes_wanted)
        X = np.asarray(X)
        X, y = self.add_padding(X, y, max_len)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            stratify=y,
                                                            test_size=0.3)

        X_test, y_test = self.apply_smote(X_test, y_test)
        X_train, y_train = self.apply_smote(X_train, y_train)

        y_test_labels = y_test
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        num_rows = X[0].shape[0]
        num_columns = X[0].shape[1]
        num_channels = 1
        X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns,
                                  num_channels)
        X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns,
                                num_channels)

        return X_train, y_train, X_test, y_test

        


