import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from modules.dataset_loader import DatasetLoader
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class VoteModule:

    DATASET_PATH = "../datasets/Original/"
    HYPEROPT_RESULTS_FOLDER = "../best_hyperopt_results/"

    def add_padding(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        max_len = len(X[0])
        for row in X:
            if len(row) > max_len:
                max_len = len(row)

        X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float64')
        return X, y

    def load_dataset(self):
        DATASET_PATH = self.DATASET_PATH + 'MFCC/'
        dataset_loader = DatasetLoader(DATASET_PATH)
        mfcc_features, y = dataset_loader.get_dataset()

        DATASET_PATH = self.DATASET_PATH + 'Prosody/'
        dataset_loader = DatasetLoader(DATASET_PATH)
        prosody_features, y = dataset_loader.get_dataset()

        new_dataset = []
        for index in range(0, len(mfcc_features)):
            new_instance = []
            for row_index in range(0, len(mfcc_features[index])):
                new_row = np.concatenate(
                    (mfcc_features[index][row_index],
                    prosody_features[index][row_index]),
                    axis= None
                )
                new_instance.append(new_row)
            new_dataset.append(new_instance)

        X = new_dataset

        DATASET_PATH = self.DATASET_PATH + 'Chroma/'
        dataset_loader = DatasetLoader(DATASET_PATH)
        chroma_features, y = dataset_loader.get_dataset()

        new_dataset = []
        for index in range(0, len(chroma_features)):
            new_instance = []
            for row_index in range(0, len(chroma_features[index])):
                new_row = np.concatenate(
                    (X[index][row_index],
                    chroma_features[index][row_index]),
                    axis= None
                )
                new_instance.append(new_row)
            new_dataset.append(new_instance)

        X = np.asarray(new_dataset)

        return X, y

    def get_best_models(self):
        models_file = []
        results_dir = os.listdir(self.HYPEROPT_RESULTS_FOLDER)
        for filename in results_dir:
            if '.h5' in filename:
               models_file.append(self.HYPEROPT_RESULTS_FOLDER + filename)
        models = dict()
        for model_file in models_file:
             models[model_file.split('/')[-1][:-3]] = keras.models.load_model(model_file)

        return models

    def evaluate(self):
        dataset_loader = DatasetLoader()
        X,y = dataset_loader.compose_complete_dataset()
        y = [inst[0] for inst in y]
        X, y = add_padding(X,y)
        y = to_categorical(y)
        num_rows = X[0].shape[0]
        num_columns = X[0].shape[1]
        num_channels = 1
        X = X.reshape(X.shape[0], num_rows, num_columns, num_channels)

        models = sef.get_best_models()
        predictions = dict()
        for model in models.keys():
            predictions[model] = models[model].predict(X)
        
        import pdb; pdb.set_trace()
            

















