import functools

import keras
import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from time_series_dataset_loader import TimeSeriesDatasetLoader


# DATASET_PATH = '../src/Features/Original/MFCC_2/'
#         checkpoint_file = '../models/model_checkpoints/original_window_2.h5'

class TimeSeriesModel:

    def __init__(self, dataset_path, checkpoint_filename, type_='default', activation_layer_size=7):
        self.dataset_path = dataset_path
        self.checkpoint_filename = checkpoint_filename
        self.type_ = type_
        self.activation_layer_size = activation_layer_size

    def run(self):
        try:
            dataset_loader = TimeSeriesDatasetLoader(self.dataset_path)

            X, y = dataset_loader.get_dataset(type_=self.type_)

            X = np.asarray(X)
            y = np.asarray(y)

            max_len = len(X[0])
            for row in X:
                if len(row) > max_len:
                    max_len = len(row)

            X = pad_sequences(X, maxlen=max_len, padding='post')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            num_rows = X[0].shape[0]
            num_columns = X[0].shape[1]
            num_channels = 1

            X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
            X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)

            model = Sequential()
            model.add(
                Conv2D(filters=32, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
            model.add(MaxPooling2D(pool_size=2))
            model.add(Dropout(0.2))

            model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
            model.add(MaxPooling2D(pool_size=2))
            model.add(Dropout(0.2))

            model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
            model.add(MaxPooling2D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(GlobalAveragePooling2D())

            model.add(Dense(self.activation_layer_size, activation='softmax'))

            top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
            top3_acc.__name__ = 'top3_acc'

            # compile the keras model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top3_acc])

            batch_size = 256
            epochs = 400

            lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.0000001)
            mcp_save = ModelCheckpoint(self.checkpoint_filename, save_best_only=True, monitor='val_loss', mode='min')
            cnnhistory = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                   validation_data=(X_test, y_test),
                                   callbacks=[mcp_save, lr_reduce])

            accuracy_list = cnnhistory.history['accuracy']
            highest_index = cnnhistory.history['accuracy'].index(np.sort(cnnhistory.history['accuracy'])[-1])
            scores_ = {'val_accuracy': cnnhistory.history['val_accuracy'][highest_index],
                       'val_top3_acc': cnnhistory.history['val_top3_acc'][highest_index],
                       'accuracy': cnnhistory.history['accuracy'][highest_index],
                       'top3_acc': cnnhistory.history['top3_acc'][highest_index]}

            return scores_

        except:
            return {'val_accuracy': 'error',
                    'val_top3_acc': 'error',
                    'accuracy': 'error',
                    'top3_acc': 'error'}
