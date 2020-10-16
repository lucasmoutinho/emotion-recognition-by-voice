import json

import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

# DATASET_PATH = '../datasets/Original/MFCC_2/'
# checkpoint_file = 'models/model_checkpoints/original_window_2.h5'
from time_series_dataset_loader import TimeSeriesDatasetLoader


# class TimeSeriesModel:

    # def __init__(self, dataset_path, checkpoint_filename, type_='default',
    #              activation_layer_size=7):
    #     self.dataset_path = dataset_path
    #     self.checkpoint_filename = checkpoint_filename
    #     self.type_ = type_
    #     self.activation_layer_size = activation_layer_size
    #     self.log_file_name = "time_series_model.log"

def data():
    DATASET_PATH = '../datasets/Original/MFCC/'
    checkpoint_file = 'models/model_checkpoints/original_window_2.h5'
    DATASET_TYPE = 'default'

    dataset_path = DATASET_PATH
    type_= DATASET_TYPE
    
    # Importing X and y
    dataset_loader = TimeSeriesDatasetLoader(dataset_path)
    X, y = dataset_loader.get_dataset(type_=type_)
    
    new_dataset = []
    for data_instance in X:
        new_instance = []
        for row in data_instance:
            new_instance.append(row[2:])
        new_dataset.append(new_instance)
        
    X = new_dataset    
    X = np.asarray(X)
    y = np.asarray(y)

    # Apply Pad Sequences
    max_len = len(X[0])
    for row in X:
        if len(row) > max_len:
            max_len = len(row)
    X = pad_sequences(X, maxlen=max_len, padding='post')


    # Reshaping to apply smote
    shape_0 = X.shape[0]
    shape_1 = X.shape[1]
    shape_2 = X.shape[2]
    X = X.reshape(shape_0, shape_1 * shape_2)

    # Apply SMOTE
    smt = SMOTE()
    X, y_train = smt.fit_sample(X, y)

    # Reshaping back to original shape dimensions 1 and 2
    X = X.reshape(X.shape[0], shape_1, shape_2)

    # Split the dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3)
    # Create categorical matrices
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Reshaping X to fit model
    num_rows = X[0].shape[0]
    num_columns = X[0].shape[1]
    num_channels = 1
    ##########################

    X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns,
                              num_channels)
    X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns,
                            num_channels)

    return X_train, y_train, X_test, y_test

def create_model(X_train, y_train, X_test, y_test):
    ###
    num_rows = X_train[0].shape[0]
    num_columns = X_train[0].shape[1]
    num_channels = 1
    ####
    # Construct model
    model = Sequential()
    model.add(Conv2D(filters={{choice([64, 128])}},
                     kernel_size=2,
                     input_shape=(num_rows, num_columns, num_channels),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Conv2D(filters={{choice([64, 128])}}, kernel_size=2,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Conv2D(filters={{choice([32, 64])}}, kernel_size=2,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(3, activation='softmax'))

    # Compile the keras model
    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  metrics=['accuracy'])

    # Define bath and epochs
    batch_size = {{choice([64, 128])}}
    epochs = 200

    # Callbacks and fitting model
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                  patience=20, min_lr=0.0000001)
    mcp_save = ModelCheckpoint('model_checkpoints/hyperopt_test.h5',
                               save_best_only=True, monitor='val_loss',
                               mode='min')
    result = model.fit(X_train, y_train, batch_size=batch_size,
                       epochs=epochs, validation_data=(X_test, y_test),
                       callbacks=[mcp_save, lr_reduce], verbose=2)

    # Get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_accuracy'])
    print('Best validation acc of epoch:', validation_acc)

    accuracy_list = result.history['accuracy']
    highest_index = result.history['accuracy'].index(
        np.sort(result.history['accuracy'])[-1])


    info = {
        'Dropout': model.layers[2].get_config()['rate'],
        'Dropout 1': model.layers[5].get_config()['rate'],
        'Dropout 2': model.layers[8].get_config()['rate'],
        'batch_size': batch_size,
        'filters': model.layers[0].get_config()['filters'],
        'filters_1': model.layers[3].get_config()['filters'],
        'filters_2': model.layers[6].get_config()['filters'],
        'optimizer': model.optimizer.__class__.__name__,
        'val_accuracy': validation_acc,
        'accuracy': result.history['accuracy'][highest_index]
    }

    f = open("result.log", "a+")
    f.write(str(info) + "\n")
    f.close()

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

# def write_log(scores_, log_file_name):


def run():
    best_run, best_model = optim.minimize(
        model=create_model,
        data=data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials())

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model)

    return best_run, best_model

best_run, best_model = run()
