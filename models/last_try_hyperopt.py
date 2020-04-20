import json
from datetime import datetime
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
    X = np.asarray(X)
    y = np.asarray(y)

    # Apply Pad Sequences
    max_len = len(X[0])
    for row in X:
        if len(row) > max_len:
            max_len = len(row)
    X = pad_sequences(X, maxlen=max_len, padding='post')

    # # Reshaping to apply smote
    # shape_0 = X.shape[0]
    # shape_1 = X.shape[1]
    # shape_2 = X.shape[2]
    # X = X.reshape(shape_0, shape_1 * shape_2)

    # # Apply SMOTE
    # smt = SMOTE()
    # X, y = smt.fit_sample(X, y)

    # # Reshaping back to original shape dimensions 1 and 2
    # X = X.reshape(X.shape[0], shape_1, shape_2)
    
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
    IGNORE_NEUTRAL = False
    y_test_labels = [np.argmax(y_inst) for y_inst in y_test]
    ###
    num_rows = X_train[0].shape[0]
    num_columns = X_train[0].shape[1]
    num_channels = 1
    ####
    # Construct model
    model = Sequential()
    model.add(Conv2D(filters=128,
                    kernel_size=2,
                    input_shape=(num_rows, num_columns, num_channels),
                    activation='tanh'))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2,
                    activation='tanh'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2,
                    activation='tanh'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(7, activation='softmax'))
    #test with sigmoid, tanh,
    # Compile the keras model
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


    # Define bath and epochs
    batch_size = {{choice([64, 128])}}
    epochs = 400

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

    predictions = model.predict(X_test, verbose=1)
    predictions_labels = []
    for predict in predictions:
        predictions_labels.append(predict.tolist().index(np.amax(predict)))


    correct = 0
    total = len(predictions_labels)
    for index in range(0, len(predictions_labels)):
        if predictions_labels[index] == y_test_labels[index]:
            correct += 1
    average = correct/total           
    print("average: {}".format(correct/total))


    grouped_predictions= []
    for value in predictions_labels:
        if value in [3,5]:
            grouped_predictions.append(1)
        else:
            if value == 0 and not IGNORE_NEUTRAL:
                grouped_predictions.append(3)
            else:
                grouped_predictions.append(2)
    grouped_predictions = np.asarray(grouped_predictions)    

    grouped_test_labels= []
    for value in y_test_labels:
        if value in [3,5]:
            grouped_test_labels.append(1)
        else:
            if value == 0 and not IGNORE_NEUTRAL:
                grouped_test_labels.append(3)
            else:
                grouped_test_labels.append(2)
    grouped_test_labels = np.asarray(grouped_test_labels)
    
    correct = 0
    total = len(grouped_predictions)
    for index in range(0, len(grouped_predictions)):
        if grouped_predictions[index] == grouped_test_labels[index]:
            correct += 1
    grouped_average = correct/total
    print("grouped average: {}".format(correct/total))

    run_name = "run " + datetime.now().strftime("%Y-%m-%d-%H-%M")
    info = {
        'run_name': run_name,
        'Dropout': model.layers[2].get_config()['rate'],
        'Dropout 1': model.layers[5].get_config()['rate'],
        'Dropout 2': model.layers[8].get_config()['rate'],
        'batch_size': batch_size,
        'filters': model.layers[0].get_config()['filters'],
        'filters_1': model.layers[3].get_config()['filters'],
        'filters_2': model.layers[6].get_config()['filters'],
        'optimizer': model.optimizer.__class__.__name__,
        'val_accuracy': validation_acc,
        'average': average,
        'grouped_average': grouped_average 
    }

    f = open("last_try_results.log", "a+")
    f.write(str(info) + "\n")
    f.close()
    
    emotion_indexes = ['neutro', 'des', 'medo', 'alegria', 'raiva', 'surpresa', 'tristeza']

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    title = "Confusion Matrix"


    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    cm = confusion_matrix(predictions_labels, y_test_labels)
    df_cm = pd.DataFrame(cm, index=emotion_indexes, columns=emotion_indexes)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)

    # df_cm = pd.DataFrame(array, range(6), range(6))
    # # plt.figure(figsize=(10,7))
    # sn.set(font_scale=1.4) # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    plt.savefig('confusion_matrix/{}.png'.format(run_name))
    
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


def run_2():
    best_run, best_model = optim.minimize(
        model=create_model,
        data=data_2,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials())

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model)

    return best_run, best_model

def run_3():
    best_run, best_model = optim.minimize(
        model=create_model,
        data=data_2,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials())

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model)

    return best_run, best_model

best_run, best_model = run()

