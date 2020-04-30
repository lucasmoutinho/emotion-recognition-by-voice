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
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
import json
from modules.dataset_loader import DatasetLoader
import random
import math


EMOTION_NUMBERS = [0,1,2,3,4,5,6]

def add_padding(X,y):
    X = np.asarray(X)
    y = np.asarray(y)
    
    max_len = len(X[0])
    for row in X:
        if len(row) > max_len:
            max_len = len(row)

    X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float64')
    return X, y

def group_labels_russel(labels):
    new_labels = []
    for value in labels:
        if value == 0:
            new_labels.append(0)
        if value == 3:
            new_labels.append(1)
        if  value == 5:
            new_labels.append(2)
        if value in [1,2,4]:
            new_labels.append(3)
        if value == 6:
            new_labels.append(4)
    
    return np.asarray(new_labels)


def group_labels(labels, ignore_neutral=False):
    new_labels = []
    for value in labels:
        if value in [3,5]:
            new_labels.append(1)
        else:
            if value == 0 and not ignore_neutral:
                new_labels.append(3)
            else:
                new_labels.append(2)
    
    return np.asarray(new_labels)


def load_dataset():
    DATASET_PATH = 'datasets/Original/MFCC/'
    dataset_loader = DatasetLoader(DATASET_PATH)
    mfcc_features, y = dataset_loader.get_dataset()

    DATASET_PATH = 'datasets/Original/Prosody/'
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

    DATASET_PATH = 'datasets/Original/Chroma/'
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

def apply_smote(X, y):
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
    
    return X,y

def wanted_emotion_indexes(wanted_emotion, y_all):
    # selection of indexes of desired emotion
    indexes = []
    for i in range(0,len(y_all)):
        if y_all[i][0] == wanted_emotion:
            indexes.append(i)
    return indexes

def binary_emotion_label(wanted_emotion, y_all):
    for i in range(0,len(y_all)):
        if y_all[i][0] == wanted_emotion:
            y_all[i][0] = 1
        else:
            y_all[i][0] = 0
    return y_all

def load_mock_dataset():
    X = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8]])
    y = [[0,'f1'],[1,'f1'],[2,'f1'],[3,'f1'],[4,'f1'],[5,'f1'],[6,'f1'],[6,'f1'],[2,'f1']]
    return X,y

def run(wanted_emotion, run_name):
    # Load dataset
    X_all,y_all = load_dataset()
    y_list = []
    
    # Create list of indexes for each emotion
    for emotion in EMOTION_NUMBERS:
        current_emotion_indexes = wanted_emotion_indexes(emotion,y_all)
        y_list.append(current_emotion_indexes)
    
    # Create a list of wanted indexes containing all indexes for the desired emotion.
    # That number will be the first half of the list, the other half will contain
    # A slice of random indexes from the other emotions. This will result in a wanted list
    # with half of indexes of the desired emotion and other half with other emotions
    indexes_wanted = y_list[wanted_emotion]
    half_dataset_number = len(y_list[wanted_emotion])
    other_emotions_number = math.ceil(half_dataset_number/(len(EMOTION_NUMBERS)-1))
    for emotion in EMOTION_NUMBERS:
        if(emotion != wanted_emotion):
            indexes_wanted = indexes_wanted + random.sample(y_list[emotion], other_emotions_number)
    
    # Adjust the labels as being the desired emotion or not
    y_all = binary_emotion_label(wanted_emotion, y_all)
    
    # Create X and y
    X = np.take(X_all, indexes_wanted)
    y = [inst_y[0] for inst_y in y_all]
    y = np.take(y, indexes_wanted)
    X = np.asarray(X)
    
    
    X, y = add_padding(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.3)
    
    X_test, y_test = apply_smote(X_test, y_test)
    X_train, y_train = apply_smote(X_train, y_train)
    
    
    y_test_labels = y_test
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    num_rows = X[0].shape[0]
    num_columns = X[0].shape[1]
    num_channels = 1
    X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns,num_channels)
    X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns,num_channels)

    batch_size = 16
    epochs = 300

    model = Sequential()
    model.add(Conv2D(filters=128,
                    kernel_size=2,
                    input_shape=(num_rows, num_columns, num_channels),
                    activation='relu'))

    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2,
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(2, activation='softmax'))
    
    # Compile the keras model
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                        patience=20, min_lr=0.000001)
    mcp_save = ModelCheckpoint(run_name,
                            save_best_only=True, monitor='val_accuracy',
                            mode='max')
    
    result = model.fit(X_train, y_train, batch_size=batch_size,
                epochs=epochs, validation_data=(X_test, y_test),
                callbacks=[mcp_save, lr_reduce], verbose=1)
    
    validation_acc = np.amax(result.history['val_accuracy'])
    
    
    model.load_weights(run_name)
    
    print('Best validation acc of epoch:', validation_acc)
    info_best_validation_acc = validation_acc

    predictions = model.predict(X_test, verbose=1)
    predictions_labels = []
    for predict in predictions:
        predictions_labels.append(predict.tolist().index(np.amax(predict)))

    correct = 0
    total = len(predictions_labels)
    for index in range(0, len(predictions_labels)):
        if predictions_labels[index] == y_test_labels[index]:
            correct += 1

    print("average: {}".format(correct/total))
    info_average = correct/total
    
    return {'run_name': run_name, 'best_validation_acc': info_best_validation_acc, 
            'average': info_average}

def translate_emotion_number(emotion_number):
        return {
            '0' : 'neu',
            '1' : 'des',
            '2' : 'med',
            '3' : 'ale',
            '4' : 'rai',
            '5' : 'sur',
            '6' : 'tri' 
        }[str(emotion_number)]

data = []
for emotion in EMOTION_NUMBERS:
    run_name = translate_emotion_number(emotion)
    data.append(run(emotion,run_name))

print(data)

with open('output.json', 'w+') as outfile:
    json.dump(json.dumps(data), outfile)
