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
from modules.dataset_loader import DatasetLoader
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold

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
            (mfcc_features[index][row_index][2:],
            prosody_features[index][row_index][2:]),
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

X, y = add_padding(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

    
y_test_labels = y_test
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_rows = X[0].shape[0]
num_columns = X[0].shape[1]
num_channels = 1
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns,num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns,num_channels)

run_name = "test_{}.h5".format(datetime.now().strftime("%m%d%H%M"))

batch_size = 128
epochs = 300

model = Sequential()
model.add(Conv2D(filters=128,
                kernel_size=2,
                input_shape=(num_rows, num_columns, num_channels),
                activation='relu'))

model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2,
                activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2,
                activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(7, activation='softmax'))
#test with sigmoid, tanh,
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
                callbacks=[mcp_save, lr_reduce], verbose=2)


validation_acc = np.amax(result.history['val_accuracy'])
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

grouped_predictions = group_labels(predictions_labels)
grouped_test_labels = group_labels(y_test_labels)
correct = 0
total = len(grouped_predictions)
for index in range(0, len(grouped_predictions)):
    if grouped_predictions[index] == grouped_test_labels[index]:
        correct += 1

print("grouped average: {}".format(correct/total))
info_grouped_average = correct/total

emotion_indexes = ['neutro', 'desdem', 'medo', 'alegria', 'raiva', 'surpresa', 'tristeza']

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

for index in range(0, len(emotion_indexes)):
    emotion = emotion_indexes[index]
    correct = cm[index][index]
    wrong = sum(cm[index]) - correct
    print("Average for {}: {}".format(emotion, correct/(correct+wrong)))
    
    
plt.savefig("cm_{}.png".format(run_name))
    
import os
duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

