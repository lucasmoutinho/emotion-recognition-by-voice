import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, MaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
import pandas as pd

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

from sklearn.model_selection import train_test_split # to split dataset into train and test
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


from pandas_profiling import ProfileReport
import sys
sys.path.append('../')
from time_series_dataset_loader import TimeSeriesDatasetLoader

def discretization(X):
    new_dataset = []
    for data_instance in X:
        new_instance = []
        for row in data_instance:
            new_instance.append(np.average(row[2:]))
        new_dataset.append(new_instance)
        
    return np.asarray(new_dataset)

def add_padding(X,y):
    X = np.asarray(X)
    y = np.asarray(y)
    
    max_len = len(X[0])
    for row in X:
        if len(row) > max_len:
            max_len = len(row)

    X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float64')
    return X, y

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

def data_preprocessing(X):
    new_dataset = []
    for data_instance in X:
        new_instance = []
        for row in data_instance:
            new_instance.append(row[2:])
        new_dataset.append(new_instance)
        
    return np.asarray(new_dataset)


DATASET_PATH = '../datasets/Original/MFCC'
dataset_loader = TimeSeriesDatasetLoader(DATASET_PATH)
X, y = dataset_loader.get_dataset(type_='default', ignore_neutral=False)
X = data_preprocessing(X)

# X = discretization(X)

X, y = add_padding(X,y)


X = np.asarray(X)
y = np.asarray(y)

## Reshaping to apply smote
# shape_0 = X.shape[0]
# shape_1 = X.shape[1]
# shape_2 = X.shape[2]
# X = X.reshape(shape_0, shape_1 * shape_2)

# # Apply SMOTE
# smt = SMOTE()
# X, y = smt.fit_sample(X, y)

# # Reshaping back to original shape dimensions 1 and 2
# X = X.reshape(X.shape[0], shape_1, shape_2)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)


# Create categorical matrices
y_test_labels = y_test
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshaping X to fit model
num_rows = X[0].shape[0]
num_columns = X[0].shape[1]
num_channels = 1
##############

X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns,num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns,num_channels)

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
                 activation='relu'))

model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=2,
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2,
                 activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling2D())

model.add(Dense(7, activation='softmax'))

# Compile the keras model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Define bath and epochs
batch_size = 128
epochs = 2

# Callbacks and fitting model
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              patience=20, min_lr=0.0000001)
mcp_save = ModelCheckpoint('model_checkpoints/new_test.h5',
                           save_best_only=True, monitor='val_loss',
                           mode='min')

result = model.fit(X_train, y_train, batch_size=batch_size,
                   epochs=epochs, validation_data=(X_test, y_test),
                   callbacks=[mcp_save, lr_reduce], verbose=2)


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
        
print("average: {}".format(correct/total))


grouped_predictions = group_labels(predictions_labels)
grouped_test_labels = group_labels(y_test_labels)
correct = 0
total = len(grouped_predictions)
for index in range(0, len(grouped_predictions)):
    if grouped_predictions[index] == grouped_test_labels[index]:
        correct += 1
        
print("grouped average: {}".format(correct/total))

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

plt.show()