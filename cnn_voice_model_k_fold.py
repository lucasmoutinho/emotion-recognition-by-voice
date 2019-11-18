## CNN MODEL FOR VOICE EMOTION DATABASE
## Coded by: Lucas da Silva Moutinho

# Neural network imports
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import functools
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split # to split dataset into train and test
from sklearn import preprocessing
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

import sys
sys.path.append("..")
from scripts.prepare_data import standarization_unit_variance, normalize, load_data_kfold, get_callbacks

DATASET_PATH = "datasets/dataset_48.csv"

# Get dataset
df = pd.read_csv(DATASET_PATH, sep=",")

# cols = df.columns[df.columns.isin(['gender'])]
# df = df[(df[cols] == 1).all(1)] # Only desired gender

# # Create more labels
# df['emotion'] = np.where((df.gender == 1), df.emotion + 6, df.emotion) # distinct labels for man and woman emotions

# # Agroup labels
# df['emotion'] = np.where((df.emotion == 0) | (df.emotion == 5), 7, df.emotion) # positive emotions
# df['emotion'] = np.where((df.emotion == 3), 8, df.emotion) # neutral emotion
# df['emotion'] = np.where((df.emotion == 1) | (df.emotion == 2) | (df.emotion == 4) | (df.emotion == 6), 9, df.emotion) # negative emotions
# df['emotion'] = df['emotion'] - 7

# See dataset details
print(df.head())
print(df.shape)

# Split the dataset in train and test with k-fold
folds, X_train, y_train = load_data_kfold(10, df)

# Create categorical matrices
y_train = to_categorical(y_train)

# See Details
print("\ny_train:\n")
print(y_train[:3])
print(y_train.shape)

# define the keras model
model = Sequential()

model.add(Conv1D(128, 5,padding='same', input_shape=(48,1)))
model.add(Activation('sigmoid'))
model.add(Conv1D(64, 5,padding='same'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(32, 5,padding='same',))
model.add(Activation('sigmoid'))
model.add(Conv1D(32, 5,padding='same',))
model.add(Activation('sigmoid'))
model.add(Flatten())
model.add(Dense(7))
model.add(Activation('softmax'))
# opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

# top-k category accuracy
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top3_acc])

# Define bath and epochs
batch_size = 12
epochs = 100

for jindex , (train_idx, val_idx) in enumerate(folds):
    print('\nFold ',jindex, ' of ', len(folds))
    X_train_cv = X_train.values[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = X_train.values[val_idx]
    y_valid_cv= y_train[val_idx]
    
    name_weights = "models/model_checkpoints/final_model_fold" + str(jindex) + "_weights.h5"
    callbacks = get_callbacks(name_weights, 10)
    
    X_traincnn = np.expand_dims(X_train_cv, axis=2)
    X_validcnn = np.expand_dims(X_valid_cv, axis=2)

    cnnhistory=model.fit(X_traincnn, y_train_cv, batch_size = batch_size, epochs = epochs, validation_data=(X_validcnn, y_valid_cv), callbacks=callbacks)

    # PLT History info
    plt.plot(cnnhistory.history['loss'])
    plt.plot(cnnhistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Model Summary

model.summary()