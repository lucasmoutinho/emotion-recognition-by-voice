## DNN MODEL FOR VOICE EMOTION DATABASE
## Coded by: Lucas da Silva Moutinho

# Neural network imports
from numpy import loadtxt # Linear Algebra
from sklearn import preprocessing
from sklearn.model_selection import train_test_split # to split dataset into train and test
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pandas as pd
import functools
import keras
from prepare_data import standarization_unit_variance, normalize

# Get dataset
df = pd.read_csv("voice-emotion-database.csv", sep=",")

# See dataset details
print(df.head())
print(df.shape)

# split into input (X) and output (y) variables
X = df[df.columns[3:16]] # Only the MFCC features
y = df.emotion # Emotion label

# Normalization of input features in X
X = normalize(X)

# See X and y details
print("\nX:\n")
print(X.head())
print(X.shape)

print("\ny:\n")
print(y.head())
print(y.shape)

# Split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# See Details
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

print("\ny_train:\n")
print(y_train.head())
print(y_train.shape)

print("\ny_test:\n")
print(y_test.head())
print(y_test.shape)

# # Standarize by removing mean and scaling to unit variance
# X_train, X_test = standarization_unit_variance(X_train, X_test)

# # See details after standarization
# print("\nX_train normalized:\n")
# print (X_train.head())
# print (X_train.shape)

# print("\nX_test normalized:\n")
# print (X_train.head())
# print (X_train.shape)

# # Create categorical matrices
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# See Details
print("\ny_train:\n")
print(y_train[:3])
print(y_train.shape)

print("\ny_test:\n")
print(y_test[:3])
print(y_test.shape)

# define the keras model
model = Sequential()
model.add(Dense(80, input_dim=13, activation='relu')) #input_dim = number of features. Hidden layer has 50, 20. Output layer has 7 (because of binarize)
model.add(Dense(7, activation='softmax'))

# top-k category accuracy

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

top3_acc.__name__ = 'top3_acc'

# compile the keras model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', top3_acc])

# Define bath and epochs
batch_size = 35
epochs = 500

# Fit model
model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test))

# Score Model
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Model Summary
model.summary()

# Saving the model

# Saving the model.json

import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
        json_file.write(model_json)