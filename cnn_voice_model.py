## CNN MODEL FOR VOICE EMOTION DATABASE
## Coded by: Lucas da Silva Moutinho

# Neural network imports
import numpy as np
import pandas as pd
import keras
import functools
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split # to split dataset into train and test
from sklearn import preprocessing
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
print(X[:3])
print(X.shape)

print(y[:3])
print(y.shape)

# Split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

# input image dimensions
img_rows, img_cols = 3, 4

# # Reshape inputs to 3D-matrices for the convolutional layers
# X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1).astype( 'float32' )
# X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1).astype( 'float32' )

# # See Details
# print("\nX_train:\n")
# print(X_train[:3])
# print(X_train.shape)

# print("\nX_test:\n")
# print(X_test[:3])
# print(X_test.shape)

# print("\ny_train:\n")
# print(y_train[:3])
# print(y_train.shape)

# print("\ny_test:\n")
# print(y_test[:3])
# print(y_test.shape)

# Create categorical matrices
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# See Details
print("\ny_train:\n")
print(y_train[:3])
print(y_train.shape)

print("\ny_test:\n")
print(y_test[:3])
print(y_test.shape)

# define input_shape
input_shape = (img_rows, img_cols, 1)

print(X_train.shape)
X_traincnn = np.expand_dims(X_train, axis=2)
X_testcnn = np.expand_dims(X_test, axis=2)
print(X_traincnn.shape)

# define the keras model
model = Sequential()

model.add(Conv1D(64, 5,padding='same', input_shape=(13,1)))
model.add(Activation('relu'))
model.add(Conv1D(32, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(32, 5,padding='same',))
model.add(Activation('relu'))
# model.add(Conv1D(32, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Conv1D(32, 5,padding='same',))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Conv1D(32, 5,padding='same',))
model.add(Activation('relu'))
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
batch_size = 16
epochs = 200

# Fit model
cnnhistory = model.fit(X_traincnn, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_testcnn, y_test))

# Score Model
score = model.evaluate(X_testcnn, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Model Summary
model.summary()

# PLT History info
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()