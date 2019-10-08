import pandas as pd # to read csv
from sklearn.model_selection import train_test_split # to split dataset into train and test
from sklearn import preprocessing

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras import backend as K


# Read dataset
data = pd.read_csv("dataset.csv")

print(data.head())
print(data.shape)

# X is for features and y is for labels
y = data.emotion
X = data.drop('emotion', axis=1)

# Split the ataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)
print(y_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)
print(y_test.shape)

# Binarize labels
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# DUVIDA: CNN TEM COMO INPUT IMAGENS. PRECISO CONVERTER CSV PARA IMAGEM? SERIA
# NECESSÁRIO PARA PRE-PROCESSING?

# # Building the model
# model = Sequential()
# K.common.image_dim_ordering()
# model.add(Convolution2D(30, 5, 5, border_mode= 'valid' , input_shape=(1, 28, 28),activation= 'relu' ))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(15, 3, 3, activation= 'relu' ))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation= 'relu' ))
# model.add(Dense(50, activation= 'relu' ))
# model.add(Dense(10, activation= 'softmax' ))

# # Compile model
# model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

# # Fit model
# model.fit(X_train, y_train, epochs=20, batch_size= 160)

# # Score model
# score = model.evaluate(X_test, y_test, batch_size=128)

# # Model Summary
# model.summary()