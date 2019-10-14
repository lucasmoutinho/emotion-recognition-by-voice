## CNN MODEL FOR VOICE EMOTION DATABASE
## Coded by: Lucas da Silva Moutinho

# Neural network imports
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split # to split dataset into train and test
from sklearn import preprocessing

# load the dataset
dataset = loadtxt('voice-emotion-database.csv', delimiter=',', skiprows=1)

# See dataset details
print(dataset[:3])
print(dataset.shape)

# split into input (X) and output (y) variables
X = dataset[:,3:15] # Only the MFCC features. Got only 12 features to create 3x4 images
y = dataset[:,19] # Emotion label

# See X and y details
print(X[:3])
print(X.shape)

print(y[:3])
print(y.shape)

# Split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

# input image dimensions
img_rows, img_cols = 3, 4

# Reshape inputs to 3D-matrices for the convolutional layers
X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1).astype( 'float32' )
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1).astype( 'float32' )

# See Details
print("\nX_train:\n")
print(X_train[:3])
print(X_train.shape)

print("\nX_test:\n")
print(X_test[:3])
print(X_test.shape)

print("\ny_train:\n")
print(y_train[:3])
print(y_train.shape)

print("\ny_test:\n")
print(y_test[:3])
print(y_test.shape)

# Binarize labels
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# See Details
print("\ny_train:\n")
print(y_train[:3])
print(y_train.shape)

print("\ny_test:\n")
print(y_test[:3])
print(y_test.shape)

# define input_shape
input_shape = (img_rows, img_cols, 1)

# define the keras model
model = Sequential()
model.add(Conv2D(30, kernel_size=(2, 2),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(15, activation= 'relu' ))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define bath and epochs
batch_size = 64
epochs = 100

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