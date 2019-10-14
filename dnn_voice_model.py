# Neural network imports
from numpy import loadtxt # Linear Algebra
from sklearn import preprocessing
from sklearn.model_selection import train_test_split # to split dataset into train and test
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('voice-emotion-database.csv', delimiter=',', skiprows=1) # Skip the header

# See dataset details
print(dataset[:3])
print(dataset.shape)

# split into input (X) and output (y) variables
X = dataset[:,3:16] # Only the MFCC features
y = dataset[:,19] # Emotion label

# See X and y details
print(X[:3])
print(X.shape)

print(y[:3])
print(y.shape)

# Split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

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

# define the keras model
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu')) #input_dim = number of features. Hidden layer has 50, 20. Output layer has 7 (because of binarize)
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='sigmoid'))

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