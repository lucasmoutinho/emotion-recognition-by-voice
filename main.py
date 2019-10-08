import pandas as pd # to read csv
from sklearn.model_selection import train_test_split # to split dataset into train and test

data = pd.read_csv("dataset.csv")

print(data.head())
print(data.shape)

y = data.emotion
X = data.drop('emotion', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)
print(y_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)
print(y_test.shape)