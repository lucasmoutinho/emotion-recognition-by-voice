import numpy as np
from sklearn import metrics
import pandas as pd

from sklearn.model_selection import train_test_split # to split dataset into train and test
from sklearn import preprocessing
from pandas_profiling import ProfileReport

import sys
from time_series_dataset_loader import TimeSeriesDatasetLoader


def discretization(X):
    new_dataset = []
    for data_instance in X:
        new_instance = []
        for row in data_instance:
            new_instance.append(np.average(row[2:]))
        new_dataset.append(new_instance)
        
    return np.asarray(new_dataset)

DATASET_PATH = '../datasets/Original/MFCC_10'
dataset_loader = TimeSeriesDatasetLoader(DATASET_PATH)
X, y = dataset_loader.get_dataset(type_='default')

X = discretization(X)
y = np.asarray(y)

print("happy profile being done...")
happy_x = np.take(X, np.where(y==0))[0]
profile =ProfileReport( pd.DataFrame(happy_x.tolist()) , title='Alegria Profile')
profile.to_file(output_file="happy_profile.html")

print("des profile being done...")
des_x = np.take(X, np.where(y==1))[0]
profile =ProfileReport( pd.DataFrame(des_x.tolist()) , title='Desdem Profile')
profile.to_file(output_file="des_profile.html")

print("fear profile being done...")
fear_x = np.take(X, np.where(y==2))[0]
profile =ProfileReport( pd.DataFrame(fear_x.tolist()) , title='fear profile')
profile.to_file(output_file="fear_profile.html")

print("neutral profile being done...")
neutral_x = np.take(X, np.where(y==3))[0]
profile =ProfileReport( pd.DataFrame(neutral_x.tolist()) , title='neutral profile')
profile.to_file(output_file="neutral_profile.html")

print("anger profile being done...")
anger_x = np.take(X, np.where(y==4))[0]
profile =ProfileReport( pd.DataFrame(anger_x.tolist()) , title='anger profile')
profile.to_file(output_file="anger_profile.html")

print("surprise profile being done...")
surprise_x = np.take(X, np.where(y==5))[0]
profile =ProfileReport( pd.DataFrame(surprise_x.tolist()) , title='surprise profile')
profile.to_file(output_file="surprise_profile.html")

print("sadness profile being done...")
sadness_x = np.take(X, np.where(y==6))[0]
profile =ProfileReport( pd.DataFrame(sadness_x.tolist()) , title='sadness profile')
profile.to_file(output_file="sadness_profile.html")

print("general profile being done...")
profile =ProfileReport( pd.DataFrame(X.tolist()) , title='general profile')
profile.to_file(output_file="general_profile.html")