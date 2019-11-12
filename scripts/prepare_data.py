from sklearn import preprocessing
import pandas as pd

# This function returns a normalized version of the input dataframe 
def normalize(df):
    normalized_df = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        normalized_df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return normalized_df

# This function Standarize by removing mean and scaling to unit variance
# It receives X_train and X_test and returns its update values
def standarization_unit_variance(X_train, X_test):
    # Select numerical columns which needs to be standarized
    train_norm = X_train[X_train.columns]
    test_norm = X_test[X_test.columns]

    # Standarize Training Data
    # Use the mean from train to scale test too
    std_scale = preprocessing.StandardScaler().fit(train_norm)
    X_train_norm = std_scale.transform(train_norm)
    X_test_norm = std_scale.transform(test_norm)

    # Converting numpy array to dataframe
    training_norm_col = pd.DataFrame(X_train_norm, index=train_norm.index, columns=train_norm.columns) 
    testing_norm_col = pd.DataFrame(X_test_norm, index=test_norm.index, columns=test_norm.columns) 
    X_train.update(training_norm_col)
    X_test.update(testing_norm_col)

    return X_train, X_test